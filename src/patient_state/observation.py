"""MRI-pretrained ResEnc-L with jointly supervised anatomy and image readouts."""
from pathlib import Path
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder

from .state import PatientState, SEQUENCES, digest
from .anatomy_readouts import readout_contract

FEATURES = (32, 64, 128, 256, 320, 320)


class JointObservation(nn.Module):
    backend = "resenc-l-joint-v1"

    def __init__(self, cavity=False):
        super().__init__()
        self.num_classes = 5 if cavity else 4
        self.backend = "resenc-l-joint-cavity-v2" if cavity else type(self).backend
        network = ResidualEncoderUNet(
            input_channels=8, n_stages=6, features_per_stage=FEATURES, conv_op=nn.Conv3d,
            kernel_sizes=[3]*6, strides=[1, 2, 2, 2, 2, 2],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6], num_classes=self.num_classes,
            n_conv_per_stage_decoder=[1]*5, conv_bias=True,
            norm_op=nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU, nonlin_kwargs={"negative_slope": 1e-2, "inplace": True})
        self.encoder = network.encoder
        self.segmentation_decoder = network.decoder
        self.image_decoder = UNetDecoder(self.encoder, num_classes=4,
                                         n_conv_per_stage=[1]*5, deep_supervision=False)
        self.model_id = "unselected-training-model"

    def initialize_pretrained(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        weights = checkpoint["network_weights"]
        plan = checkpoint["nnssl_adaptation_plan"]
        if plan["architecture_plans"]["arch_class_name"] != "ResEncL":
            raise ValueError("Expected MRI-pretrained ResEnc-L")
        adapted = {}
        for key, target in self.encoder.state_dict().items():
            source = weights["encoder." + key]
            if source.shape != target.shape:
                if source.ndim != 5 or source.shape[1] != 1 or target.shape[1] != 8 or not key.startswith("stem."):
                    raise ValueError("Unexpected pretrained tensor mismatch: " + key)
                # Official adaptation repeats/averages image weights. New availability
                # channels start at zero, preserving the pretrained image response.
                value = torch.zeros_like(target)
                value[:, :4] = source.repeat(1, 4, 1, 1, 1) / 4
                source = value
            adapted[key] = source
        self.encoder.load_state_dict(adapted, strict=True)
        return {"checkpoint_sha256": digest(path), "encoder_tensors_loaded": len(adapted),
                "pretraining": "OpenMind ResEnc-L MAE", "input_adaptation": "four averaged image channels; four zero-initialized availability channels",
                "citations": checkpoint.get("citations", [])}

    def initialize_observation(self, path):
        """Expand the selected four-class baseline without changing existing channels."""
        value = torch.load(path, map_location="cpu", weights_only=False)
        if self.num_classes != 5 or value.get("backend") != type(self).backend:
            raise ValueError("Cavity adaptation requires the original four-class observation model")
        weights = value["model"]
        adapted = {}
        for name, target in self.state_dict().items():
            source = weights[name]
            if source.shape != target.shape:
                if "segmentation_decoder.seg_layers." not in name or source.shape[0] != 4 or target.shape[0] != 5:
                    raise ValueError("Unexpected observation adaptation: " + name)
                expanded = target.clone()
                expanded[:4] = source
                expanded[4] = source[2]
                if name.endswith("bias"):
                    expanded[4] -= 2.3
                source = expanded
            adapted[name] = source
        self.load_state_dict(adapted, strict=True)
        return {"checkpoint_sha256": digest(path), "initialization": "preserved joint observation",
                "cavity_initialization": "copy core head with bias -2.3; existing four channels unchanged"}

    def features(self, images, availability):
        if images.ndim != 5 or images.shape[1] != 4 or availability.shape != images.shape[:2]:
            raise ValueError("Expected four sequences and explicit availability")
        if not availability.bool().any(1).all():
            raise ValueError("At least one sequence is required")
        a = availability.to(images.dtype)[..., None, None, None]
        return self.encoder(torch.cat((images.masked_fill(~availability.bool()[..., None, None, None], 0), a.expand_as(images)), dim=1))

    def decode_features(self, features):
        return self.segmentation_decoder(features), self.image_decoder(features)

    def forward(self, images, availability):
        return self.decode_features(self.features(images, availability))

    @torch.inference_mode()
    def encode(self, images, metadata):
        self.eval()
        shape = tuple(images.shape[2:])
        padding = tuple(v for size in reversed(shape) for v in (0, (-size) % 32))
        x = F.pad(images, padding)
        availability = torch.tensor([metadata["availability"]], device=x.device)
        metadata = dict(metadata, backend=self.backend, model_id=self.model_id,
                        shape_xyz=list(shape), anatomical_classes=self.num_classes,
                        anatomy_readouts=readout_contract(self.num_classes, metadata.get("dataset")),
                        cavity_status="predicted; annotation availability is dataset-specific" if self.num_classes == 5 else "not modeled",
                        uncertainty_status="observation uncertainty not fitted")
        if max(x.shape[2:]) <= 128:
            with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
                features = self.features(x, availability)
            return PatientState({f"feature{i}": f for i, f in enumerate(features)}, metadata)
        # Match patch-training normalization at inference. Each patch's complete
        # feature pyramid belongs to the saved state; decoding never re-reads MRI.
        axes = [list(range(0, s-128+1, 96)) for s in x.shape[2:]]
        for positions, s in zip(axes, x.shape[2:]):
            if positions[-1] != s-128:
                positions.append(s-128)
        patches, stored = [], {}
        for n, origin in enumerate(product(*axes)):
            slices = (slice(None), slice(None), *(slice(s, s+128) for s in origin))
            with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
                features = self.features(x[slices], availability)
            prefix = f"patch{n:03d}_"
            stored.update({prefix+f"feature{i}": f.cpu() for i, f in enumerate(features)})
            patches.append({"prefix": prefix, "origin": list(origin), "size": 128})
        metadata.update(patches=patches, padded_shape=list(x.shape[2:]),
                        inference="overlapping 128-cube complete states, weighted decoded blending")
        return PatientState(stored, metadata)

    @torch.inference_mode()
    def read(self, state):
        if state.metadata["model_id"] != self.model_id:
            raise ValueError("State and readout identities differ")
        if "patches" in state.metadata:
            shape = state.metadata["padded_shape"]
            sums = [torch.zeros((1, channels, *shape), dtype=torch.float32) for channels in (self.num_classes, 4)]
            weights = torch.zeros((1, 1, *shape), dtype=torch.float32)
            axis = (1-torch.linspace(-1, 1, 128).abs()).clamp_min(.05)
            window = (axis[:, None, None]*axis[None, :, None]*axis[None, None, :])[None, None]
            device = next(self.parameters()).device
            for patch in state.metadata["patches"]:
                features = [state.features[patch["prefix"]+f"feature{i}"].to(device) for i in range(6)]
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    outputs = self.decode_features(features)
                slices = (slice(None), slice(None), *(slice(s, s+128) for s in patch["origin"]))
                for total, output in zip(sums, outputs):
                    total[slices] += output.float().cpu()*window
                weights[slices] += window
            slices = (slice(None), slice(None), *(slice(0, s) for s in state.metadata["shape_xyz"]))
            return {"logits": (sums[0]/weights)[slices], "images": (sums[1]/weights)[slices],
                    "image_sequences": list(SEQUENCES)}
        device = next(self.parameters()).device
        features = [state.features[f"feature{i}"].to(device) for i in range(6)]
        with torch.autocast(features[0].device.type, dtype=torch.bfloat16, enabled=features[0].is_cuda):
            logits, images = self.decode_features(features)
        slices = (slice(None), slice(None), *(slice(0, s) for s in state.metadata["shape_xyz"]))
        return {"logits": logits[slices], "images": images[slices], "image_sequences": list(SEQUENCES)}

    @classmethod
    def from_checkpoint(cls, path, device="cuda"):
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("backend") not in (cls.backend, "resenc-l-joint-cavity-v2"):
            raise ValueError("Not a joint observation checkpoint")
        model = cls(cavity=value["backend"] == "resenc-l-joint-cavity-v2")
        model.load_state_dict(value["model"], strict=True)
        model.model_id = digest(path)
        model.is_smoke = value.get("config", {}).get("smoke", False)
        return model.to(device).eval().requires_grad_(False)
