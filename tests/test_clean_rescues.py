"""Scientific contract tests for the clean-data rescue experiments."""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np
import torch

from src.harness.analysis.clean_rescues import (aligned_prediction, alignment,
                                               checked_cache, fit_gain)
from src.harness.data.anatomy_manifest import lumiere_visits
from src.harness.train.rescues import BASIN, OBJECTIVES, arm_config
from src.model.lesion_forecaster import LesionTransitionForecaster
from src.harness.train.lesion import pair_validity


def main():
    torch.set_num_threads(2)
    assert pair_validity([0, 1, 3, 4], torch.tensor([0., 1., 10., 0.])).tolist() == [True, False, False]
    # Make the residual nonzero so zero initialization cannot hide leakage.
    torch.manual_seed(42)
    model = LesionTransitionForecaster(input_dim=8, observation_dim=4, hidden_dim=4).eval()
    torch.nn.init.normal_(model.delta.weight)
    features = torch.randn(6, 8, requires_grad=True)
    gaps = torch.tensor([0., 10., 20., 30., 40., 50.], requires_grad=True)
    volumes = torch.randn(6, 3, requires_grad=True)
    prediction = model(features, gaps, volumes)["prediction"]
    for cutoff in range(5):
        changed = features.detach().clone(); changed[cutoff + 1:] += 100
        changed_gaps = gaps.detach().clone(); changed_gaps[cutoff + 1:] += 1000
        changed_volumes = volumes.detach().clone(); changed_volumes[cutoff + 1:] += 100
        actual = model(changed, changed_gaps, changed_volumes)["prediction"]
        assert torch.allclose(actual[:cutoff + 1], prediction[:cutoff + 1], atol=1e-6)
        prefix = model(features[:cutoff + 2], gaps[:cutoff + 2], volumes[:cutoff + 2])["prediction"]
        assert torch.allclose(prefix, prediction[:cutoff + 1], atol=1e-6)
        gradients = torch.autograd.grad(prediction[cutoff].sum(), (features, gaps, volumes), retain_graph=True)
        assert all(torch.count_nonzero(gradient[cutoff + 1:]) == 0 for gradient in gradients)
    # Query futures must never enter a support-fitted gain.
    support = {"source": torch.tensor([[1., 0.], [1., 0.]]),
               "raw": torch.tensor([[1., 1.], [1., 1.]]),
               "target": torch.tensor([[1., .25], [1., .25]]),
               "groups": np.array(["a", "b"])}
    assert abs(fit_gain(support) - .25) < .02
    # Inverting alignment must preserve the native target geometry, including
    # a shifted/scaled site. Scoring aligned targets would change the floor.
    native = torch.randn(40, 3)
    ref = native * 2 + 5
    mapping = alignment(native, ref, "diagonal")
    rows = {"states": native[:4], "target": native[4:8]}
    targets = rows["target"].clone()
    recovered = aligned_prediction(torch.nn.Identity(), rows, mapping, mapping)
    assert torch.allclose(recovered, native[:4], atol=1e-5)
    assert torch.equal(targets, rows["target"])
    base = {"data": {"include_survival": False}, "training": {}, "aux": {}}
    for arm in (*BASIN, *OBJECTIVES):
        cfg = arm_config(base, arm, "output")
        assert cfg["data"]["include_survival"] is False
        assert cfg["training"]["max_epochs"] == (6 if arm in BASIN else 30)
        assert sum([cfg["data"]["augment_train"], cfg["training"]["surgery_window"],
                    cfg["training"]["transition_weighting"]]) == (0 if arm in BASIN else 1)
    with tempfile.TemporaryDirectory() as td:
        # Bare/legacy caches cannot masquerade as the cleaned rerun inputs.
        path = Path(td) / "legacy.pt"; torch.save({"patients": {}}, path)
        try: checked_cache(path, "clean.pt")
        except ValueError: pass
        else: raise AssertionError("legacy cache accepted")
        masks = Path(td) / "masks"; images = Path(td) / "images"
        rel = Path("Patient-001/week-001")
        dest = masks / rel / "DeepBraTumIA-segmentation/atlas/segmentation"
        dest.mkdir(parents=True); (images / rel).mkdir(parents=True)
        data = np.zeros((3, 3, 3), dtype=np.uint8); data[0, 0, 0] = 1
        nib.save(nib.Nifti1Image(data, np.diag([2., 3., 4., 1.])), dest / "seg_mask.nii")
        visits = lumiere_visits(str(images), str(Path(td) / "absent"), str(masks))
        assert visits[0]["measurement"]["geometry_verified"]
        assert abs(visits[0]["measurement"]["values_mm3"]["enhancing"] - 24) < 1e-6
    print("clean rescue contracts passed")


if __name__ == "__main__": main()
