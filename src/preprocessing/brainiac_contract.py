"""BRAINIAC preprocessing contract: N4 -> 1mm iso -> rigid MNI -> HD-BET.

Offline entry point used by scripts/preprocess.py. Each step is idempotent
and writes an intermediate file so long runs can resume.
"""
from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from .n4_bias import n4_bias_correct
from .registration import resample_to_iso, rigid_register_to_template
from .skull_strip import hd_bet_skull_strip


def preprocess_sequence(
    input_path: str,
    output_path: str,
    template_path: str | None = None,
    device: str = "cpu",
    work_dir: str | None = None,
    target_size: tuple[int, int, int] = (96, 96, 96),
) -> str:
    """Run the full BRAINIAC contract on one sequence NIfTI.

    Steps:
      1. N4 bias correction
      2. Resample to 1mm isotropic
      3. Rigid registration to template (if template_path given)
      4. HD-BET skull-strip
      5. Resize to 96^3 + z-score normalize (nonzero)

    Returns the output path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp = work_dir or os.path.join(os.path.dirname(output_path), "_tmp")
    os.makedirs(tmp, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0].replace(".nii", "")

    img = sitk.ReadImage(input_path)

    # 1. N4
    corrected = n4_bias_correct(img)
    p1 = os.path.join(tmp, f"{base}_n4.nii.gz")
    sitk.WriteImage(corrected, p1)

    # 2. 1mm iso
    iso = resample_to_iso(corrected, spacing=(1.0, 1.0, 1.0))
    p2 = os.path.join(tmp, f"{base}_iso.nii.gz")
    sitk.WriteImage(iso, p2)

    # 3. rigid registration (optional)
    reg_in = p2
    if template_path is not None:
        template = sitk.ReadImage(template_path)
        reg_img, _ = rigid_register_to_template(iso, template)
        p3 = os.path.join(tmp, f"{base}_reg.nii.gz")
        sitk.WriteImage(reg_img, p3)
        reg_in = p3

    # 4. skull-strip
    p4 = os.path.join(tmp, f"{base}_bet.nii.gz")
    hd_bet_skull_strip(reg_in, p4, device=device)

    # 5. resize + z-score, write final
    nii = nib.load(p4)
    data = nii.get_fdata(dtype=np.float32)
    t = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    t = F.interpolate(t, size=target_size, mode="trilinear", align_corners=False).squeeze(0)
    mask = t != 0
    if mask.sum() > 0:
        vals = t[mask]
        t[mask] = (t[mask] - vals.mean()) / vals.std().clamp_min(1e-6)
    out = nib.Nifti1Image(t.squeeze(0).numpy().astype(np.float32), np.eye(4))
    nib.save(out, output_path)
    return output_path
