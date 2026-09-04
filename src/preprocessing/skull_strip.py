"""HD-BET skull-stripping wrapper with graceful fallback."""
from __future__ import annotations

import os
import subprocess
import tempfile

import nibabel as nib
import numpy as np


def hd_bet_skull_strip(
    input_path: str,
    output_path: str,
    device: str = "cpu",
    mode: str = "fast",
) -> str:
    """Run HD-BET on a NIfTI file.

    Calls `hd-bet -i <in> -o <out>` as a subprocess so the heavy optional
    dependency stays out of the import path. Falls back to a simple
    intensity-threshold mask if HD-BET is not installed.

    Returns the output path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    try:
        subprocess.run(
            ["hd-bet", "-i", input_path, "-o", output_path,
             "-device", device, "-mode", mode, "-tta", "0"],
            check=True,
            capture_output=True,
        )
        return output_path
    except (FileNotFoundError, subprocess.CalledProcessError):
        return _threshold_fallback(input_path, output_path)


def _threshold_fallback(input_path: str, output_path: str) -> str:
    """Crude brain mask: keep voxels above the 10th percentile of nonzero."""
    img = nib.load(input_path)
    data = img.get_fdata(dtype=np.float32)
    nz = data[data > 0]
    thr = float(np.percentile(nz, 10)) if nz.size else 0.0
    masked = np.where(data > thr, data, 0.0).astype(np.float32)
    nib.save(nib.Nifti1Image(masked, img.affine, img.header), output_path)
    return output_path


def apply_brain_mask(image_path: str, mask_path: str, output_path: str) -> str:
    """Multiply an image by a binary brain mask."""
    img = nib.load(image_path)
    mask = nib.load(mask_path).get_fdata() > 0
    data = img.get_fdata(dtype=np.float32) * mask.astype(np.float32)
    nib.save(
        nib.Nifti1Image(data.astype(np.float32), img.affine, img.header),
        output_path,
    )
    return output_path
