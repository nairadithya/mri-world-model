"""N4 bias field correction via SimpleITK."""
from __future__ import annotations

import SimpleITK as sitk


def n4_bias_correct(
    image: sitk.Image,
    mask: sitk.Image | None = None,
    shrink_factor: int = 2,
    num_fitting_levels: int = 4,
) -> sitk.Image:
    """Run N4 bias field correction on a SimpleITK image.

    Args:
        image: input image (any scalar pixel type; cast to Float32 internally).
        mask: optional Otsu-derived or brain mask. If None, an Otsu mask is built.
        shrink_factor: downsampling factor for speed.
        num_fitting_levels: N4 fitting levels.
    """
    img_f = sitk.Cast(image, sitk.sitkFloat32)
    if mask is None:
        mask = sitk.OtsuThreshold(img_f, 0, 1, 200)
    shrinker = sitk.ShrinkImageFilter()
    shrinker.SetShrinkFactors([shrink_factor] * img_f.GetDimension())
    small = shrinker.Execute(img_f)
    small_mask = shrinker.Execute(mask)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50] * num_fitting_levels)
    corrected_small = corrector.Execute(small, small_mask)
    log_bias = corrector.GetLogBiasFieldAsImage(small)
    full_bias = sitk.Exp(
        sitk.Resample(
            log_bias,
            img_f,
            sitk.Transform(),
            sitk.sitkLinear,
        )
    )
    corrected = sitk.Divide(img_f, full_bias)
    return corrected
