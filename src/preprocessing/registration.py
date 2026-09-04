"""Rigid registration to a 1mm-isotropic template via SimpleITK."""
from __future__ import annotations

import SimpleITK as sitk


def resample_to_iso(
    image: sitk.Image,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolator=sitk.sitkLinear,
) -> sitk.Image:
    """Resample an image to isotropic spacing, preserving origin/direction."""
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(orig_size, orig_spacing, spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform(image.GetDimension(), sitk.sitkIdentity))
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def rigid_register_to_template(
    moving: sitk.Image,
    fixed: sitk.Image,
    num_iterations: int = 200,
) -> tuple[sitk.Image, sitk.Transform]:
    """Rigid (Euler3D) registration of moving -> fixed with Mattes MI.

    Returns (resampled_moving, final_transform).
    """
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)

    initial = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.05)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=num_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial, inPlace=False)
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])

    final_transform = registration.Execute(fixed_f, moving_f)
    resampled = sitk.Resample(
        moving_f, fixed_f, final_transform, sitk.sitkLinear, 0.0, moving_f.GetPixelID()
    )
    return resampled, final_transform
