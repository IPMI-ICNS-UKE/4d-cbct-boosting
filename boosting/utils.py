from pathlib import Path
from typing import Tuple, Optional

import SimpleITK as sitk
import docker
import numpy as np
from docker.errors import ImageNotFound

from boosting.common_types import PathLike, PositiveNumber


def check_docker_image_exists(
    image_name: str, raise_error: bool = False
) -> Optional[bool]:
    docker_client = docker.from_env()
    try:
        docker_client.images.get(image_name)
        return True
    except ImageNotFound:
        if raise_error:
            raise
        else:
            return False


def rescale_range(
    values: np.ndarray,
    input_range: Tuple[float, float],
    output_range: Tuple[float, float],
    clip: bool = True,
):
    in_min, in_max = input_range
    out_min, out_max = output_range
    rescaled = (((values - in_min) * (out_max - out_min)) / (in_max - in_min)) + out_min
    if clip:
        return np.clip(rescaled, out_min, out_max)
    return rescaled


def crop_or_pad(
    image: np.ndarray, target_shape: Tuple[int, ...], pad_value: float = 0.0
) -> Tuple[np.ndarray, Tuple[slice, ...]]:
    valid_content_slicing = [slice(None, None)] * image.ndim

    for i_axis in range(image.ndim):
        if target_shape[i_axis] is not None:
            if image.shape[i_axis] < target_shape[i_axis]:
                # perform padding
                padding = target_shape[i_axis] - image.shape[i_axis]
                padding_left = padding // 2
                padding_right = padding - padding_left

                pad_width = [(0, 0)] * image.ndim
                pad_width[i_axis] = (padding_left, padding_right)
                image = np.pad(
                    image, pad_width, mode="constant", constant_values=pad_value
                )
                valid_content_slicing[i_axis] = slice(padding_left, -padding_right)
            elif image.shape[i_axis] > target_shape[i_axis]:
                # perform cropping
                cropping = image.shape[i_axis] - target_shape[i_axis]
                cropping_left = cropping // 2
                cropping_right = cropping - cropping_left

                cropping_slicing = [slice(None, None)] * image.ndim
                cropping_slicing[i_axis] = slice(cropping_left, -cropping_right)
                image = image[tuple(cropping_slicing)]

    return image, tuple(valid_content_slicing)


def iec61217_to_rsp(image: sitk.Image) -> sitk.Image:
    size = image.GetSize()
    spacing = image.GetSpacing()
    dimension = image.GetDimension()

    if dimension == 3:
        image.SetDirection((1, 0, 0, 0, 0, -1, 0, -1, 0))
        origin = np.subtract(
            image.GetOrigin(),
            image.TransformContinuousIndexToPhysicalPoint(
                (size[0] / 2, size[1] / 2, size[2] / 2)
            ),
        )
        origin = np.add(origin, (spacing[0] / 2, -spacing[1] / 2, -spacing[2] / 2))
    elif dimension == 4:
        image.SetDirection((1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 1))
        origin = np.subtract(
            image.GetOrigin(),
            image.TransformContinuousIndexToPhysicalPoint(
                (size[0] / 2, size[1] / 2, size[2] / 2, size[3] / 2)
            ),
        )
        origin = np.add(
            origin, (spacing[0] / 2, -spacing[1] / 2, -spacing[2] / 2, spacing[0] / 2)
        )
    else:
        ValueError(f"Cannot handle {dimension}D images")

    image.SetOrigin(origin)

    return image


def resample_itk_image(
    image: sitk.Image,
    new_spacing: Tuple[PositiveNumber, PositiveNumber, PositiveNumber],
    resampler=sitk.sitkLinear,
    default_voxel_value=0.0,
):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2]))),
    ]
    resampled_img = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        resampler,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        default_voxel_value,
        image.GetPixelID(),
    )
    return resampled_img


def replace_root(path: PathLike, new_root: PathLike) -> Path:
    path = Path(path)
    new_root = Path(new_root)

    if not new_root.is_absolute():
        raise ValueError("new_root has to be an absolute path")

    return new_root / path.relative_to("/")
