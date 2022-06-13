import subprocess
from pathlib import Path
from typing import Generator, Tuple

import numpy as np


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
):
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


def to_path(*args) -> Generator:
    for arg in args:
        if arg:
            yield Path(arg)
        else:
            yield None


def filter_kwargs(valid_keys, **kwargs):
    return {k: v for k, v in kwargs.items() if k in valid_keys}


def create_bin_call(base_call, prefix="--", **kwargs):
    bc = list(base_call)
    for key, value in kwargs.items():
        if value is None:
            # skip None values
            continue
        if key.endswith("_"):
            key = key[:-1]
        bc.append(f"{prefix}{key}")
        if isinstance(value, list) or isinstance(value, tuple):
            bc.append(",".join([str(i) for i in value]))
        else:
            bc.append(str(value))

    return bc


def call(bin_call):
    return subprocess.check_output(bin_call)


def iec61217_to_rsp(image):
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

    image.SetOrigin(origin)

    return image
