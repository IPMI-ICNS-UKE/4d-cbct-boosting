from pathlib import Path
import SimpleITK as sitk
from boosting.common_types import PathLike
import torch
from typing import Tuple
from boosting.learning.datasets import CBCTBoostingDataset
import numpy as np
import torch.nn as nn

def boost_image(
    filepath: PathLike, voxel_value_percentiles: Tuple[float, float], model: nn.Module
):
    image = sitk.ReadImage(str(filepath), sitk.sitkFloat32)
    image = sitk.GetArrayFromImage(image)

    for i_phase, phase_image in enumerate(image):

        phase_image = CBCTBoostingDataset.preprocess(
            phase_image,
            target_image_shape=phase_image.shape,
            input_value_range=voxel_value_percentiles,
            output_value_range=(0.0, 1.0),
        )

        predictions = []
        for i_coronal in range(phase_image.shape[1]):
            coronal_slice = phase_image[np.newaxis, np.newaxis, :, i_coronal]
            coronal_slice = torch.as_tensor(
                coronal_slice, dtype=torch.float32, device='cuda'
            )

            prediction = model(coronal_slice)
            prediction = prediction.detach().cpu().numpy()
            prediction = prediction.squeeze(axis=(0, 1))
            prediction = prediction[:, np.newaxis, :]
        predictions.append(prediction)


        return predictions


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = torch.load("/home/fmadesta/research/4d_cbct_boosting/model.pt")

    p = boost_image(
        "/datalake/4d_cbct_lmu/Hamburg/CBCT4D.mhd",
        voxel_value_percentiles=(0.0, 0.01794880257919429),
        model=model
    )

