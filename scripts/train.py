import logging
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from boosting.booster import CBCTBooster
from boosting.learning.datasets import CBCTBoostingDataset
from boosting.learning.losses import BoostingLoss
from boosting.learning.models import FlexUNet
from boosting.logger import init_fancy_logging

init_fancy_logging()
import matplotlib.pyplot as plt

logging.getLogger(__name__).setLevel(logging.INFO)
logging.getLogger("boosting").setLevel(logging.INFO)
np.random.seed(1337)

# this is the data folder containing the reconstructions created by reconstruct.py
DATA_FOLDER = Path("/datalake_fast/4d_cbct_boosting/phantom_scan/reconstructions")

# patch shape of (a, b==1, c): randomly extracted axial 2D patches of shape (a, c)
# patch shape of (a, b>1, c): randomly extracted 3D patches of shape (a, b, c)
# both training and inference/boosting is implemented for 2D and 3D
TRAINING_PATCH_SHAPE = (384, 1, 384)
INFERENCE_PATCH_SHAPE = (384, 1, 384)
RESIDUAL_ALPHA = 0.9

n_dimensions = 2 if any(ps == 1 for ps in TRAINING_PATCH_SHAPE) else 3

dataset = CBCTBoostingDataset(patch_shape=TRAINING_PATCH_SHAPE, n_patches_per_image=100)
dataset.add_patient_from_filepath(
    "pat1",
    pseudo_average_image=DATA_FOLDER / "rooster4d_pseudo_average.mha",
    average_image=DATA_FOLDER / "fdk3d.mha",
)

if n_dimensions == 2:
    encoder_filters = [32, 16, 8, 4]
    decoder_filters = [4, 8, 16, 32]
    model = FlexUNet(
        n_input_channels=1,
        n_output_channels=1,
        n_levels=4,
        n_filters=[32, *encoder_filters, *decoder_filters, 32],
        convolution_layer=nn.Conv2d,
        downsampling_layer=nn.AvgPool2d,
        upsampling_layer=nn.Upsample,
        norm_layer=None,
        skip_connections=True,
        convolution_kwargs=None,
        downsampling_kwargs={"kernel_size": 2},
        upsampling_kwargs={"scale_factor": 2, "mode": "bilinear"},
        return_bottleneck=False,
        residual_alpha=RESIDUAL_ALPHA,
    )
else:
    encoder_filters = [32, 16, 8, 4]
    decoder_filters = [4, 8, 16, 32]
    model = FlexUNet(
        n_input_channels=1,
        n_output_channels=1,
        n_levels=4,
        n_filters=[32, *encoder_filters, *decoder_filters, 32],
        convolution_layer=nn.Conv3d,
        downsampling_layer=nn.AvgPool3d,
        upsampling_layer=nn.Upsample,
        norm_layer=None,
        skip_connections=True,
        convolution_kwargs=None,
        downsampling_kwargs={"kernel_size": 2},
        upsampling_kwargs={"scale_factor": 2, "mode": "trilinear"},
        return_bottleneck=False,
        residual_alpha=RESIDUAL_ALPHA,
    )

booster = CBCTBooster(
    n_dims=n_dimensions,
    model=model,
    device="cuda:1",
    low_vram_mode=False,
    mixed_precision=True,
)
optimizer = Adam(
    lr=1e-3, params=[p for p in booster.model.parameters() if p.requires_grad]
)

scheduler = ReduceLROnPlateau(
    optimizer,
    "min",
    factor=0.9,
    threshold=0.001,
    threshold_mode="rel",
    patience=200,
    cooldown=200,
)
loss_function = BoostingLoss(dt_regularization=0.00, gradient_attention=0.0)
losses = booster.train(
    dataset,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_function=loss_function,
    batch_size=10,
    n_epochs=200,
    debug_plot=False,
)


# after training, you can boost the 4D CBCT phase images
# Note:
# you can apply the model to the patient you trained with
# since there is no ground truth you can overfit to anyway.
# In general, the best strategy is to train a boosting model on all available
# 4D CBCT patients and fine-tune for a few iterations on the patient
# the boosting model should be applied to.
phase_images = sitk.ReadImage(str(DATA_FOLDER / "rooster4d_phase.mha"))

# save image properties
image_origin = phase_images.GetOrigin()
image_spacing = phase_images.GetSpacing()
image_direction = phase_images.GetDirection()

phase_images = sitk.GetArrayFromImage(phase_images)
boosted_phase_images = np.zeros_like(phase_images)
fig, ax = plt.subplots(2, 10, sharex=True, sharey=True)
for i, phase_image in enumerate(phase_images):
    boosted = booster.boost(
        phase_image=phase_image,
        patch_shape=INFERENCE_PATCH_SHAPE,
        phase_image_percentiles=dataset.percentiles["approx_global"][
            "pseudo_average_image"
        ],
        average_image_percentiles=dataset.percentiles["approx_global"]["average_image"],
    )
    boosted_phase_images[i] = boosted

    ax[0, i].imshow(phase_image[:, 75, :], clim=(0, 0.025))
    ax[1, i].imshow(boosted[:, 75, :], clim=(0, 0.025))

boosted_phase_images = sitk.GetImageFromArray(boosted_phase_images, isVector=False)
boosted_phase_images.SetOrigin(image_origin)
boosted_phase_images.SetSpacing(image_spacing)
boosted_phase_images.SetDirection(image_direction)

sitk.WriteImage(boosted_phase_images, str(DATA_FOLDER / "rooster4d_phase_boosted.mha"))
