from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional, Tuple, List

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from boosting.common_types import TorchDevice
from boosting.learning.datasets import CBCTBoostingDataset
from boosting.learning.patching import PatchExtractor, PatchStitcher
from boosting.logger import LoggerMixin, tqdm


class CBCTBooster(LoggerMixin):
    def __init__(
        self,
        n_dims: int = 2,
        model: Optional[nn.Module] = None,
        device: TorchDevice = "cuda",
        mixed_precision: bool = True,
        low_vram_mode: bool = False,
    ):
        self.n_dims = n_dims
        self.model = model

        self.device = device
        self.mixed_precision = mixed_precision
        self.low_vram_mode = low_vram_mode
        self.model = self.model.to(self.device)

        self.optimizer = None
        self.loss_function = None
        self.scaler = torch.cuda.amp.GradScaler()
        self.trailing_losses = deque(maxlen=100)

    def _train_one_epoch(
        self, i_epoch: int, data_loader: DataLoader, debug_plot: bool = False
    ):
        try:
            miniters = len(data_loader.dataset) // 10
        except TypeError:
            # dataset has no length, that is the case for, e.g., IterableDataset
            miniters = None

        epoch_losses = []
        self.model.train()
        progress_bar = tqdm(
            total=len(data_loader.dataset),
            logger=self.logger,
            log_level=logging.INFO,
            desc="Train boosting",
            unit="patches",
        )

        for i_batch, data in enumerate(data_loader):
            pseudo_average_image = data["pseudo_average_image"]
            average_image = data["average_image"]

            actual_batch_size = len(average_image)

            pseudo_average_image = torch.as_tensor(
                pseudo_average_image, dtype=torch.float32, device=self.device
            )
            average_image = torch.as_tensor(
                average_image, dtype=torch.float32, device=self.device
            )

            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=self.mixed_precision):
                if self.low_vram_mode:
                    batch_loss = 0.0
                    n_samples_in_batch = len(pseudo_average_image)
                    for i_sample in range(n_samples_in_batch):
                        prediction = self.model(
                            pseudo_average_image[i_sample : i_sample + 1]
                        )
                        sample_loss = self.loss_function(
                            prediction, average_image[i_sample : i_sample + 1]
                        )
                        loss = sample_loss / n_samples_in_batch
                        batch_loss += loss

                        self.scaler.scale(loss).backward()

                else:
                    prediction = self.model(pseudo_average_image)
                    batch_loss = self.loss_function(prediction, average_image)
                    batch_loss_pavg = self.loss_function(
                        pseudo_average_image, average_image
                    )
                    batch_loss = (batch_loss + 1e-6) / (batch_loss_pavg + 1e-6)

            if not self.low_vram_mode:
                self.scaler.scale(batch_loss).backward()

            # self.scaler.unscale_(self.optimizer)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(),
            self.scaler.step(self.optimizer)
            self.scaler.update()

            progress_bar.update(actual_batch_size)

            if debug_plot and i_batch % 50 == 0:
                with plt.ioff():
                    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
                    ax[0].imshow(
                        pseudo_average_image[0, 0].detach().cpu().numpy(),
                        clim=(0.00, 0.03),
                    )
                    ax[1].imshow(
                        average_image[0, 0].detach().cpu().numpy(), clim=(0.00, 0.03)
                    )
                    ax[2].imshow(
                        prediction[0, 0].detach().cpu().numpy(), clim=(0.00, 0.03)
                    )
                    plt.savefig(
                        f"/home/fmadesta/research/4d_cbct_boosting/debug_plots/batch_{i_epoch:02d}_{i_batch:05d}.png",
                        dpi=300,
                    )

            loss = float(batch_loss)
            epoch_losses.append(loss)
            self.trailing_losses.append(loss)
            mean_trailing_loss = np.mean(self.trailing_losses)
            self.scheduler.step(mean_trailing_loss)

            progress_bar.set_postfix(
                {
                    "trailing loss": f"{mean_trailing_loss:.6f}",
                    "lr": self.scheduler._last_lr[0],
                }
            )

        return epoch_losses

    def train(
        self,
        dataset: CBCTBoostingDataset,
        optimizer: optim.Optimizer | None = None,
        scheduler=None,
        loss_function: nn.Module | None = None,
        n_epochs: int = 1,
        batch_size: int | None = None,
        debug_plot: bool = False,
    ) -> List[float]:
        _epoch_string_width = len(str(n_epochs))
        batch_size = batch_size or dataset.n_phases
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if not optimizer:
            optimizer = optim.Adam(
                params=[p for p in self.model.parameters() if p.requires_grad]
            )
            self.logger.debug(
                f"No optimizer passed. Using {optimizer.__class__.__name__}"
            )
        self.optimizer = optimizer
        self.scheduler = scheduler

        if not loss_function:
            loss_function = nn.MSELoss()
            self.logger.debug(
                f"No loss function passed. Using {loss_function.__class__.__name__}"
            )
        self.loss_function = loss_function

        losses = []
        for i in range(n_epochs):
            self.logger.info(
                f"Epoch {str(i + 1):>{_epoch_string_width}s}/"
                f"{str(n_epochs):{_epoch_string_width}s}: started"
            )
            t_start = time.monotonic()
            epoch_losses = self._train_one_epoch(
                i_epoch=i, data_loader=data_loader, debug_plot=debug_plot
            )
            self.logger.info(
                f"Epoch {str(i + 1):>{_epoch_string_width}s}/"
                f"{str(n_epochs):{_epoch_string_width}s}: finisehd. "
                f"Epoch took {time.time() - t_start:.2f} seconds"
            )
            mean_epoch_loss = float(np.mean(epoch_losses))
            self.logger.info(
                f"Epoch {str(i + 1):>{_epoch_string_width}s}/"
                f"{str(n_epochs):{_epoch_string_width}s}: "
                f"mean {self.loss_function!r} = {mean_epoch_loss}"
            )
            losses.append(mean_epoch_loss)

        return losses

    def boost(
        self,
        phase_image: np.ndarray,
        patch_shape: Tuple[int, int] | Tuple[int, int, int],
        phase_image_percentiles: Tuple[float, float],
        average_image_percentiles: Tuple[float, float],
    ) -> sitk.Image:
        self.model.eval()
        image_spatial_shape = phase_image.shape
        image_arr = phase_image
        # image_arr = CBCTBoostingDataset.preprocess(
        #     phase_image,
        #     input_value_range=phase_image_percentiles,
        #     output_value_range=(0, 1),
        # )

        # squeeze 1-sized patch shape axes, e.g. (512, 1, 512) -> (512, 512)
        sqeeze_axes = {i for i, size in enumerate(patch_shape) if size == 1}

        extractor = PatchExtractor(
            patch_shape=patch_shape,
            array_shape=(1, *image_spatial_shape),
            color_axis=0,
            squeeze_patch_axes=sqeeze_axes,
        )
        stitcher = PatchStitcher(array_shape=(1, *image_spatial_shape), color_axis=0)

        image_arr = torch.as_tensor(
            image_arr[None, None], dtype=torch.float32, device=self.device
        )

        slicings = list(
            extractor.extract_ordered(
                stride=tuple(max(1, ps // 2) for ps in patch_shape), flush=True
            )
        )

        for slicing in tqdm(
            slicings,
            logger=self.logger,
            log_level=logging.INFO,
            desc="Boosting image",
            miniters=len(slicings) // 10,
        ):
            self.logger.debug(f"Boost patch {slicing=}")
            with torch.inference_mode():
                prediction = self.model(image_arr[(slice(None),) + slicing])

            # squeeze batch axis (==1)
            prediction = prediction[0].cpu().numpy()
            stitcher.add_patch(prediction, slicing=slicing)

        # calculate mean prediction and squeeze color axis (==1)
        mean_prediction = stitcher.calculate_mean()[0]
        # mean_prediction = CBCTBoostingDataset.preprocess(
        #     mean_prediction,
        #     input_value_range=(0, 1),
        #     output_value_range=average_image_percentiles,
        # )
        # prediction = prediction.squeeze()
        # prediction = sitk.GetImageFromArray(prediction, isVector=False)
        # prediction.CopyInformation(image)

        return mean_prediction
