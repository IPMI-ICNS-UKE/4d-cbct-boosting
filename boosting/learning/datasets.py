from __future__ import annotations

import logging
import pickle
from typing import Union, Tuple, List

import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset

from boosting.common_types import PathLike
from boosting.learning.patching import PatchExtractor
from boosting.logger import init_fancy_logging
from boosting.utils import rescale_range, crop_or_pad

logger = logging.getLogger(__name__)


class CBCTBoostingDataset(Dataset):
    def __init__(
        self,
        patch_shape: Tuple[int, int, int] = (512, 1, 512),
        n_patches_per_image: int = 10,
        n_phases: int = 10,
    ):
        self.patch_shape = patch_shape
        self.n_patches_per_image = n_patches_per_image
        self.n_phases = n_phases
        self._patches: List[dict] = []

        self.patch_extractor: PatchExtractor | None = None

        self.patients = {}
        self.percentiles = None

    def calculate_percentiles(self, lower: int = 0.1, upper: int = 99.9):
        logger.debug(f"Calculating {lower=}/{upper=} percentiles")
        percentiles = {}
        accumulated = {}
        for patient_id, patient in self.patients.items():
            percentiles[patient_id] = {}
            for image_name, image in patient.items():
                lo, up = np.percentile(image, (lower, upper))
                percentiles[patient_id][image_name] = (lo, up)
                try:
                    accumulated[image_name].append((lo, up))
                except KeyError:
                    accumulated[image_name] = [(lo, up)]
        percentiles["approx_global"] = {}

        for image_name, all_percentiles in accumulated.items():
            mean_percentile = tuple(np.mean(all_percentiles, axis=0))
            percentiles["approx_global"][image_name] = mean_percentile

        self.percentiles = percentiles

        logger.info(
            f"Dataset {lower=}/{upper=} percentiles are "
            f"{percentiles['approx_global']}. Use these values for boosting/inference."
        )

    @staticmethod
    def preprocess(
        image: np.ndarray,
        target_image_shape: Tuple[Union[None, int], ...] = None,
        input_value_range: Tuple[float, float] = None,
        output_value_range: Tuple[float, float] = None,
    ):
        if input_value_range is not None and output_value_range is not None:
            image = rescale_range(
                image,
                input_range=input_value_range,
                output_range=output_value_range,
                clip=True,
            )

        if target_image_shape:
            image, _ = crop_or_pad(image, target_shape=target_image_shape)

        return image

    def add_patient(
        self,
        patient_id: Union[str, int],
        pseudo_average_image: np.ndarray,
        average_image: np.ndarray,
    ):
        if pseudo_average_image.ndim != 4:
            raise RuntimeError(
                "Pseudo average image must be a 4D image, i.e., (time, x, y, z)"
            )
        if average_image.ndim != 3:
            raise RuntimeError("Average image must be a 3D image, i.e., (x, y, z)")

        # add color channels and add to dict
        self.patients[patient_id] = {
            "pseudo_average_image": pseudo_average_image[:, None],
            "average_image": average_image[None],
        }

    def add_patient_from_filepath(
        self,
        patient_id: Union[str, int],
        pseudo_average_image: PathLike,
        average_image: PathLike,
    ):
        pseudo_average_image = sitk.ReadImage(
            str(pseudo_average_image), sitk.sitkFloat32
        )
        average_image = sitk.ReadImage(str(average_image), sitk.sitkFloat32)

        # convert to numpy
        pseudo_average_image = sitk.GetArrayFromImage(pseudo_average_image)
        average_image = sitk.GetArrayFromImage(average_image)

        # swap axes for ITK -> numpy
        # pseudo_average_image = np.swapaxes(pseudo_average_image, 0, 2)
        # average_image = np.swapaxes(average_image, 0, 2)

        self.add_patient(
            patient_id=patient_id,
            pseudo_average_image=pseudo_average_image,
            average_image=average_image,
        )

    @classmethod
    def load(cls, filepath: PathLike):
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        dataset = cls()
        dataset.__dict__ = state

        return dataset

    def save(self, filepath: PathLike):
        with open(filepath, "wb") as f:
            pickle.dump(self.__dict__, f)

    def compile_patch_slicings(self):
        self._patches = []
        for patient_id, patient in self.patients.items():
            average_image = patient["average_image"]
            logger.debug(f"Create patch slicings for {patient_id=}")

            image_shape = average_image.shape

            # squeeze 1-sized patch shape axes, e.g. (512, 1, 512) -> (512, 512)
            sqeeze_axes = {i for i, size in enumerate(self.patch_shape) if size == 1}

            extractor = PatchExtractor(
                patch_shape=self.patch_shape,
                array_shape=image_shape,
                squeeze_patch_axes=sqeeze_axes,
            )

            proba_map = average_image - average_image.min()
            # proba map has no color axis
            proba_map = proba_map[0]
            # roi = np.zeros_like(proba_map)
            # roi[:, 125:126, :] = 1
            # proba_map *= roi
            slicings = extractor.extract_random(
                n_random=self.n_patches_per_image, proba_map=proba_map
            )

            for slicing in slicings:
                for i_phase in range(self.n_phases):
                    self._patches.append(
                        {
                            "patient_id": patient_id,
                            "slicing": slicing,
                            "i_phase": i_phase,
                        }
                    )

    def __len__(self):
        return len(self.patients) * self.n_patches_per_image * self.n_phases

    def __getitem__(self, idx):
        if not self.percentiles:
            self.calculate_percentiles()
        if not self._patches:
            self.compile_patch_slicings()

        data = self._patches[idx]
        patient_id = data["patient_id"]
        slicing = data["slicing"]
        i_phase = data["i_phase"]

        logger.debug(f"Return patch {patient_id} / {slicing=} / {i_phase=}")
        pseudo_average_image = self.patients[patient_id]["pseudo_average_image"]
        average_image = self.patients[patient_id]["average_image"]
        average_patch = average_image[slicing]
        # average_patch = CBCTBoostingDataset.preprocess(
        #     average_patch,
        #     input_value_range=self.percentiles["approx_global"]["average_image"],
        #     output_value_range=(0.0, 1.0),
        # )

        pseudo_average_patch = pseudo_average_image[(i_phase,) + slicing]

        # pseudo_average_patch = CBCTBoostingDataset.preprocess(
        #     pseudo_average_patch,
        #     input_value_range=self.percentiles["approx_global"]["pseudo_average_image"],
        #     output_value_range=(0.0, 1.0),
        # )

        return {
            "pseudo_average_image": pseudo_average_patch,
            "average_image": average_patch,
        }


if __name__ == "__main__":
    init_fancy_logging()

    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logging.getLogger("boosting").setLevel(logging.DEBUG)
    dataset = CBCTBoostingDataset(patch_shape=(512, 1, 512))
    dataset.add_patient_from_filepath(
        "pat1",
        pseudo_average_image="/datalake_fast/boosting_test/raw_data/reconstructions/rooster4d_pseudo_average.mha",
        average_image="/datalake_fast/boosting_test/raw_data/reconstructions/fdk3d.mha",
    )

    dataset.calculate_percentiles()
    import matplotlib.pyplot as plt

    i = 10

    fig, ax = plt.subplots(2, 10, sharex=True, sharey=True, squeeze=False)
    for n in range(10):
        d = dataset[i + n]
        ax[0, n].imshow(d["pseudo_average_image"][0, :, :])
        ax[1, n].imshow(d["average_image"][0, :, :])
