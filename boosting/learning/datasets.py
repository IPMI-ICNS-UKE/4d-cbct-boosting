import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Dict, Union, Tuple

import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset

from boosting.binning.phase import PseudoAverageBinning
from boosting.reconstruction import presets
from boosting.reconstruction.fdk import FDKReconstructor
from boosting.reconstruction.cg import CGReconstructor
from boosting.reconstruction.rooster import ROOSTER4DReconstructor
from boosting.common_types import PathLike
from boosting.utils import to_path, rescale_range, crop_or_pad


class CBCTBoostingDataset(Dataset):
    def __init__(
            self, iteration_axis: int = 1,
            target_image_shape: Tuple[int, int] = (512, 512)
    ):
        self.iteration_axis = iteration_axis
        self.target_image_shape = target_image_shape

        self.patients = {}
        self.percentiles = None

    def calculate_percentiles(self, lower: int = 1, upper: int = 99):
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
        self.patients[patient_id] = {
            "pseudo_average_image": pseudo_average_image,
            "average_image": average_image,
        }

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

    def __len__(self):
        return sum(
            patient["average_image"].shape[self.iteration_axis]
            for patient in self.patients.values()
        )

    def __getitem__(self, idx):
        if not self.percentiles:
            raise RuntimeError("Please calculate percentiles first")

        for patient in self.patients.values():
            iteration_axis_length = patient["average_image"].shape[self.iteration_axis]
            if (_idx := idx - iteration_axis_length) < 0:

                pavg_slice = patient["pseudo_average_image"][:, :, idx, :]
                avg_slice = patient["average_image"][:, idx, :]

                pavg_slice = CBCTBoostingDataset.preprocess(
                    pavg_slice,
                    target_image_shape=(None,) + self.target_image_shape,
                    input_value_range=self.percentiles["approx_global"][
                        "pseudo_average_image"
                    ],
                    output_value_range=(0.0, 1.0),
                )
                avg_slice = CBCTBoostingDataset.preprocess(
                    avg_slice,
                    target_image_shape=self.target_image_shape,
                    input_value_range=self.percentiles["approx_global"][
                        "average_image"
                    ],
                    output_value_range=(0.0, 1.0),
                )

                return {
                    "pseudo_average_image": pavg_slice,
                    "average_image": avg_slice,
                }
            else:
                idx = _idx
        else:
            raise IndexError("Index out of range")

    @staticmethod
    def compile_from_reconstructions(
            pseudo_average_filepath: PathLike, average_filepath: PathLike
    ):
        pseudo_average_image = sitk.ReadImage(
            str(pseudo_average_filepath), sitk.sitkFloat32
        )
        average_image = sitk.ReadImage(str(average_filepath), sitk.sitkFloat32)

        # convert to numpy
        pseudo_average_image = sitk.GetArrayFromImage(pseudo_average_image)
        average_image = sitk.GetArrayFromImage(average_image)

        if pseudo_average_image.ndim != 4:
            raise RuntimeError(
                "Pseudo average image must be a 4D image, i.e., "
                "(time dim, spatial dim, spatial dim, spatial dim)"
            )
        if average_image.ndim != 3:
            raise RuntimeError(
                "Average image must be a 3D image, i.e., "
                "(spatial dim, spatial dim, spatial dim)"
            )

        return pseudo_average_image, average_image

    @staticmethod
    def compile_from_raw_data(
            projections_filepath: PathLike,
            geometry_filepath: PathLike,
            signal_filepath: PathLike,
            motion_mask_filepath: Optional[PathLike] = None,
            n_bins: int = 10,
            reconstruction_params: Optional[Dict] = None,
    ):
        # convert all filepaths to pathlib's Path objects for consistent handling
        (
            projections_filepath,
            geometry_filepath,
            signal_filepath,
            motion_mask_filepath,
        ) = to_path(
            projections_filepath,
            geometry_filepath,
            signal_filepath,
            motion_mask_filepath,
        )

        pseudo_average_binning = PseudoAverageBinning.from_file(
            signal_filepath, n_bins=n_bins
        )
        reconstructor_pseudo_average = ROOSTER4DReconstructor(
            detector_binning=1, respiratory_binning=pseudo_average_binning
        )
        reconstructor_average = FDKReconstructor(
            detector_binning=1, respiratory_binning=None
        )
        reconstructor_average_cg = CGReconstructor(
            detector_binning=1, respiratory_binning=None
        )

        with TemporaryDirectory() as temp_directory:
            temp_directory = Path(temp_directory)


            reconstructor_average_cg.reconstruct(
                path=projections_filepath.parent,
                regexp=projections_filepath.name,
                geometry=geometry_filepath,
                output=temp_directory / "average.mha",
                post_process=False,
                **{
                    'dimension': (304, 250, 390),
                    'spacing': (1.0, 1.0, 1.0),
                    'origin': (-161, -128, -195),
                    'fp': 'CudaRayCast',
                    'bp': 'CudaVoxelBased',
                    'niterations': 40,
                }
            )

            # # run reconstruction for corresponding real average images
            # reconstructor_average.reconstruct(
            #     path=projections_filepath.parent,
            #     regexp=projections_filepath.name,
            #     geometry=geometry_filepath,
            #     output=temp_directory / "average.mha",
            #     post_process=False,
            #     **presets.FDK,
            # )

            # run reconstruction for pseudo average images
            reconstructor_pseudo_average.reconstruct(
                path=projections_filepath.parent,
                regexp=projections_filepath.name,
                geometry=geometry_filepath,
                motionmask=motion_mask_filepath,
                output=temp_directory / "pseudo_average.mha",
                post_process=False,
                **presets.ROOSTER4D,
            )

            return CBCTBoostingDataset.compile_from_reconstructions(
                pseudo_average_filepath=temp_directory / "pseudo_average.mha",
                average_filepath=temp_directory / "average.mha",
            )


if __name__ == "__main__":
    dataset = CBCTBoostingDataset(target_image_shape=(390, 304))
    pseudo_average_image, average_image = dataset.compile_from_raw_data(
        projections_filepath='/datalake/4d_cbct_lmu/Hamburg/correctedProjs.mhd',
        geometry_filepath='/datalake/4d_cbct_lmu/Hamburg/geom.xml',
        signal_filepath='/datalake/4d_cbct_lmu/Hamburg/cphase_adjusted_0.08.txt'
    )
    dataset.add_patient(
        'pat1',
        pseudo_average_image=pseudo_average_image,
        average_image=average_image
    )
    dataset.calculate_percentiles()
    dataset.save("/datalake/4d_cbct_lmu/Hamburg/dataset.pkl")

    dataset = CBCTBoostingDataset.load(
        "/datalake/4d_cbct_lmu/Hamburg/dataset.pkl"
    )

    d = dataset[178]
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 11, sharex=True, sharey=True)
    for n in range(10):
        ax[n].imshow(d['pseudo_average_image'][n])
    ax[10].imshow(d['average_image'])
