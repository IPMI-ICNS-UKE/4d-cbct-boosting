import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence, Tuple, List
from zipfile import ZipFile

import SimpleITK as sitk
import h5py as h5
import numpy as np
import yaml

from boosting.common_types import PathLike
from boosting.reconstruction.binning import save_curve
from boosting.reconstruction.varian.geometry import generate_geometry
from boosting.reconstruction.varian.projections import convert_xim

logger = logging.getLogger(__name__)


def remove_incomplete_projections(
    filepaths: Sequence[PathLike], filesize_threshold: int = 512 * 1024
):
    filepaths = [Path(f) for f in filepaths]
    cleaned_filepaths = []
    for filepath in [f for f in filepaths if "Proj" in f.name]:
        if os.path.getsize(filepath) < filesize_threshold:
            os.remove(filepath)
        else:
            cleaned_filepaths.append(filepath)

    return cleaned_filepaths


def get_nan_sections(curve: np.ndarray) -> List[Tuple[int, int]]:
    nan_idx = np.where(np.isnan(curve))[0]

    first = None
    indices = []
    for idx, difference in zip(nan_idx, np.diff(nan_idx)):
        if first is None:
            first = idx
        if difference > 1:
            last = idx
            indices.append((first, last))
            first = None
    if first and idx:
        indices.append((first, idx + 1))

    return indices


def interpolate_nan_phases(phase: np.ndarray) -> np.ndarray:
    phase = phase.copy()
    nan_sections = get_nan_sections(phase)

    for first, last in nan_sections:
        first_valid_idx = first - 1
        last_valid_idx = last + 1

        n_nans = last - first + 1

        interpolated = np.linspace(
            phase[first_valid_idx], phase[last_valid_idx], num=n_nans + 2
        )
        logger.info(f"Interpolated phase section ({first}, {last})")

        phase[first_valid_idx : last_valid_idx + 1] = interpolated

    return phase


def read_respiratory_curve(
    image_params_filepath: PathLike,
) -> Tuple[np.ndarray, np.ndarray]:
    amplitudes = []
    phases = []
    with h5.File(image_params_filepath, "r") as f:
        for p in range(len(f["ImageParameters"])):
            projection_number = str(p).zfill(5)
            current_amplitude = f["ImageParameters"][projection_number].attrs.get(
                "GatingAmplitude", np.nan
            )
            current_phase = f["ImageParameters"][projection_number].attrs.get(
                "GatingPhase", np.nan
            )

            if current_amplitude is np.nan:
                logger.warning(f"Amplitude is NaN for projection {projection_number}")

            if current_phase is np.nan:
                logger.warning(f"Phase is NaN for projection {projection_number}")

            amplitudes.append(current_amplitude)
            phases.append(current_phase)

    return (
        np.array(amplitudes, dtype=np.float32).squeeze(),
        np.array(phases, dtype=np.float32).squeeze(),
    )


def _extract_file_and_move(
    zip_file,
    compressed_file: str,
    temp_dir: Path,
    output_folder: Path,
    sub_folder: Path,
):
    zip_file.extract(compressed_file, temp_dir)
    extracted_filepath = temp_dir / compressed_file

    destination_filepath = output_folder / sub_folder / extracted_filepath.name
    destination_filepath.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(extracted_filepath, destination_filepath)

    return destination_filepath


def extract_data_from_zip(
    filepath: PathLike,
    output_folder: PathLike,
    air_scan: PathLike = "AIR-Half-Bowtie-125KV/Current/FilterBowtie.xim",
    clean_projections: bool = True,
) -> dict:
    # overwrite converted types
    filepath = Path(filepath)
    output_folder = Path(output_folder)

    projection_filepaths = []
    calibration_filepaths = []
    with ZipFile(filepath, "r") as zip_file, TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        compressed_files = zip_file.namelist()
        for compressed_file in compressed_files:
            if compressed_file.endswith(".xim") and "Acquisitions" in compressed_file:
                file_description = "projections"
                sub_folder = Path("projections")

            elif compressed_file.endswith(".xim") and "Calibrations" in compressed_file:
                file_description = "calibrations"
                sub_folder = Path("calibrations") / "/".join(
                    Path(compressed_file).parts[-3:-1]
                )

            elif compressed_file.endswith("Scan.xml"):
                file_description = "scan definition"
                sub_folder = Path("meta")

            elif compressed_file.endswith("ImgParameters.h5"):
                file_description = "breathing curve"
                sub_folder = Path("meta")

            else:
                continue

            logger.info(f"extract {file_description}: {compressed_file}")

            destination_filepath = _extract_file_and_move(
                zip_file=zip_file,
                compressed_file=compressed_file,
                temp_dir=temp_dir,
                output_folder=output_folder,
                sub_folder=sub_folder,
            )
            if sub_folder.parts[0] == "projections":
                projection_filepaths.append(destination_filepath)
            elif sub_folder.parts[0] == "calibrations":
                calibration_filepaths.append(destination_filepath)

    # find right air scan/calibration
    try:
        air_scan_filepath = next(
            c for c in calibration_filepaths if str(c).endswith(air_scan)
        )
    except StopIteration:
        raise FileNotFoundError(f"Air scan {air_scan} not found")

    if clean_projections:
        projection_filepaths = remove_incomplete_projections(projection_filepaths)

    # process and save respiratory curve
    projections_config = output_folder / "meta/ImgParameters.h5"
    amplitudes, phases = read_respiratory_curve(projections_config)

    save_curve(amplitudes, filepath=output_folder / "meta" / "amplitudes.txt")
    save_curve(phases, filepath=output_folder / "meta" / "phases.txt")

    filepaths = {
        "projections": projection_filepaths,
        "air_scan": air_scan_filepath,
        "scan_config": output_folder / "meta/Scan.xml",
        "projections_config": projections_config,
    }

    with open(output_folder / "files.yaml", "w") as f:
        yaml.dump(filepaths, f)

    return filepaths


def prepare_for_reconstruction(extracted_folder: PathLike):
    extracted_folder = Path(extracted_folder)
    normalized_projections_filepath = extracted_folder / "normalized_projections.mha"
    projections_filepath = extracted_folder / "projections.mha"
    geometry_filepath = extracted_folder / "meta" / "geometry.xml"
    files_filepath = extracted_folder / "files.yaml"

    with open(files_filepath, "r") as f:
        filepaths = yaml.load(f, Loader=yaml.Loader)

    if not normalized_projections_filepath.exists():
        projections_normalized, projections = convert_xim(
            xim_files=filepaths["projections"],
            air_scan_filepath=filepaths["air_scan"],
            detector_spacing=(0.388, 0.388),
            show_progress=True,
        )
        sitk.WriteImage(projections_normalized, str(normalized_projections_filepath))
        sitk.WriteImage(projections, str(projections_filepath))

    if not geometry_filepath.exists():
        # create scan geometry as XML file
        generate_geometry(
            scan_xml_filepath=filepaths["scan_config"],
            projection_folder=filepaths["projections"][0].parent,
            output_filepath=geometry_filepath,
            use_docker=True,
        )
