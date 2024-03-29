import logging
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import SimpleITK as sitk
import docker
import numpy as np
from docker.errors import ImageNotFound

import boosting.reconstruction.defaults as defaults
from boosting.common_types import PathLike
from boosting.logger import LoggerMixin
from boosting.reconstruction.binning import save_curve
from boosting.shell import create_cli_command, execute, execute_in_docker
from boosting.utils import check_docker_image_exists
from boosting.utils import iec61217_to_rsp

logger = logging.getLogger(__name__)


class Reconstructor(ABC, LoggerMixin):
    def __init__(
        self,
        executable: PathLike,
        detector_binning: int = 1,
        use_docker: bool = True,
        gpu_id: int = 0,
    ):
        self.executable = executable
        self.detector_binning = detector_binning
        self.use_docker = use_docker
        self.gpu_id = gpu_id

        if self.use_docker:
            if not check_docker_image_exists(defaults.DOCKER_IMAGE):
                raise ImageNotFound(f"Docker image {defaults.DOCKER_IMAGE} not found.")
            self.docker_client = docker.from_env()
        else:
            self.docker_client = None

    @property
    def _execute_function(self):
        execute_func = (
            partial(execute_in_docker, gpus=[self.gpu_id])
            if self.use_docker
            else partial(execute, gpus=[self.gpu_id])
        )
        return execute_func

    @property
    def detector_binning(self):
        return self.__detector_binning

    @detector_binning.setter
    def detector_binning(self, value):
        self.__detector_binning = value

    @abstractmethod
    def _preprocessing(self, **kwargs):
        pass

    @abstractmethod
    def _reconstruct(self, output_filepath: PathLike, **kwargs) -> Path:
        pass

    @abstractmethod
    def _postprocessing(self, reconstruction_filepath: PathLike, **kwargs):
        pass

    def reconstruct(
        self, output_filepath: PathLike, post_process: bool = True, **kwargs
    ) -> Path:
        self.logger.debug(f"Start reconstruction with params: {kwargs}")
        self._preprocessing(**kwargs)

        reconstruction_filepath = self._reconstruct(
            output_filepath=output_filepath, **kwargs
        )

        if post_process:
            self._postprocessing(reconstruction_filepath)

        return reconstruction_filepath


class RTKReconstructor(Reconstructor):
    def _preprocessing(self, **kwargs):
        pass

    def _reconstruct(self, output_filepath: PathLike, **kwargs) -> Path:
        logger.info(f"Start reconstruction with the following params: {kwargs}")
        bin_call = create_cli_command(
            self.executable,
            output=output_filepath,
            path_prefix=defaults.DOCKER_PATH_PREFIX if self.use_docker else None,
            convert_underscore=None,
            verbose=True,
            **kwargs,
        )
        self.logger.debug(f"Converted to binary call: {bin_call}")
        self._execute_function(bin_call)

        return Path(output_filepath)

    def _postprocessing(self, reconstruction_filepath: PathLike, **kwargs):
        reconstruction_filepath = str(reconstruction_filepath)
        image = sitk.ReadImage(reconstruction_filepath)
        image = iec61217_to_rsp(image)

        sitk.WriteImage(image, reconstruction_filepath)


class ROOSTER4DReconstructor(RTKReconstructor):
    def __init__(
        self,
        phase_signal: Optional[np.ndarray],
        executable: PathLike = "rtkfourdrooster",
        detector_binning: int = 1,
        use_docker: bool = True,
        gpu_id: int = 0,
    ):
        super().__init__(
            executable=executable,
            detector_binning=detector_binning,
            use_docker=use_docker,
            gpu_id=gpu_id,
        )
        self.phase_signal = phase_signal

    def _reconstruct(self, output_filepath: PathLike, **kwargs) -> Path:
        with TemporaryDirectory() as temp_dir:
            signal_filepath = Path(temp_dir) / "signal.txt"
            save_curve(self.phase_signal, filepath=signal_filepath)

            kwargs["signal"] = signal_filepath

            return super()._reconstruct(output_filepath=output_filepath, **kwargs)


class FDKReconstructor(RTKReconstructor):
    def __init__(
        self,
        executable: PathLike = "rtkfdk",
        detector_binning: int = 1,
        use_docker: bool = True,
        gpu_id: int = 0,
    ):
        super().__init__(
            executable=executable,
            detector_binning=detector_binning,
            use_docker=use_docker,
            gpu_id=gpu_id,
        )
