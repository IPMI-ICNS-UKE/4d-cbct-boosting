from abc import ABC, abstractmethod

from boosting.binning.base import Binning
from boosting.binning.phase import PhaseBinning, PseudoAverageBinning
from boosting.logger import LoggerMixin


class Reconstructor(ABC, LoggerMixin):
    def __init__(
        self,
        bin_rootdir="/home/fmadesta/software/rtk/built_v2.0.0/bin",
        detector_binning=2,
        respiratory_binning=None,
    ):
        self.bin_rootdir = bin_rootdir

        self.detector_binning = detector_binning
        self.respiratory_binning = respiratory_binning

    @property
    def detector_binning(self):
        return self.__detector_binning

    @detector_binning.setter
    def detector_binning(self, value):
        self.__detector_binning = value

    @property
    def respiratory_binning(self):
        return self.__respiratory_binning

    @respiratory_binning.setter
    def respiratory_binning(self, value):
        if value is None or isinstance(value, Binning):
            self.__respiratory_binning = value
        else:
            raise ValueError("Invalid binning method set.")

    @abstractmethod
    def _preprocessing(self, **kwargs):
        pass

    @abstractmethod
    def _reconstruct(self, **kwargs):
        pass

    @abstractmethod
    def _postprocessing(self, reconstruction_filepath, **kwargs):
        pass

    def reconstruct(self, post_process=True, **kwargs):
        self.logger.debug(f"Start reconstruction with params: {kwargs}")
        self._preprocessing(**kwargs)
        reconstruction_filepath = self._reconstruct(**kwargs)
        if post_process:
            self._postprocessing(reconstruction_filepath)


if __name__ == "__main__":
    binning = PhaseBinning.from_file(
        "/datalake/4d_cbct_lmu/Hamburg/cphase_adjusted_0.08.txt", n_bins=10
    )
    pa_binning = PseudoAverageBinning.from_file(
        "/datalake/4d_cbct_lmu/Hamburg/cphase_adjusted_0.08.txt", n_bins=10
    )
    pa_binning.get_shifted_signal()

    # plt.plot(binning.signal)
    # plt.plot(pa_binning.get_shifted_signal())

    # reconstructor_rooster = ROOSTER4D(
    #     detector_binning=1,
    #     respiratory_binning=binning,
    #     bin_rootdir='/home/fmadesta/software/rtk/built_v2.0.0/bin'
    # )
    #
    # reconstructor_rooster.reconstruct(
    #     path=os.path.split('/datalake/4d_cbct_lmu/Hamburg/correctedProjs.mhd')[0],
    #     regexp=os.path.split('/datalake/4d_cbct_lmu/Hamburg/correctedProjs.mhd')[1],
    #     geometry='/datalake/4d_cbct_lmu/Hamburg/geom.xml',
    #     motionmask='/datalake/4d_cbct_lmu/Hamburg/body_mask.mhd',
    #     fp='CudaRayCast',
    #     bp='CudaVoxelBased',
    #     dimension=(304, 250, 390, 10),
    #     spacing=(1.0, 1.0, 1.0, 1.0),
    #     origin=(-161, -128, -195, 0),
    #     niter=10,
    #     cgiter=4,
    #     tviter=10,
    #     gamma_time=0.0002,
    #     gamma_space=0.00005,
    #     output='/datalake/4d_cbct_lmu/Hamburg/test.mha',
    #     post_process=False
    # )
