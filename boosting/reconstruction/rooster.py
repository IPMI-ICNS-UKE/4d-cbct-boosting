import os
from tempfile import TemporaryDirectory

import SimpleITK as sitk
import numpy as np

from boosting.binning.phase import PseudoAverageBinning
from boosting.reconstruction.base import Reconstructor
from boosting.utils import call, create_bin_call, iec61217_to_rsp

class ROOSTER4DReconstructor(Reconstructor):
    def _preprocessing(self, **kwargs):
        pass

    def _reconstruct(self, **kwargs):
        base_call = [
            f'{self.bin_rootdir}/rtkfourdrooster'
        ]
        assert self.respiratory_binning
        with TemporaryDirectory() as temp_dir:
            signal_filepath = os.path.join(temp_dir, 'signal.txt')
            if isinstance(self.respiratory_binning, PseudoAverageBinning):
                pseudo_signal = self.respiratory_binning.get_shifted_signal()
                np.savetxt(signal_filepath, pseudo_signal / 360.0, fmt='%.4f')
            else:
                np.savetxt(
                    signal_filepath, self.respiratory_binning.signal / 360.0,
                    fmt='%.4f'
                )

            kwargs['signal'] = signal_filepath

            bin_call = create_bin_call(base_call, **kwargs)
            self.logger.debug(f'Converted to binary call: {bin_call}')
            call(bin_call)

            return kwargs['output']

    def _postprocessing(self, reconstruction_filepath, **kwargs):
        image = sitk.ReadImage(reconstruction_filepath)
        image = iec61217_to_rsp(image)

        sitk.WriteImage(image, reconstruction_filepath)
