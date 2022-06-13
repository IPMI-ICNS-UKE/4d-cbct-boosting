from boosting.reconstruction.base import Reconstructor
from boosting.utils import call, create_bin_call, iec61217_to_rsp
import SimpleITK as sitk


class CGReconstructor(Reconstructor):
    def _preprocessing(self, **kwargs):
        pass

    def _reconstruct(self, **kwargs):
        base_call = [
            f'{self.bin_rootdir}/rtkconjugategradient',
            '-v',
        ]

        bin_call = create_bin_call(base_call, **kwargs)
        call(bin_call)

        return kwargs['output']


    def _postprocessing(self, reconstruction_filepath, **kwargs):
        image = sitk.ReadImage(reconstruction_filepath)
        image = iec61217_to_rsp(image)

        sitk.WriteImage(image, reconstruction_filepath)
