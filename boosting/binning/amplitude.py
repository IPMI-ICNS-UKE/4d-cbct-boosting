import numpy as np

from boosting.binning.base import Binning
from boosting.utils import call, create_bin_call


class AmplitudeBinning(Binning):
    def _set_signal(self, value):
        self._signal = self._interpolate_nans(value)
        self._bin_idx = self._calculate_bins(self._signal)

    @staticmethod
    def _mod_phase_bins(signal):
        for i, _ in enumerate(signal):
            if signal[i] > 360.0:
                signal[i] = signal[i] - 360.0
            elif signal[i] < 0.0:
                signal[i] = signal[i] + 360.0
        return signal

    def _bin_phase_array(self, signal, bins_left, bins_right):
        phase_binned = np.zeros(len(signal), dtype=np.int16)
        for i, p in enumerate(signal):
            for bin_number, (b_left, b_right) in enumerate(
                    zip(bins_left, bins_right)):
                if 0 < bin_number < self.n_bins and b_left <= p < b_right:
                    phase_binned[i] = bin_number
        return phase_binned

    def _calculate_bins(self, signal):
        bins_centers = np.linspace(0, 360, self.n_bins + 1)
        bins_left = self._mod_phase_bins(
            bins_centers - 360.0 / (2 * self.n_bins))
        bins_right = self._mod_phase_bins(
            bins_centers + 360.0 / (2 * self.n_bins))
        return self._bin_phase_array(signal, bins_left,
                                     bins_right)

    def get_bin_idx(self, i_bin):
        try:
            return np.where(self._bin_idx == i_bin)[0]
        except IndexError:
            raise IndexError('No projections found for this bin!')

    @property
    def _valid_kwargs(self):
        return ('path',
                'geometry',
                'regexp',
                'out_geometry',
                'out_proj',
                'list')

    def _bin_projections(self, i_bin, **kwargs):
        base_call = ['/home/fmadesta/software/rtk/built_v2.0.0/bin/rtksubselect',
                     '-v']

        kwargs['list'] = ','.join(self.get_bin_idx(i_bin).astype(str))
        bin_call = create_bin_call(base_call, **kwargs)
        return call(bin_call)

    @classmethod
    def from_file(cls, filepath, **kwargs):
        phases = np.loadtxt(filepath, dtype=np.float32)

        if phases.min() >= 0.0 and phases.max() <= 1.0:
            phases *= 360.0

        phase_binning = cls(**kwargs)
        phase_binning.signal = phases

        return phase_binning


if __name__ == '__main__':
    signal = np.sin(np.linspace(0, 6, 100))