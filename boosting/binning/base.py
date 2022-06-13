from abc import ABC, abstractmethod
from boosting.utils import filter_kwargs
import numpy as np


class Binning(ABC):
    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, value):
        self._set_signal(value)

    @property
    def n_bins(self):
        return self.__n_bins

    @n_bins.setter
    def n_bins(self, value):
        self.__n_bins = value

    @property
    def _bin_idx(self):
        return self.__bin_idx

    @_bin_idx.setter
    def _bin_idx(self, value):
        self.__bin_idx = value

    @property
    def _valid_kwargs(self):
        raise NotImplementedError()

    @abstractmethod
    def _set_signal(self, value):
        pass

    @abstractmethod
    def _calculate_bins(self, signal):
        pass

    @abstractmethod
    def get_bin_idx(self, i_bin):
        pass

    @abstractmethod
    def _bin_projections(self, i_bin, **kwargs):
        pass

    def bin_projections(self, i_bin, **kwargs):
        valid_kwargs = filter_kwargs(self._valid_kwargs, **kwargs)
        return self._bin_projections(i_bin=i_bin, **valid_kwargs)

    @staticmethod
    def _interpolate_nans(a):
        is_nan, get_nan_idx = np.isnan(a), lambda z: np.where(z)[0]
        a[is_nan] = np.interp(get_nan_idx(is_nan), get_nan_idx(~is_nan),
                              a[~is_nan])

        return a
