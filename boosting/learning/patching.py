from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np


class PatchStitcher:
    def __init__(
        self,
        array_shape: Tuple[int, ...],
        color_axis: int | None = 0,
        dtype=np.float32,
        max_expected_overlap: int = 255,
    ):
        self.array_shape = array_shape
        self.color_axis = color_axis

        self.dtype = dtype
        self.max_expected_overlap = max_expected_overlap
        if self.max_expected_overlap < 2**8:
            self._n_dtype = np.uint8
        elif self.max_expected_overlap < 2**16:
            self._n_dtype = np.uint16
        else:
            self._n_dtype = np.uint32
        self._hard_overlap_limit = np.iinfo(self._n_dtype).max
        self._unsigned_dtype = None

        self.n_patches_added = 0

        self.reset()

    @property
    def array_shape(self):
        return self.__array_shape

    @array_shape.setter
    def array_shape(self, value):
        self.__array_shape = value

    @property
    def color_axis(self):
        return self.__color_axis

    @color_axis.setter
    def color_axis(self, value):
        if value is None:
            self.__color_axis = value
        else:
            assert -self.n_total_dims <= value < self.n_total_dims
            self.__color_axis = value if value >= 0 else value + self.n_total_dims

    @property
    def n_total_dims(self):
        return len(self.array_shape)

    def reset(self):
        self.k = np.zeros(self.array_shape, dtype=self.dtype)
        self.n = np.zeros(self.array_shape, dtype=self._n_dtype)
        self.sum = np.zeros(self.array_shape, dtype=self.dtype)
        self.sum_squared = np.zeros(
            self.array_shape, dtype=self._unsigned_dtype or self.dtype
        )

        self.n_patches_added = 0

    def print_internal_stats(self):
        stats = (
            f"internal min/max stats:\n"
            f"n: min={self.n.min()}, max={self.n.max()}\n"
            f"k: min={self.k.min()}, max={self.k.max()}\n"
            f"sum: min={self.sum.min()}, max={self.sum.max()}\n"
            f"sum_squared: min={self.sum_squared.min()}, max={self.sum_squared.max()}"
        )
        print(stats)

    @property
    def coverage(self):
        return self.n

    def add_patch(self, data: np.ndarray, slicing: Tuple[slice, ...]):
        with np.errstate(over="raise", under="raise"):
            n_masking = self.n[slicing] == 0
            self.k[slicing][n_masking] = data[n_masking]
            self.n[slicing] += 1
            diff = data - self.k[slicing]

            self.sum[slicing] += diff
            self.sum_squared[slicing] += diff**2

    def add_patches(
        self, data: Sequence[np.ndarray], slicings: Sequence[Tuple[slice, ...]]
    ):
        for patch, slicing in zip(data, slicings):
            self.add_patch(patch, slicing)

    def calculate_mean(self, default_value: float = 0.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(self.k + self.sum / self.n, nan=default_value)

    def calculate_variance(self, ddof: int = 0):
        return (self.sum_squared - self.sum**2 / self.n) / (self.n - ddof)


class PatchExtractor:
    def __init__(
        self,
        patch_shape: Tuple[int | None, ...],
        array_shape: Tuple[int, ...],
        color_axis: int | None = 0,
        squeeze_patch_axes: set[int] | None = None,
    ):
        self.array_shape = array_shape
        self.color_axis = color_axis
        self.patch_shape = patch_shape

        self.squeeze_patch_axes = (
            set(squeeze_patch_axes) if squeeze_patch_axes else set()
        )

        self._check_squeeze_patch_dims()

        self._rng = np.random.default_rng()

    def _check_squeeze_patch_dims(self):
        for squeeze_axis in self.squeeze_patch_axes:
            if self.patch_shape[squeeze_axis] != 1:
                raise ValueError(
                    f"Can only squeeze patch axes where size is 1, but got "
                    f"size {self.patch_shape[squeeze_axis]} in patch axis {squeeze_axis}"
                )

    def central_to_lower_index(
        self, central_index: Tuple[int, ...], correct: bool = True
    ):
        lower_index = tuple(
            c - hps for c, hps in zip(central_index, self.half_patch_shape)
        )
        if correct:
            return self.correct_index(lower_index)
        return lower_index

    def lower_to_cental_index(self, lower_index: Tuple[int, ...], correct: bool = True):
        central_index = tuple(
            lower + half_ps
            for lower, half_ps in zip(lower_index, self.half_patch_shape)
        )
        if correct:
            return self.correct_index(central_index)
        return central_index

    def correct_index(self, index: Tuple[int, ...]):
        index = list(index)
        low_corrections = tuple(idx - l for idx, l in zip(index, self.min_lower_index))
        high_corrections = tuple(idx - h for idx, h in zip(index, self.max_upper_index))

        if all(lc == 0 for lc in low_corrections) and all(
            hc == 0 for hc in high_corrections
        ):
            return index

        for i_dim, low_correction in enumerate(low_corrections):
            if low_correction < 0:
                index[i_dim] -= low_correction

        for i_dim, high_correction in enumerate(high_corrections):
            if high_correction > 0:
                index[i_dim] -= high_correction

        return tuple(index)

    @property
    def half_patch_shape(self) -> Tuple[int, ...]:
        return tuple(ps // 2 for ps in self.patch_shape)

    @property
    def color_axis(self) -> int | None:
        return self.__color_axis

    @color_axis.setter
    def color_axis(self, value):
        if value is None:
            self.__color_axis = value
        else:
            assert -self.n_total_dims <= value < self.n_total_dims
            self.__color_axis = value if value >= 0 else value + self.n_total_dims

    @property
    def array_shape(self) -> Tuple[int, ...]:
        return self.__array_shape

    @array_shape.setter
    def array_shape(self, value: Tuple[int, ...]):
        self.__array_shape = value

    @property
    def patch_shape(self) -> Tuple[int, ...]:
        return self.__patch_shape

    @patch_shape.setter
    def patch_shape(self, value: Tuple[int, ...]):
        n_color_dim = 1 if self.color_axis is not None else 0
        assert len(value) + n_color_dim == len(self.array_shape)
        self.__patch_shape = value

    @property
    def n_total_dims(self) -> int:
        return len(self.array_shape)

    @property
    def spatial_dims(self) -> Tuple[int, ...]:
        return tuple(i for i in range(self.n_total_dims) if i != self.color_axis)

    @property
    def n_spatial_dims(self) -> int:
        return len(self.spatial_dims)

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        return tuple(self.array_shape[i] for i in self.spatial_dims)

    @property
    def min_lower_index(self) -> Tuple[int, ...]:
        return (0,) * len(self.spatial_dims)

    @property
    def max_upper_index(self) -> Tuple[int, ...]:
        upper_index = tuple(
            self.array_shape[i_dim] - p_s
            for (p_s, i_dim) in zip(self.patch_shape, self.spatial_dims)
        )
        return upper_index

    @staticmethod
    def _convert_to_tuple(value: Any, n_dims: int = 1) -> Tuple[Any, ...]:
        if isinstance(value, tuple):
            return value
        return (value,) * n_dims

    def calculate_number_ordered_indices(
        self, stride: int | Tuple[int, ...] = 1
    ) -> Tuple[int, ...]:
        stride = self._convert_to_tuple(stride, n_dims=len(self.spatial_dims))
        n = []
        for i in range(len(self.spatial_dims)):
            n.append(
                1 + (self.max_upper_index[i] - self.min_lower_index[i]) // stride[i]
            )
        return tuple(n)

    def _get_ordered_indices(
        self, stride: int | Tuple[int, ...], flush: bool = False
    ) -> List[Tuple[int, ...]]:
        stride = self._convert_to_tuple(stride, n_dims=self.n_spatial_dims)
        ranges = []
        lower = self.min_lower_index
        upper = self.max_upper_index

        for i in range(len(self.spatial_dims)):
            if flush and upper[i] > lower[i]:
                # upper[i] == lower[i] if this dim is completely covered by one patch
                # in this case we do not need to flush
                ranges.append(
                    np.arange(
                        lower[i], upper[i] + stride[i] + 1, stride[i], dtype=np.uint16
                    )
                )
            else:
                ranges.append(
                    np.arange(lower[i], upper[i] + 1, stride[i], dtype=np.uint16)
                )

        mesh = np.meshgrid(*ranges, indexing="ij", sparse=False)
        indices = np.concatenate(
            tuple(m[..., np.newaxis] for m in mesh), axis=-1
        ).reshape((-1, len(self.spatial_dims)))
        return [self.correct_index(idx) for idx in indices]

    def get_patch_slicing(self, lower_index: Tuple[int, ...]) -> Tuple[slice, ...]:
        if len(lower_index) != self.n_spatial_dims:
            raise ValueError(f"Index does not have {self.n_spatial_dims} dimensions")
        slicing = [slice(None, None, None)] * self.n_total_dims
        for i, spatial_dim in enumerate(self.spatial_dims):
            if i not in self.squeeze_patch_axes:
                slicing[spatial_dim] = slice(
                    lower_index[i], lower_index[i] + self.patch_shape[i]
                )
            else:
                # patch size in this dimention is None or 0
                slicing[spatial_dim] = lower_index[i]

        return tuple(slicing)

    def full_to_spatial_slicing(
        self, full_slicing: Tuple[slice, ...]
    ) -> Tuple[slice, ...]:
        spatial_slicing = []
        for i, spatial_dim in enumerate(self.spatial_dims):
            spatial_slicing.append(full_slicing[spatial_dim])

        return tuple(spatial_slicing)

    def extract_ordered(
        self,
        stride: int | Tuple[int, ...] | None = None,
        flush: bool = True,
        mask: np.ndarray | None = None,
    ):
        if stride is None:
            stride = self.patch_shape

        ordered_indices = self._get_ordered_indices(stride=stride, flush=flush)

        for idx in ordered_indices:
            slicing = self.get_patch_slicing(idx)
            if mask is not None:
                spatial_slicing = self.full_to_spatial_slicing(slicing)
                if not mask[spatial_slicing].any():
                    continue
            yield slicing

    def _sample_indices(self, n_random: int, proba_map: np.ndarray | None):
        if isinstance(proba_map, np.ndarray):
            proba_map = proba_map.ravel() / proba_map.sum()
            indices = np.arange(len(proba_map))
            n_non_zero = (proba_map > 0).sum()
            n_random = min(n_random, n_non_zero)
        else:
            indices = np.arange(np.prod(self.array_shape))

        selected_indices = np.column_stack(
            np.unravel_index(
                self._rng.choice(indices, p=proba_map, size=n_random, replace=False),
                self.spatial_shape,
            )
        )

        return [central_index for central_index in selected_indices]

    def extract_random(
        self,
        n_random: int = 1,
        proba_map: np.ndarray | None = None,
    ):
        assert n_random >= 1, "Please sample at least one patch"
        if proba_map is None:
            proba_map = np.ones(self.spatial_shape)
        else:
            if proba_map.shape != self.spatial_shape:
                raise ValueError("Shape mismatch")
        random_indices = self._sample_indices(n_random=n_random, proba_map=proba_map)
        for random_index in random_indices:
            lower_index = self.central_to_lower_index(random_index)
            lower_index = self.correct_index(lower_index)

            yield self.get_patch_slicing(lower_index=lower_index)
