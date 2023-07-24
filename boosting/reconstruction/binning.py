from dataclasses import dataclass
from math import ceil
from typing import List, Tuple

import numpy as np
import scipy.signal as signal

from boosting.common_types import PathLike
from boosting.constants import PI2


def find_peaks(x: np.ndarray, scale: int = None, debug: bool = False):
    """Find peaks in quasi-periodic noisy signals using AMPD algorithm.
    Extended implementation handles peaks near start/end of the signal.
    Optimized implementation by Igor Gotlibovych, 2018.

    Taken from https://github.com/ig248/pyampd

    Parameters
    ----------
    x : ndarray
        1-D array on which to find peaks
    scale : int, optional
        specify maximum scale window size of (2 * scale + 1)
    debug : bool, optional
        if set to True, return the Local Scalogram Matrix, `LSM`,
        weigted number of maxima, 'G',
        and scale at which G is maximized, `l`,
        together with peak locations
    Returns
    -------
    pks: ndarray
        The ordered array of peak indices found in `x`
    """
    x = signal.detrend(x)
    N = len(x)
    L = N // 2
    if scale:
        L = min(scale, L)

    # create LSM matix
    LSM = np.ones((L, N), dtype=bool)
    for k in np.arange(1, L + 1):
        LSM[k - 1, 0 : N - k] &= x[0 : N - k] > x[k:N]  # compare to right neighbours
        LSM[k - 1, k:N] &= x[k:N] > x[0 : N - k]  # compare to left neighbours

    # Find scale with most maxima
    G = LSM.sum(axis=1)
    G = G * np.arange(
        N // 2, N // 2 - L, -1
    )  # normalize to adjust for new edge regions
    l_scale = np.argmax(G)

    # find peaks that persist on all scales up to l
    pks_logical = np.min(LSM[0:l_scale, :], axis=0)
    pks = np.flatnonzero(pks_logical)
    if debug:
        return pks, LSM, G, l_scale
    return pks


@dataclass
class RespiratoryStatistics:
    mean_cycle_period: float
    median_cycle_period: float
    std_cycle_period: float
    n_complete_cycles: float
    mean_cycle_span: float
    std_cycle_span: float
    total_length_secs: float


def split_into_cycles(curve: np.ndarray, peaks: np.ndarray = None) -> List[np.ndarray]:
    if peaks is None:
        peaks = find_peaks(curve)

    # discard potentially incomplete first and last cycle
    slicing = slice(None)
    if peaks[0] == 0:
        slicing = slice(1, None)
    if peaks[-1] == len(curve) - 1:
        slicing = slice(slicing.start, -1)

    return np.split(curve, peaks[slicing])


def align_cycles(cycles: List[np.ndarray]) -> np.ndarray:
    minimum_indices = [np.argmin(c) for c in cycles]

    lefts = [c[:min_idx] for (c, min_idx) in zip(cycles, minimum_indices)]
    rights = [c[min_idx:] for (c, min_idx) in zip(cycles, minimum_indices)]

    max_left_length = max(len(left) for left in lefts)
    max_right_length = max(len(right) for right in rights)

    for i, (left, right) in enumerate(zip(lefts, rights)):
        lefts[i] = np.pad(
            left,
            (max_left_length - len(left), 0),
            mode="constant",
            constant_values=np.nan,
        )
        rights[i] = np.pad(
            right,
            (0, max_right_length - len(right)),
            mode="constant",
            constant_values=np.nan,
        )

    return np.hstack((lefts, rights))


def calculate_median_cycle_length(curve: np.ndarray) -> int:
    cycles = split_into_cycles(curve)
    return int(np.median([len(c) for c in cycles]))


def calculate_respiratory_statistics(
    amplitudes: np.ndarray, sampling_rate: float = 1.0
) -> RespiratoryStatistics:
    cycles = split_into_cycles(amplitudes)
    cycle_lengths = [len(c) / sampling_rate for c in cycles]
    cycle_spans = [max(c) - min(c) for c in cycles]
    return RespiratoryStatistics(
        mean_cycle_period=float(np.mean(cycle_lengths)),
        median_cycle_period=float(np.median(cycle_lengths)),
        std_cycle_period=float(np.std(cycle_lengths)),
        n_complete_cycles=len(cycle_lengths),
        mean_cycle_span=float(np.mean(cycle_spans)),
        std_cycle_span=float(np.std(cycle_spans)),
        total_length_secs=float(np.sum(cycle_lengths)),
    )


def calculate_median_cycle(curve: np.ndarray) -> np.ndarray:
    cycles = split_into_cycles(curve)
    resp_stats = calculate_respiratory_statistics(curve)

    cycles = [
        c
        for c in cycles
        if resp_stats.median_cycle_period - resp_stats.std_cycle_period
        <= len(c)
        <= resp_stats.median_cycle_period + resp_stats.std_cycle_period
    ]

    # stretch each cycle to median cycle length
    cycles = [
        np.interp(
            x=np.linspace(
                0, len(c) - 1, int(resp_stats.median_cycle_period), endpoint=True
            ),
            xp=np.arange(len(c)),
            fp=c,
        )
        for c in cycles
    ]

    return np.median(cycles, axis=0)


def calculate_amplitude_bins(
    breathing_curve: np.ndarray, n_bins: int = 10
) -> np.ndarray:
    median_cycle = calculate_median_cycle(breathing_curve)
    min_amplitude, max_amplitude = median_cycle.min(), median_cycle.max()

    edges = np.linspace(min_amplitude, max_amplitude, num=n_bins + 1, endpoint=True)

    bins = np.digitize(breathing_curve, edges) - 1

    return bins


def calculate_phase(
    breathing_curve: np.ndarray, phase_range: Tuple[float, float] = (0, PI2)
) -> List[np.ndarray]:
    peaks = list(find_peaks(breathing_curve))

    # skip peaks at start/end of breathing curve since they are not reliable
    if peaks[0] == 0:
        peaks = peaks[1:]
    elif peaks[-1] == len(breathing_curve) - 1:
        peaks = peaks[:-1]

    phase = np.zeros_like(breathing_curve, dtype=np.float32) * np.nan

    for left_peak, right_peak in zip(peaks[:-1], peaks[1:]):
        n_timesteps = right_peak - left_peak
        phase[left_peak:right_peak] = np.linspace(
            phase_range[0], phase_range[1], num=n_timesteps
        )

    # fill incomplete cycles at start/end with phase of median cycle
    median_cycle = calculate_median_cycle(breathing_curve)

    median_cycle_phase = np.linspace(
        phase_range[0], phase_range[1], num=len(median_cycle)
    )
    len_start_part = peaks[0]
    len_end_part = len(breathing_curve) - peaks[-1]

    # if start/end part is longer than median cycle: repeat median cycle
    n_repeats = ceil(max(len_start_part, len_end_part) / len(median_cycle))
    median_cycle_phase = np.tile(median_cycle_phase, reps=n_repeats)

    # do the filling
    phase[:len_start_part] = median_cycle_phase[-len_start_part:]
    phase[-len_end_part:] = median_cycle_phase[:len_end_part]

    return np.split(phase, peaks)


def calculate_pseudo_average_phase(
    breathing_curve: np.ndarray,
    phase_range: Tuple[float, float] = (0, PI2),
    n_bins: int = 10,
) -> List[np.ndarray]:
    phase = calculate_phase(breathing_curve, phase_range=phase_range)

    min_phase, max_phase = phase_range[0], phase_range[1]
    abs_phase_range = max_phase - min_phase

    pseudo_average_phase = []
    for i_cycle, cycle_phase in enumerate(phase):
        shift = (abs_phase_range / n_bins) * (i_cycle % n_bins)

        shifted_cycle_phase = (cycle_phase - shift) % max_phase
        pseudo_average_phase.append(shifted_cycle_phase)

    return pseudo_average_phase


def calculate_phase_bins(breathing_curve: np.ndarray, n_bins: int = 10) -> np.ndarray:
    phase = calculate_phase(breathing_curve)
    edges = np.linspace(0, PI2, num=n_bins + 1, endpoint=True) - PI2 / (2 * n_bins)
    edges[edges < 0.0] = 0.0
    bins = []
    for cycle_phase in phase:
        bins.append(np.digitize(cycle_phase, edges) - 1)

    bins = np.hstack(bins)
    bins[bins == n_bins] = 0

    return bins


def save_curve(
    curve: np.ndarray,
    filepath: PathLike,
    scaling_factor: float = 1.0,
    format: str = "%.4f",
):
    np.savetxt(filepath, curve * scaling_factor, fmt=format)


def load_curve(
    filepath: PathLike,
    scaling_factor: float = 1.0,
) -> np.ndarray:
    curve = np.loadtxt(filepath, dtype=np.float32)
    curve = curve * scaling_factor

    return curve
