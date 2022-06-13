import numpy as np
import h5py as h5
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import os
from scipy.ndimage import uniform_filter1d
from scipy.signal import detrend, savgol_filter
from typing import List, Tuple
from dataclasses import dataclass
from math import ceil
from boosting.constants import PI2


@dataclass
class RespiratoryStatistics:
    mean_cycle_period: float
    median_cycle_period: float
    std_cycle_period: float
    n_complete_cycles: float


def read_curve(image_params_filepath):
    amplitudes = []
    phases = []
    with h5.File(image_params_filepath, "r") as f:
        for p in range(len(f["ImageParameters"])):
            projection_number = str(p).zfill(5)
            try:
                current_amplitude = f["ImageParameters"][projection_number].attrs[
                    "GatingAmplitude"
                ]
                current_phase = f["ImageParameters"][projection_number].attrs[
                    "GatingPhase"
                ]
            except KeyError:
                current_amplitude = np.nan
                current_phase = np.nan
            amplitudes.append(current_amplitude)
            phases.append(current_phase)

    return (
        np.array(amplitudes, dtype=np.float32).squeeze(),
        np.array(phases, dtype=np.float32).squeeze(),
    )


def read_curve_from_zip(zip_filepath):
    with TemporaryDirectory() as temp_dir:
        with ZipFile(zip_filepath, "r") as zip_file:
            compressed_files = zip_file.namelist()
            for compressed_file in compressed_files:
                if compressed_file.endswith("ImgParameters.h5"):
                    zip_file.extract(compressed_file, temp_dir)
                    return read_curve(os.path.join(temp_dir, compressed_file))
            raise FileNotFoundError("Could not find ImgParameters.h5!")


def find_peaks(x, scale=None, debug=False):
    """Find peaks in quasi-periodic noisy signals using AMPD algorithm.
    Extended implementation handles peaks near start/end of the signal.
    Optimized implementation by Igor Gotlibovych, 2018

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
    x = detrend(x)
    N = len(x)
    L = N // 2
    if scale:
        L = min(scale, L)

    # create LSM matix
    LSM = np.ones((L, N), dtype=bool)
    for k in np.arange(1, L + 1):
        LSM[k - 1, 0: N - k] &= x[0: N - k] > x[k:N]  # compare to right neighbours
        LSM[k - 1, k:N] &= x[k:N] > x[0: N - k]  # compare to left neighbours

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


def split_into_cycles(curve: np.ndarray, peaks: np.ndarray = None) -> List[np.ndarray]:
    if peaks is None:
        peaks = find_peaks(curve)
    # discard potentially incomplete first and last cycle
    if peaks[0] == 0:
        peaks = peaks[1:]
    elif peaks[-1] == len(curve) - 1:
        peaks = peaks[:-1]
    return np.split(curve, peaks)


def align_cycles(cycles: List[np.ndarray]) -> np.ndarray:
    minimum_indices = [np.argmin(c) for c in cycles]

    lefts = [c[:min_idx] for (c, min_idx) in zip(cycles, minimum_indices)]
    rights = [c[min_idx:] for (c, min_idx) in zip(cycles, minimum_indices)]

    max_left_length = max(len(l) for l in lefts)
    max_right_length = max(len(r) for r in rights)

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


def calculate_respiratory_statistics(amplitudes: np.ndarray) -> RespiratoryStatistics:
    cycles = split_into_cycles(amplitudes)
    cycle_lengths = [len(c) for c in cycles]
    return RespiratoryStatistics(
        mean_cycle_period=float(np.mean(cycle_lengths)),
        median_cycle_period=float(np.median(cycle_lengths)),
        std_cycle_period=float(np.std(cycle_lengths)),
        n_complete_cycles=len(cycle_lengths),
    )


def calculate_median_cycle(curve: np.ndarray) -> np.ndarray:
    cycles = split_into_cycles(curve)
    resp_stats = calculate_respiratory_statistics(amplitudes)

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
    bins[bins == 10] = 0

    return bins


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from boosting.binning.phase import PhaseBinning, PseudoAverageBinning
    from boosting.utils import rescale_range
    from scipy.signal import hilbert

    amplitudes, phases = read_curve_from_zip(
        "/datalake_fast/daten_lukas/Recon_Frederic_Phantomstudie/2022-04-07_173010.zip"
    )

    p = find_peaks(amplitudes)
    t = np.arange(len(amplitudes))
    median_cycle = calculate_median_cycle(amplitudes)

    amplitude_bins = calculate_amplitude_bins(amplitudes, n_bins=5)
    phase_bins = calculate_phase_bins(amplitudes)

    amplitude_bins[amplitude_bins < 0] = 0
    amplitude_bins[amplitude_bins >= 10] = 9



    # bins[bins < 0] = 0
    # bins[bins > 9] = 9

    np.savetxt(
        '/datalake_fast/daten_lukas/Recon_Frederic_Phantomstudie/tmp/meta/signal.txt',
        amplitude_bins,
        fmt='%.4f'
    )

    peaks = find_peaks(amplitudes)
    _amplitude = split_into_cycles(amplitudes, peaks=peaks)
    _amplitude_bins = split_into_cycles(amplitude_bins, peaks=peaks)
    _phase_bins = split_into_cycles(phase_bins, peaks=peaks)


    b = []
    for (a, ab, pb) in zip(
            _amplitude,
            _amplitude_bins,
            _phase_bins
    ):
        min_idx = np.argmin(a)
        exhale_cycle = a[:min_idx]
        inhale_cycle = a[min_idx:]

        exhale_ab = -(ab[:min_idx] - 5)
        inhale_ab = ab[min_idx:] + 5


        transformed = np.hstack((exhale_ab, inhale_ab))

        b.append(transformed)

    b = np.hstack(b)
    b = np.clip(b, 0, 9)

    np.savetxt(
        '/datalake_fast/daten_lukas/Recon_Frederic_Phantomstudie/tmp/meta/phase_bins.txt',
        phase_bins / 10.0,
        fmt='%.4f'
    )

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(amplitudes)
    ax[0].scatter(t[p], amplitudes[p], marker="x")
    ax[1].plot(amplitude_bins)
    ax[1].plot(phase_bins)
    ax[2].plot(b)


    # fig, ax = plt.subplots(2, sharex=True)
    # ax[0].plot(a)
    # ax[1].plot(ab)
    # ax[1].plot(pb)
    # ax[1].plot(transformed)
    #
    # if len(a) == 77:
    #     break

    # amplitudes -= amplitudes.mean()
    # h = hilbert(amplitudes)
    # instantaneous_phase = np.unwrap(np.angle(h))
    #
    #
    #
    #
    #
    # plt.figure()
    # plt.plot(median_cycle)
    #
    # fig, ax = plt.subplots(2, sharex=True)
    # ax[0].plot(amplitudes)
    # ax[0].scatter(t[p], amplitudes[p], marker="x")
    # ax[1].plot(amplitude_bins)
    # ax[1].plot(phase_bins)

    # median_cycle = calculate_median_cycle(amplitudes)
    # amplitudes2 = rescale_range(
    #     amplitudes,
    #     input_range=(median_cycle.min(), median_cycle.max()),
    #     output_range=(0, 1)
    # )
    #
    # cycles = split_into_cycles(amplitudes)
    #
    # amplitudes -= amplitudes.mean()
    #
    # h = hilbert(amplitudes)
    # instantaneous_phase = np.unwrap(np.angle(h))
    #
    # fig, ax = plt.subplots(2, sharex=True)
    # ax[0].plot(amplitudes)
    # ax[1].plot(np.angle(h))

    # p = calculate_phase(amplitudes)
    # pa = calculate_pseudo_average_phase(amplitudes)
    #
    # p = np.hstack(p)
    # pa = np.hstack(pa)
    #
    # p360 = np.hstack(p) / PI2 * 360.0
    #
    # binner = PhaseBinning()
    # binner.signal = p360
    #
    # bp = calculate_phase_bins(amplitudes)
    # ba = calculate_amplitude_bins(amplitudes)
    #
    #
    # m = calculate_median_cycle(amplitudes)
    # fig, ax = plt.subplots()
    # ax2 = ax.twinx()
    # ax.plot(amplitudes, label="amplitudes")
    # ax2.plot(p, label="p", c="orange")
    # ax2.plot(pa, label="pa", c="red")
    # ax3.plot(np.hstack(p), label="phases", c="green")
    # ax4.plot(ba, label="phases", c="red")

    # p = find_peaks(curve)
    #
    # t = np.arange(len(curve))
    # fig, ax = plt.subplots()
    # ax.plot(t, curve)
    # ax.scatter(t[p], curve[p], marker="x")
    #
    # median_cycle = calculate_median_cycle(curve)
    #
    # plt.figure()
    # plt.plot(median_cycle)
    #
    # edges = calculate_amplitude_bins(curve)
    #
    # for e in edges:
    #     ax.plot([e] * len(t))
