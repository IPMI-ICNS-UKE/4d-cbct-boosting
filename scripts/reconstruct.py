import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from boosting.logger import init_fancy_logging
from boosting.reconstruction import binning
from boosting.reconstruction.binning import load_curve
from boosting.reconstruction.presets import FDK, ROOSTER4D
from boosting.reconstruction.reconstructors import (
    FDKReconstructor,
    ROOSTER4DReconstructor,
)
from boosting.reconstruction.varian.processing import interpolate_nan_phases

"""
This script will reconstruct 4D CBCT raw data using RTK. Especially, it can handle the
Varian TrueBeam 4D CBCT data extracted by the previous script (prepare_varian.py).
Of course any raw data can be feeded into this reconstruction pipeline as long as
it is in the right RTK format.

The following folder/file structure is required (the root folder name, 
here 'phantom_scan', can be arbitrarily chosen): 

phantom_scan/
├── meta
│   ├── amplitudes.txt
│   ├── geometry.xml
│   └── phases.txt
└── normalized_projections.mha

"""


logger = logging.getLogger(__name__)
init_fancy_logging()

logging.getLogger(__name__).setLevel(logging.DEBUG)
logging.getLogger("boosting").setLevel(logging.DEBUG)

input_folder = Path("/datalake_fast/4d_cbct_boosting/phantom_scan")


output_folder = input_folder / "reconstructions"
output_folder.mkdir(exist_ok=True)

reconstructor = FDKReconstructor(use_docker=True, gpu_id=0)
reconstructor.reconstruct(
    path=input_folder,
    regexp="normalized_projections.mha",
    geometry=input_folder / "meta" / "geometry.xml",
    hardware="cuda",
    pad=FDK["pad"],
    hann=FDK["hann"],
    hannY=FDK["hann_y"],
    dimension=FDK["dimension"],
    spacing=FDK["spacing"],
    output_filepath=output_folder / "fdk3d.mha",
)


binning_methods = {"phase": {}, "recalc_phase": {}, "pseudo_average": {}}

varian_amplitude_signal = load_curve(input_folder / "meta" / "amplitudes.txt")
varian_phase_signal = load_curve(input_folder / "meta" / "phases.txt")


for binning_method in binning_methods.keys():
    if binning_method == "phase":
        varian_phase_signal = interpolate_nan_phases(varian_phase_signal)
        phase_signal = varian_phase_signal / 360.0
    elif binning_method == "recalc_phase":
        phase_signal = binning.calculate_phase(
            varian_amplitude_signal, phase_range=(0, 1)
        )
        phase_signal = np.hstack(phase_signal)
    elif binning_method == "pseudo_average":
        pseudo_average_phase = binning.calculate_pseudo_average_phase(
            varian_amplitude_signal, phase_range=(0, 1)
        )
        pseudo_average_phase = np.hstack(pseudo_average_phase)
        phase_signal = pseudo_average_phase
    else:
        raise RuntimeError(f"Unknown binning {binning}")

    binning_methods[binning_method]["phase_signal"] = phase_signal

    reconstructor = ROOSTER4DReconstructor(
        phase_signal=phase_signal, use_docker=True, gpu_id=0
    )
    reconstructor.reconstruct(
        path=input_folder,
        regexp="normalized_projections.mha",
        geometry=input_folder / "meta" / "geometry.xml",
        fp=ROOSTER4D["fp"],
        bp=ROOSTER4D["bp"],
        dimension=ROOSTER4D["dimension"],
        spacing=ROOSTER4D["spacing"],
        niter=ROOSTER4D["n_iter"],
        cgiter=ROOSTER4D["cg_iter"],
        tviter=ROOSTER4D["tv_iter"],
        gamma_time=ROOSTER4D["gamma_time"],
        gamma_space=ROOSTER4D["gamma_space"],
        output_filepath=output_folder / f"rooster4d_{binning_method}.mha",
    )

with plt.ioff():
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(varian_amplitude_signal, label="amplitude")
    for binning_method, binning_config in binning_methods.items():
        ax[1].plot(binning_config["phase_signal"], label=binning_method)

    for _ax in ax:
        _ax.grid(True)
        _ax.legend()

    plt.savefig(output_folder / "respiratory_binning.png", dpi=600)
