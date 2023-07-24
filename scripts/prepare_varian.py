import logging
from pathlib import Path

from boosting.logger import init_fancy_logging
from boosting.reconstruction.varian.processing import (
    extract_data_from_zip,
    prepare_for_reconstruction,
)

"""
This script will prepare Varian TrueBeam 4D CBCT raw data, i.e.
  - extract needed files
  - convert projection files to single projection stack [1]
  - air normalize projection stack [2]
  - create RTK geometry [3]
  - extract respiratory curve (phase and amplitude) from the projections [4]
  
The resulting folder/file structure will look like this:

phantom_scan/
├── calibrations
│   ├── AIR
│   │   └── Factory
│   │       ├── Bowtie.xim
│   │       ├── FilterBowtie.xim
│   │       ├── Filter.xim
│   │       └── Source.xim
│   └── AIR-Half-Bowtie-125KV
│       └── Current
│           ├── FilterBowtie.xim
│           └── Filter.xim
├── meta
│   ├── amplitudes.txt [cf. 4]
│   ├── geometry.xml [cf. 3]
│   ├── ImgParameters.h5
│   ├── phases.txt [cf. 4]
│   └── Scan.xml
├── projections
│   ├── Proj_00000.xim
│   ├── Proj_00001.xim
│   ├── ...    
│   ├── Proj_00892.xim
│   └── Proj_00893.xim
├── files.yaml
├── normalized_projections.mha [cf. 2]
└── projections.mha [cf. 1]
"""

logger = logging.getLogger(__name__)
init_fancy_logging()

logging.getLogger(__name__).setLevel(logging.DEBUG)
logging.getLogger("boosting").setLevel(logging.DEBUG)

folder = Path("/datalake_fast/4d_cbct_boosting")

output_folder = folder / "phantom_scan"

filepath = extract_data_from_zip(
    filepath=Path(folder / "phantom_scan.zip"), output_folder=output_folder
)

prepare_for_reconstruction(output_folder)
