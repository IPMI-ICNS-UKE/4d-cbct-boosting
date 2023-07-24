# Self-Contained Boosting of 4D Cone-Beam CT

This repository contains the official code for the following Medical Physics [paper](https://doi.org/10.1002/mp.14441):

```
@article{Madesta2020,
  author = {Frederic Madesta and Thilo Sentker and Tobias Gauer and Ren{\'{e}} Werner},
  title = {Self-Contained Deep Learning-Based Boosting of 4D Cone-Beam {CT} Reconstruction},
  journal = {Medical Physics},
  year = {2020},
  month = Aug,
  publisher = {Wiley},
  doi = {10.1002/mp.14441},
  url = {https://doi.org/10.1002/mp.14441},
}
```

## Installation
### Build Docker
Internally, this project uses [RTK](https://github.com/RTKConsortium/RTK) for reconstructing CBCT images.
As compiling CUDA-related stuff can be cumbersome, we provide a Docker image with batteries included.
In order to build this image you need to perform the following steps:

0. You need a machine with a NVIDIA GPU and installed NVIDIA drivers
1. Install [Docker](https://docs.docker.com/engine/install/ubuntu/)
   and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#setting-up-nvidia-container-toolkit) (if not already installed)
3. Build the Docker image by executing `cd docker && ./build_docker.sh`. This will take some time.

## Phantom data set
We privide a [4D CBCT phantom data set](https://icns-nas1.uke.uni-hamburg.de/owncloud10/index.php/s/94bs82gk3hxOBzD) for test purposes.

Details:
- 4D CBCT Scanner: Varian TrueBeam
- Phantom: [Dynamic Thorax Phantom: Model 008A](https://www.cirsinc.com/products/radiation-therapy/dynamic-thorax-motion-phantom/)
- The following scans are included:
  1. SI amplitude of insert: ±10mm, pattern: sin, period: 5.0s
  2. SI amplitude of insert: ±10mm, pattern: cos**4, period: 5.0s
  3. SI amplitude of insert: ±10mm, pattern: sin, period: 2.5s
  4. SI amplitude of insert: ±10mm, pattern: cos**4, period: 2.5s
  5. SI amplitude of insert: ±10mm, pattern: sin, period: 7.5s
  6. SI amplitude of insert: ±10mm, pattern: cos**4, period: 7.5s

## Usage
The following scripts are included inside the `scripts` folder:
- `prepare_varian.py`:  
This script will prepare Varian TrueBeam 4D CBCT raw data, i.e.
  - extract needed files
  - convert projection files to single projection stack
  - air normalize projection stack
  - create RTK geometry
  - extract respiratory curve (phase and amplitude) from the projections

- `reconstruct.py`:  
This script will reconstruct 4D CBCT raw data using RTK. Especially, it can handle the
Varian TrueBeam 4D CBCT data extracted by the previous script (`prepare_varian.py`).
Of course any raw data can be feeded into this reconstruction pipeline as long as
it is in the right RTK format.

- `train.py`:  
This script will train the 4D CBCT boosting network on the reconstructed data 
(you can use the provided phantom data set for test purposes). In the end, the trained 
- model is applied to the 4D CBCT phase images.