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


## Usage
Please see `boosting/learning/training.py`