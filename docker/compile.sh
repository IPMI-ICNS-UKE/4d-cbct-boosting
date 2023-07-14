#!/bin/bash
set -x
# build ITK and RTK (using CUDA)
mkdir /ITK_build && cd /ITK_build &&
  cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DBUILD_TESTING:BOOL=OFF \
    -DBUILD_EXAMPLES:BOOL=OFF \
    -DModule_RTK:BOOL=ON \
    -DRTK_BUILD_APPLICATIONS:BOOL=ON \
    -DRTK_USE_CUDA:BOOL=ON \
    -DRTK_CUDA_PROJECTIONS_SLAB_SIZE:STRING=16 \
    ../ITK && make -j 32

# make everything executable
chmod -R 755 /ITK_build/bin

# link everything
ln -s /ITK_build/bin/* /usr/local/bin
