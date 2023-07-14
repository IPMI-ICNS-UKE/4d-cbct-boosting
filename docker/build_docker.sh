#!/bin/bash
set -x
# build docker image
docker build --tag 4d-cbct-boosting:latest .
# do the advanced (CUDA) compiling
container_id=$(docker run -it --detach 4d-cbct-boosting:latest)
docker exec $container_id /bin/bash -c "chmod +x /compile.sh && /compile.sh"
docker stop $container_id
docker commit $container_id 4d-cbct-boosting:latest
