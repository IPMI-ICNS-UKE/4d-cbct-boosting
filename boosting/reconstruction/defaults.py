from pathlib import Path

# Docker defaults
DOCKER_IMAGE = "4d-cbct-boosting:latest"
DOCKER_PATH_PREFIX = Path("/host")
DOCKER_MOUNTS = {"/": {"bind": str(DOCKER_PATH_PREFIX), "mode": "rw"}}
