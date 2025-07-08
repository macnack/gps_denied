#!/bin/bash
docker run -d --rm \
    --ipc=host \
    --gpus all \
    --env="XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" \
    --volume="$(pwd)/..:/home/mackop/workspace" \
    --privileged \
    --network=host \
    --name=gps_denied \
    --user ":" \
    homography_estimation "$@"
