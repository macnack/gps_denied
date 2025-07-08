docker run -d \
    --ipc=host \
    --gpus all \
    --env="XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" \
    --volume="$(pwd)/..:/home/maciek/workspace" \
    --privileged \
    --network=host \
    --name=gps_denied_proj \
    uav
