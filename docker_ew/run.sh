docker run -it \
    --ipc=host \
    --gpus all \
    --env="XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" \
    --volume="$(pwd)/..:/home/user/work" \
    --privileged \
    --network=host \
    --name=gps_denied_new \
    uav10
