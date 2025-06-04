docker run -it \
    --env="XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR" \
    --volume="$(pwd)/..:/home/user/work" \
    --privileged \
    --network=host \
    --name=gps_denied \
    uav
