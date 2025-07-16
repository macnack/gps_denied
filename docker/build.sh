#!/bin/bash
set -e  # Exit on error

# Define expected wheel path
WHEEL_PATH="theseus_docker_3.10/theseus_ai-0.2.2-cp310-cp310-manylinux_2_17_x86_64.whl"
TARGET_WHEEL="theseus_ai-0.2.2-cp310-cp310-manylinux_2_17_x86_64.whl"

# Only build the wheel if it doesn't exist
if [ ! -f "$TARGET_WHEEL" ]; then
    echo "🔧 .whl file not found, building wheel..."
    bash ../third_party/theseus/build_scripts/build_wheel.sh . 0.2.2 11.8

    if [ ! -f "$WHEEL_PATH" ]; then
        echo "❌ Build failed or .whl not found at: $WHEEL_PATH"
        exit 1
    fi

    mv "$WHEEL_PATH" .
else
    echo "✅ Found existing wheel: $TARGET_WHEEL"
fi

# Gather user info
USERNAME=$(whoami)
USER_UID=$(id -u)
USER_GID=$(id -g)

# Build Docker image
docker build \
  --build-arg USERNAME=$USERNAME \
  --build-arg USER_UID=$USER_UID \
  --build-arg USER_GID=$USER_GID \
  -t homography_estimation .

# Generate run.sh
cat <<EOF > run.sh
#!/bin/bash
docker run -d --rm \\
    --ipc=host \\
    --gpus all \\
    --env="XDG_RUNTIME_DIR=\$XDG_RUNTIME_DIR" \\
    --volume="$(cd "$(dirname "$0")/.." && pwd):/home/$USERNAME/workspace" \\
    --privileged \\
    --network=host \\
    --name=gps_denied \\
    --user "$USER_UID:$USER_GID" \\
    homography_estimation "\$@"
EOF

chmod +x run.sh
echo "✅ Docker image built and run.sh generated — use it like: ./run.sh bash"
