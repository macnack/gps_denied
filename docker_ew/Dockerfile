######################################################################
# 0. Base image – Python 3.10
######################################################################
FROM python:3.10-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

######################################################################
# 1. System-level deps (only what the wheels need)
######################################################################
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential      \
        git                  \
        wget curl            \
        libgl1               \
        libglib2.0-0         \
    && apt-get clean         \
    && rm -rf /var/lib/apt/lists/*

######################################################################
# 2. Python packages – **versions that support Py 3.10**
######################################################################
RUN pip install --upgrade --no-cache-dir pip \
 && pip install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.11.4 \
        matplotlib==3.8.4 \
        torch>=2.4.0 \
        torchvision==0.22.0 \
        opencv-python-headless==4.10.0.82 \
        requests \
        neptune \
        torchinfo \
        rich

######################################################################
# 3. Workspace
######################################################################
WORKDIR /workspace
CMD ["/bin/bash"]
