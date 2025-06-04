FROM python:3.6-slim-buster

ENV DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------
# 1. System-level build/runtime dependencies for SciPy-stack + OpenCV
# --------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    curl \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk2.0-dev \
    libatlas-base-dev \
    gfortran \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------
# 2. Python packages – same pinned versions
# --------------------------------------------------------------------
RUN pip install --no-cache-dir \
    numpy==1.13.3 \
    scipy==0.19.1 \
    matplotlib==2.0.2 \
    torch==1.0.0 \
    torchvision==0.2.0 \
    opencv-python
# --------------------------------------------------------------------
# 3. Build OpenCV 3.3.0 (Python 3 bindings only)
# --------------------------------------------------------------------
WORKDIR /opt
# RUN git clone --depth 1 --branch 3.3.0 https://github.com/opencv/opencv.git \
#  && mkdir -p opencv/build \
#  && cd opencv/build \
#  && cmake -D CMAKE_BUILD_TYPE=RELEASE \
#           -D CMAKE_INSTALL_PREFIX=/usr/local \
#           -D PYTHON3_EXECUTABLE=$(which python) \
#           -D BUILD_opencv_python3=ON \
#           -D BUILD_EXAMPLES=OFF .. \
#  && make -j"$(nproc)" \
#  && make install \
#  && ldconfig \
#  && rm -rf /opt/opencv

# --------------------------------------------------------------------
# 4. Default workspace
# --------------------------------------------------------------------
WORKDIR /workspace
CMD ["/bin/bash"]
