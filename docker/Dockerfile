FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-11.8 \
    CONDA_ENV_NAME=theseus \
    PYTHON_VERSION=3.10 \
    CONDA_DIR=/opt/conda

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    wget \
    git \
    bzip2 \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    curl \
    imagemagick \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda create -y -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION}
ENV PATH=$CONDA_DIR/envs/${CONDA_ENV_NAME}/bin:$PATH
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc

SHELL ["conda", "run", "-n", "theseus", "/bin/bash", "-c"]

# Install PyTorch (CUDA 11.8)
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu118

RUN conda install -y -c conda-forge suitesparse

RUN pip install --upgrade pip && \
    pip install \
        numpy==1.26.4 \
        scipy==1.11.4 \
        matplotlib==3.8.4 \
        torchvision==0.22.0 \
        opencv-python-headless==4.10.0.82 \
        requests \
        neptune \
        torchinfo \
        rich \
        kornia \
        hydra-core --upgrade

COPY ./theseus_ai-0.2.2-cp310-cp310-manylinux_2_17_x86_64.whl /tmp/
RUN pip install /tmp/theseus_ai-0.2.2-cp310-cp310-manylinux_2_17_x86_64.whl

ARG USERNAME=maciek
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash && \
    usermod -aG sudo $USERNAME && \
    apt-get update && apt-get install -y sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R $USERNAME:$USERNAME /var/lib/apt/lists

USER $USERNAME
WORKDIR /home/$USERNAME/workspace

# Auto-activate conda env + colorful prompt
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate ${CONDA_ENV_NAME}" >> ~/.bashrc && \
    echo "alias ll='ls -alF --color=auto'" >> ~/.bashrc && \
    echo "alias la='ls -A --color=auto'" >> ~/.bashrc && \
    echo "alias l='ls -CF --color=auto'" >> ~/.bashrc && \
    echo "PS1='\[\e[1;32m\]\u@\h:\w \$\[\e[0m\] '" >> ~/.bashrc

CMD [ "bash" ]