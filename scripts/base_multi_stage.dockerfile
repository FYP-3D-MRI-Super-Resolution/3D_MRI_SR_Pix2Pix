FROM nvcr.io/nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04 AS build

# https://medium.com/the-artificial-impostor/smaller-docker-image-using-multi-stage-build-cb462e349968

ARG PYTHON_VERSION=3.8.5
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda

# Fix for running cv2: ffmpeg libsm6 libxext6
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN conda install -y python=$PYTHON_VERSION && \
    conda install pytorch torchvision cudatoolkit=11.0 -c pytorch && \
    conda install -y matplotlib numpy=>1.19.2 Pillow>=8.0.1 && \
    conda install -y scipy six && \
    pip install nibabel opencv-python==4.4.0.44 torchio==0.17.50 visdom && \
    conda clean --yes --all

FROM nvcr.io/nvidia/cuda:11.2.1-base

ARG CONDA_DIR=/opt/conda
ENV PATH /opt/conda/bin:$PATH

COPY --from=build $CONDA_DIR $CONDA_DIR

WORKDIR Pix2PixNIfTI
ADD . .