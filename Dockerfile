FROM ubuntu:20.04 as base

ENV DEBIAN_FRONTEND=noninteractive
ENV DBUS_SESSION_BUS_ADDRESS=unix:path=/var/run/dbus/system_bus_socket

COPY environment.yml /coursera_deep_learning/environment.yml

WORKDIR /coursera_deep_learning

RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get update && apt-get install build-essential software-properties-common -y \
    sudo \
    git \
    cmake \
    wget \
    zsh \
    graphviz \
    ffmpeg \
    unzip \
    libgl1-mesa-dev \
    libhdf5-dev \
    libfreetype6-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    libfreeimage-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH=$PATH:/opt/conda/bin

RUN conda env create -n coursera_deep_learning --file environment.yml

# SHELL ["conda", "run", "-n", "coursera_deep_learning", "/bin/bash", "-c"]

# ENV CONDA_DEFAULT_ENV=coursera_deep_learning

# RUN echo "conda activate coursera_deep_learning" >> ~/.bashrc