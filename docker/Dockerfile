# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

ARG BASE_IMAGE_NAME="ubuntu"
ARG BASE_IMAGE_TAG="22.04"

FROM ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends --fix-missing -y \
    ca-certificates \
    wget \
    git &&\
    apt-get clean && \
    apt-get purge && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# Default Workspace
RUN mkdir -p /workspace

SHELL ["/bin/bash", "-c"]
# Install reqs via conda
ARG CONDA_INSTALL_PATH=/opt/conda
ARG MINICONDA_VERSION="latest"
# Miniconda Installation
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p ${CONDA_INSTALL_PATH} && \
    rm miniconda.sh && \
    ln -s ${CONDA_INSTALL_PATH}/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". ${CONDA_INSTALL_PATH}/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH="${CONDA_INSTALL_PATH}/bin:${PATH}"
# Install libmamba and set as default solver
RUN conda install -n base conda-libmamba-solver -y && \
    conda config --set solver libmamba
# Create Conda Environment + Install reqs via conda
WORKDIR /workspace
COPY env env
RUN INTEL_PATH=./env/intel_env.yml && \
    ENV_NAME=$(awk '/name:/  {print $2}' "$INTEL_PATH") && \
    conda env create -f "$INTEL_PATH" && \
    conda install -n "$ENV_NAME" -c intel ipykernel -y && \
    conda create -n jupyter_server -c intel nb_conda_kernels notebook -y
# Clean steps
RUN rm -rf env/ && \
    conda clean -ya
