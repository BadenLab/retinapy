FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
#FROM pytorch/pytorch:latest

# I edit the nvimrc too often for it to be a base image.
# FROM nvimi
ARG USER_ID=1001
ARG GROUP_ID=101
ARG USER=app
ENV USER=$USER
ARG PROJ_ROOT=/$USER
# This is for convenience within this dockerfile. If there was a non-argument
# variable type, I'd use it instead.
ARG NEOVIM_DIR=/home/$USER/.config/nvim

USER root
RUN groupadd --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

RUN mkdir $PROJ_ROOT && chown $USER $PROJ_ROOT
WORKDIR $PROJ_ROOT	

# These next two folders will be where we will mount our local data and out
# directories. We create them manually (automatic creation when mounting will
# create them as being owned by root, and then our program cannot use them).
RUN mkdir data && chown $USER data
RUN mkdir out && chown $USER out

# tzdata configuration stops for an interactive prompt without the env var.
# https://serverfault.com/questions/949991/how-to-install-tzdata-on-a-ubuntu-docker-image
# https://stackoverflow.com/questions/51023312/docker-having-issues-installing-apt-utils/56569081#56569081
ENV TZ=Europe/London
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-recommends -y \
	tzdata

RUN apt-get update && apt-get install -y --no-install-recommends \
	 curl \
	 ca-certificates \
	 git \
	 libsm6 \
	 libxext6 \
     libxrender-dev \
	 jq \
	 locales \
	 ffmpeg && \
	 rm -rf /var/lib/apt/lists/*

# Set the locale
RUN locale-gen en_US.UTF-8  
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8  

###############################################################################
#
# Conda
# 	
###############################################################################

# Set up the Conda environment (using Miniforge)
ENV PATH=/home/$USER/mambaforge/bin:$PATH
#COPY environment.yml /app/environment.yml
RUN curl -sLo ./mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/22.9.0-1/Mambaforge-22.9.0-1-Linux-x86_64.sh \
 && chmod +x ./mambaforge.sh \
 && ./mambaforge.sh -b -p /home/$USER/mambaforge \
 && rm ./mambaforge.sh \
# && mamba env update -n base -f /app/environment.yml \
# && rm /app/environment.yml \
 && mamba clean -ya


###############################################################################
# /Conda
###############################################################################

RUN pip install --upgrade pip
RUN pip install \
		torch \
		numpy \
		build \
		twine \
		h5py \
		opencv-python \
		pandas \
		jupyterlab \
		ipywidgets  \
		jupyterlab-vim \
		jupyter-dash \
		jupyterlab-spellchecker \
		matplotlib \
		pytest \
		scipy \
		plotly \
		scikit-learn \
		icecream \
		bidict \ 
		einops \
		ipympl \
		cprofilev \
		configargparse \
		pyyaml \
		tensorboard \
		kaleido \
		torchinfo \
		semantic_version \
		dash \
		scinot \
		flask \
		flask-sqlalchemy \
		polars \
		mypy

RUN conda install --yes -c conda-forge nodejs'>=12.12.0'

# From: https://stackoverflow.com/questions/67050036/enable-jupyterlab-extensions-by-default-via-docker
COPY --chown=$USER tools/jupyter_notebook_config.py /etc/jupyter/
# Try to get Jupyter Lab to allow extensions on startup.
# This file was found by diffing a container running jupyterlab that had 
# extensions manually enabled.
COPY --chown=$USER tools/plugin.jupyterlab-settings /home/$USER/.jupyter/lab/user-settings/@jupyterlab/extensionmanager-extension/
# Keyboard shortcuts
COPY --chown=$USER tools/shortcuts.jupyterlab-settings /home/$USER/.jupyter/lab/user-settings/@jupyterlab/shortcuts-extension/
# Getting some permission errors printed in terminal after running Jupyter Lab, 
# and trying the below line to fix:
RUN chown -R $USER:$USER /home/$USER/.jupyter


RUN jupyter lab build # needed for Dash. Very annoying. Takes ages.


RUN pip install scikit-image moviepy


USER root
# In order to allow the Python package to be edited without
# a rebuild, install all code as a volume. We will still copy the
# files initially, so that things like the below pip install can work.
COPY --chown=$USER ./ ./

# Install our own project as a module.
# This is done so the tests and JupyterLab code can import it.
RUN pip install -e ./retinapy

# Switching to our new user. Do this at the end, as we need root permissions 
# in order to create folders and install things.
USER $USER

