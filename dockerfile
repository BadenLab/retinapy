FROM nvidia/cuda:11.5.1-base-ubuntu20.04
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

USER root
ARG PROJ_ROOT=/app

RUN mkdir $PROJ_ROOT && chown $USER $PROJ_ROOT
WORKDIR $PROJ_ROOT	

# These next two folders will be where we will mount our local data and out
# directories. We create them manually (automatic creation when mounting will
# create them as being owned by root, and then our program cannot use them).
RUN mkdir data && chown $USER data
RUN mkdir out && chown $USER out

# When switching to mounting the whole project as a volume, it
# seemed wrong to mount it at the existing /home/app directory. So,
# going one level deeper. I think an alternative is to just work from
# /app, but I think some programs like JupyterLab have some issues
# when running from outside the home tree.
WORKDIR /home/$USER

# tzdata configuration stops for an interactive prompt without the env var.
# https://serverfault.com/questions/949991/how-to-install-tzdata-on-a-ubuntu-docker-image
# https://stackoverflow.com/questions/51023312/docker-having-issues-installing-apt-utils/56569081#56569081
ENV TZ=Europe/London
RUN DEBIAN_FRONTEND=noninteractive \
	apt-get update && apt-get install --no-install-recommends -y \
	 tzdata

RUN apt-get update && apt-get install -y --no-install-recommends \
	 curl \
	 ca-certificates \
	 git \
	 libsm6 \
	 libxext6 \
     libxrender-dev \
	 ffmpeg && \
	 rm -rf /var/lib/apt/lists/*

###############################################################################
#
# Conda
# 	
###############################################################################

# Set up the Conda environment (using Miniforge)
ENV PATH=/home/$USER/mambaforge/bin:$PATH
#COPY environment.yml /app/environment.yml
RUN curl -sLo ./mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/4.12.0-2/Mambaforge-4.12.0-2-Linux-x86_64.sh \
 && chmod +x ./mambaforge.sh \
 && ./mambaforge.sh -b -p ./mambaforge \
 && rm ./mambaforge.sh \
# && mamba env update -n base -f /app/environment.yml \
# && rm /app/environment.yml \
 && mamba clean -ya

###############################################################################
# /Conda
###############################################################################


###############################################################################
#
# Neovim
# 	
###############################################################################

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
	add-apt-repository ppa:neovim-ppa/unstable

RUN apt-get update && apt-get install -y --no-install-recommends \
	neovim  \
	gcc \
	make \
	autoconf \
	automake \
	locales \
	# For airline font support
	fonts-powerline && \
	rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
	pip install neovim

# Set the locale
RUN locale-gen en_US.UTF-8  
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8  

RUN conda install --yes -c conda-forge nodejs'>=12.12.0' --repodata-fn=repodata.json

###############################################################################
# \Neovim
###############################################################################

RUN conda config --add channels conda-forge 
RUN conda install --yes \
	python=3.9 \
	pip \
	pytest \
	numpy \
	pandas \
	scipy \
	pytorch \
	jupyterlab  \
	matplotlib \
	plotly \
	build \
	twine \
	h5py \
    #ipykernel>=6 \
    # xeus-python \
    ipywidgets && \
	conda clean -ya

RUN conda install -c conda-forge jupyterlab-spellchecker
#RUN jupyter labextension install jupyterlab_vim
# From: https://stackoverflow.com/questions/67050036/enable-jupyterlab-extensions-by-default-via-docker
COPY --chown=$USER tools/jupyter_notebook_config.py /etc/jupyter/
# Try to get Jupyter Lab to allow extensions on startup.
# This file was found by diffing a container running jupyterlab that had 
# extensions manually enabled.
COPY --chown=$USER tools/plugin.jupyterlab-settings /home/$USER/.jupyter/lab/user-settings/@jupyterlab/extensionmanager-extension/
# Getting some permission errors printed in terminal after running Jupyter Lab, 
# and trying the below line to fix:
RUN chown -R $USER:$USER /home/$USER/.jupyter

RUN pip install --upgrade pip
RUN pip install \
        opencv-python \
		jupyterlab-vim \
        icecream \
		bidict \ 
		einops \
		ipympl \
		cprofilev \
		mypy 

###############################################################################
# Neovim
###############################################################################
# For some reason, /home/$USER/.config is owned by root. 
RUN mkdir -p /home/$USER/.config  && chown $USER:$USER /home/$USER/.config

COPY --chown=$USER_ID tools/nvim $NEOVIM_DIR

# Currently, assume that NeoSolarized file is copied.
# RUN git clone https://github.com/overcache/NeoSolarized.git
# RUN mkdir -p $NEOVIM_DIR/colors/ && chown $USER_ID $NEOVIM_DIR/colors
# RUN cp ./NeoSolarized/colors/NeoSolarized.vim $NEOVIM_DIR/colors/

# Switching to our new user. Do this at the end, as we need root permissions 
# in order to create folders and install things.
USER $USER

RUN curl -fLo /home/$USER/.local/share/nvim/site/autoload/plug.vim --create-dirs \
       https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
RUN nvim --headless +PlugInstall +qa
RUN nvim --headless -c "CocInstall -sync coc-pyright coc-html | qa"

###############################################################################
# /Neovim
###############################################################################

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

