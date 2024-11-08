#!/bin/bash

# install conda
apt update && apt install git nano wget

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda


apt install libgl1-mesa-glx libglib2.0-0

conda create -n unified-io-2 python=3.8

conda activate unified-io-2
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

conda env update -f environment.yaml