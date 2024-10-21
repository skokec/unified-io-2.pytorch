#!/bin/bash

# HPC-Vega:
# GPUS_PER_NODE=4
# STORAGE_DIR=/ceph/hpc/data/FRI/tabernikd/

# HPC-Arnes:
GPUS_PER_NODE=2
STORAGE_DIR=/d/hpc/projects/FRI/tabernikd/

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# PROJECT specific settings
module purge
module load Anaconda3

USE_CONDA_ENV=${USE_CONDA_ENV:-unified-io-2}

if ! conda env list | grep -q "$USE_CONDA_ENV"; then
    "Conda envirionment $USE_CONDA_ENV does not exists -- will create it first"
    conda env create -y -n "$ENV_NAME" python=3.11

    # and install ccc-tools
    pip install git+https://github.com/vicoslab/ccc-tools
fi

# manually call bashrc to setup conda init
source ~/.bashrc

conda activate $USE_CONDA_ENV
echo "Using conda env '$USE_CONDA_ENV'"

###################################################
######## INPUT/OUTPUT PATH
###################################################

export SOURCE_DIR=${SOURCE_DIR:-$(realpath "$(dirname $BASH_SOURCE)/..")}

# add source dir to python path
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR

###################################################
######## DATASET PATHS
###################################################

export STORAGE_DIR=${STORAGE_DIR:-/storage/}

export LLAMA2_TOKENIZER=$(realpath ~/Projects/llama2_tokenizer.model)
export VICOS_TOWEL_DATASET=$STORAGE_DIR/datasets/ViCoSTowelDataset

###################################################
######## DATA PARALLEL SETTINGS
###################################################
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_BLOCKING_WAIT=0
export NCCL_SHM_DISABLE=0

###################################################
######## CCC CLUSTER INFO
###################################################

export CLUSTER_INFO_FILE=${CLUSTER_INFO_FILE:-$STORAGE_DIR/private/vicos/vicos_cluster_info.json}