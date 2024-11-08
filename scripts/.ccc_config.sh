#!/bin/bash

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# PROJECT specific settings
###################################################
######## ACTIVATE CONDA ENV
###################################################
echo "Loading conda env ..."

USE_CONDA_HOME=${USE_CONDA_HOME:-~/conda}
USE_CONDA_ENV=${USE_CONDA_ENV:-unified-io-2}

. $USE_CONDA_HOME/etc/profile.d/conda.sh

conda activate $USE_CONDA_ENV
echo "... done - using $USE_CONDA_ENV"

###################################################
######## INPUT/OUTPUT PATH
###################################################

export SOURCE_DIR=${SOURCE_DIR:-$(realpath "$(dirname $BASH_SOURCE)/..")}
export OUTPUT_DIR=${OUTPUT_DIR:-$(realpath "$(dirname $BASH_SOURCE)/../exp")}

# add source dir to python path
export PYTHONPATH=$PYTHONPATH:$SOURCE_DIR


###################################################
######## DATASET PATHS
###################################################

export STORAGE_DIR=${STORAGE_DIR:-/storage/}

export LLAMA2_TOKENIZER="/home/domen/Projects/vision-language/llama2/tokenizer.model"
export VICOS_TOWEL_DATASET=$STORAGE_DIR/datasets/ClothDataset/ClothDatasetVICOS
#export VICOS_TOWEL_DATASET=/storage/local/ssd/cache/ClothDatasetVICOS
export VICOS_MUJOCO_DATASET=$STORAGE_DIR/datasets/ClothDataset/MuJoCo

###################################################
######## DATA PARALLEL SETTINGS
###################################################
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth2}

###################################################
######## CCC CLUSTER INFO
###################################################

export CLUSTER_INFO_FILE=${CLUSTER_INFO_FILE:-$STORAGE_DIR/private/vicos/vicos_cluster_info.json}

