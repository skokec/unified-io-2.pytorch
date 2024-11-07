#!/bin/bash

STORAGE_DIR_VEGA=/ceph/hpc/home/tabernikd/storage/
STORAGE_DIR_ARNES=/d/hpc/projects/FRI/tabernikd/
STORAGE_DIR_FRIDA=/shared/home/domen.tabernik/

# set storage dir based on which HPC is active
if [ -d "$STORAGE_DIR_VEGA" ]; then 
    echo "Detected HPC-VEGA"
    STORAGE_DIR=$STORAGE_DIR_VEGA
elif [ -d "$STORAGE_DIR_ARNES" ]; then 
    echo "Detected HPC-ARNES"
    STORAGE_DIR=$STORAGE_DIR_ARNES
elif [ -d "$STORAGE_DIR_FRIDA" ]; then
    echo "Detected HPC-FRIDA"
    STORAGE_DIR=$STORAGE_DIR_FRIDA
fi

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# PROJECT specific settings
USE_CONDA_ENV=${USE_CONDA_ENV:-unified-io-2}

if [ "$SLURM_CLUSTER_NAME" == "frida" ]; then
    # FRIDA does not have modules but instead requires using containers 
    # -- this config will be run once in main script on login node where conda and modules are not present, 
    # -- and once within container where conda is present

    SLURM_CONTAINER_IMG="$STORAGE_DIR/containers/unified-io-2.sqfs"
    SLURM_CONTAINER_MOUNTS="$STORAGE_DIR/:$STORAGE_DIR/,$STORAGE_DIR/Projects/llama2_tokenizer.model:/root/Projects/llama2_tokenizer.model:ro,$STORAGE_DIR/.cache/:/root/.cache/"
    SLURM_CONTAINER_WORKDIR=$(dirname $BASH_SOURCE)

    # export arguments for TASK in slurm (this is used by ccc run and passed to srun)
    export SLURM_TASK_ARGS="--container-image=$SLURM_CONTAINER_IMG --container-mounts=$SLURM_CONTAINER_MOUNTS --container-workdir=$SLURM_CONTAINER_WORKDIR"

    # when running within container we need to setup conda env (use absence of 'srun' as indication that this is within container)
    if [ -z "$(which srun)" ]; then
        # manually call conda.sh to setup conda init
        . "/root/miniconda3/etc/profile.d/conda.sh"

        conda activate $USE_CONDA_ENV
        echo "Using conda env '$USE_CONDA_ENV'"

    fi
else
    # for HPC-VEGA and HPC-ARNES with support for modules
    module purge
    module load Anaconda3

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

fi

# set default output logs to SLURM TASK
export SLURM_TASK_ARGS="$SLURM_TASK_ARGS --output=logs/%j-node-%t.out"

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
export VICOS_MUJOCO_DATASET=$STORAGE_DIR/datasets/MuJoCo


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