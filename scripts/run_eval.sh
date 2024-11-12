#!/bin/bash

# include some config and utils
source $(ccc file_cfg)
source $(ccc file_utils)

export USE_CONDA_ENV=unified-io-2
export DISABLE_X11=1

export SLURM_JOB_ARGS="--time=5:00:00 --mem-per-gpu=192G --partition=frida --cpus-per-task=32" # HPC-FRIDA # GPU_BRD:H100, GPU_MEM:80GB
export SLURM_JOB_ARGS="$SLURM_JOB_ARGS --output=/dev/null --error=/dev/null "
#GPUS_FILE=$(ccc gpus --on_cluster=$CLUSTER_INFO_FILE --gpus=1 --tasks=1 --hosts="crushinator(1+2+3+4)" --gpus_as_single_host=True)
GPUS_FILE=$(ccc gpus --on_cluster=$CLUSTER_INFO_FILE --gpus=1 --tasks=5 --hosts="crushinator,kif,zapp" --gpus_as_single_host=True)

#ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=100 save_interval=25 eval_type=test eval_epoch="_025" &
#ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=100 save_interval=25 eval_type=train eval_epoch="_025" &

#ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=300 save_interval=25 eval_type=test eval_epoch="" &
#ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=300 save_interval=25 eval_type=test eval_epoch=_200 &
#ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=300 save_interval=25 eval_type=test eval_epoch=_100 &
#ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=300 save_interval=25 eval_type=test eval_epoch=_050 &

#ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=300 save_interval=25 eval_type=train eval_epoch="" &

for epoch in _005 _010 _015 _020 _025 _030 _035 _040 _045 ""; do #
    #ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel+mujoco/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=50 save_interval=5 eval_type=test eval_epoch=$epoch skip_if_exists=True &
    ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel+mujoco/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=50 save_interval=5 eval_type=train eval_epoch=$epoch skip_if_exists=True &
done

wait_or_interrupt
rm $GPUS_FILE
