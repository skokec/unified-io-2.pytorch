#!/bin/bash

# include some config and utils
source $(ccc file_cfg)
source $(ccc file_utils)

export USE_CONDA_ENV=unified-io-2
export DISABLE_X11=0


# 15 GB + 2-3 GB per batch ??
# GPU 18GB .. batch_size=2 
# GPU 22GB .. batch_size=4
# GPU 27GB .. batch_size=6
# GPU 34GB .. batch_size=8 (25GB on 2x GPU)
# GPU 54GB .. batch_size=16 (interupted at 5. epoch)
# GPU 72GB .. batch_size=24 (done)
# GPU 2x 52GB .. batch_size=32 (interupted at 7. epoch)


export SLURM_JOB_ARGS="--output=logs/%j-node-%t.out --time=72:00:00 --mem-per-gpu=128G --partition=gpu --cpus-per-task=32 --constraint=h100"
GPUS_FILE=$(ccc gpus --on_cluster=$CLUSTER_INFO_FILE --gpus=1 --tasks=1 --hosts="flexo(1)" --gpus_as_single_host=True)

#ccc run $GPUS_FILE python -m train ../config/vicos_towel/train.py --config  n_epochs=100 train_dataset.batch_size=2 train_dataset.workers=16 model.lr=1e-3 model.accumulate_grads_iter=1 train_dataset.kwargs.num_cpu_threads=4
#ccc run $GPUS_FILE python -m train ../config/vicos_towel/train.py --config train_dataset.batch_size=24 model.lr=1e-3 n_epochs=300 save_interval=25
echo "flexo:0" > $GPUS_FILE

ccc run $GPUS_FILE python -m train ../config/vicos_towel+mujoco/train.py --config  n_epochs=20 save_interval=2 train_dataset.batch_size=8 train_dataset.workers=16 model.lr=1e-3

wait_or_interrupt
rm $GPUS_FILE
