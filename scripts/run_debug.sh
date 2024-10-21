#!/bin/bash

# include some config and utils
source $(ccc file_cfg)
source $(ccc file_utils)

export USE_CONDA_ENV=unified-io-2
export DISABLE_X11=0

export SLURM_JOB_ARGS="--output=logs/%j-node-%t.out --time=01:00:00 --mem-per-gpu=64G --partition=gpu --cpus-per-task=16 --constraint=h100"
GPUS_FILE=$(ccc gpus --on_cluster=$CLUSTER_INFO_FILE --gpus=1 --tasks=1 --hosts="crushinator(3+4+5+6)" --gpus_as_single_host=False)

CMD=("attach_debug=5678"
    "train_dataset.batch_size=2"
    "train_dataset.shuffle=False"
    "train_dataset.workers=0"
    "model.lr=0"
    "n_epochs=11"
    "resume_path=/storage/user/Projects/vision-language/unified-io-2/exp/vicos-towels/model=allenai/uio2-large_resize_factor=1_batchsize=24/num_train_epoch=100/depth=False/checkpoint.pth")

ccc run $GPUS_FILE python -m train ../config/vicos_towel/train.py --config ${CMD[@]} &

wait_or_interrupt
rm $GPUS_FILE
