#!/bin/bash

# include some config and utils
source $(ccc file_cfg)
source $(ccc file_utils)

export USE_CONDA_ENV=unified-io-2
export USE_SRUN=0
export DISABLE_X11=0


# 15 GB + 2-3 GB per batch ??
# GPU 18GB .. batch_size=2 
# GPU 22GB .. batch_size=4
# GPU 27GB .. batch_size=6
# GPU 34GB .. batch_size=8 (25GB on 2x GPU)
# GPU 54GB .. batch_size=16 (interupted at 5. epoch)
# GPU 72GB .. batch_size=24 (done)
# GPU 2x 52GB .. batch_size=32 (interupted at 7. epoch)


#export SLURM_JOB_ARGS="--output=logs/%j-node-%t.out --time=72:00:00 --mem-per-gpu=128G --partition=gpu --cpus-per-task=32 --constraint=h100" # HPC-ARNES
export SLURM_JOB_ARGS="--time=72:00:00 --mem-per-gpu=128G --partition=frida --cpus-per-task=32 --constraint=GPU_MEM:80GB" # HPC-FRIDA # GPU_BRD:H100
GPUS_FILE=$(ccc gpus --on_cluster=$CLUSTER_INFO_FILE --gpus=1 --tasks=1 --hosts="crushinator(1+2+3+4)" --gpus_as_single_host=True)

# with keypoint order randomization and weight decay
#ccc run $GPUS_FILE python -m train ../config/vicos_towel/train.py --config  n_epochs=100 train_dataset.batch_size=16 train_dataset.workers=16 model.lr=1e-3 model.accumulate_grads_iter=8 train_dataset.kwargs.num_cpu_threads=4
#ccc run $GPUS_FILE python -m train ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 model.weight_decay=1e-5 n_epochs=100 save_interval=10 extra_str="_weight_decay=1e-5" &
#ccc run $GPUS_FILE python -m train ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 model.weight_decay=1e-6 n_epochs=100 save_interval=10 extra_str="_weight_decay=1e-6" &

# with MUJOCO and with keypoint order randomization
#ccc run $GPUS_FILE python -m train ../config/vicos_towel+mujoco/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=50 save_interval=5  &

# with MUJOCO and with keypoint order randomization
ccc run $GPUS_FILE python -m train ../config/vicos_towel+mujoco/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=35 save_interval=5 train_dataset.keypoint_preprocesser_kwargs.jitter_keypoints_px=3 extra_str="_keypoint_jitter=3"  &
ccc run $GPUS_FILE python -m train ../config/vicos_towel+mujoco/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=35 save_interval=5 train_dataset.keypoint_preprocesser_kwargs.jitter_keypoints_px=5 extra_str="_keypoint_jitter=5"  &
ccc run $GPUS_FILE python -m train ../config/vicos_towel+mujoco/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=35 save_interval=5 train_dataset.keypoint_preprocesser_kwargs.jitter_keypoints_px=7 extra_str="_keypoint_jitter=7"  &


wait_or_interrupt
rm $GPUS_FILE

