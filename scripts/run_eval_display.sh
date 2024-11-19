#!/bin/bash

# include some config and utils
source $(ccc file_cfg)
source $(ccc file_utils)

export USE_CONDA_ENV=unified-io-2
export DISABLE_X11=1


export SLURM_JOB_ARGS="--time=5:00:00 --mem-per-gpu=128G --partition=frida --cpus-per-task=32 --exclude=ixh" # HPC-FRIDA # do not use H100 for inference
#export SLURM_JOB_ARGS="--time=5:00:00 --mem-per-gpu=128G --partition=frida --cpus-per-task=32" # HPC-FRIDA # GPU_BRD:H100, GPU_MEM:80GB
#export SLURM_JOB_ARGS="--time=1:00:00 --mem-per-gpu=32G --partition=dev --cpus-per-task=32" # HPC-FRIDA # GPU_BRD:H100, GPU_MEM:80GB
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

DISPLAY_CFG="display_to_file=True"

for epoch in ""; do # 
    for eval_cropped in False True; do
        # with MUJOCO, with keypoint order randomization and keypoint location jitter
        #ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel+mujoco/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=35 save_interval=5 train_dataset.keypoint_preprocesser_kwargs.jitter_keypoints_px=3 extra_str="_keypoint_jitter=3" eval_type=test eval_epoch=$epoch eval_cropped=$eval_cropped skip_if_exists=True  &
        #ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel+mujoco/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=35 save_interval=5 train_dataset.keypoint_preprocesser_kwargs.jitter_keypoints_px=3 extra_str="_keypoint_jitter=3" eval_type=train eval_epoch=$epoch eval_cropped=$eval_cropped skip_if_exists=True &

        #ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel+mujoco/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=35 save_interval=5 train_dataset.keypoint_preprocesser_kwargs.jitter_keypoints_px=5 extra_str="_keypoint_jitter=5" eval_type=test eval_epoch=$epoch eval_cropped=$eval_cropped skip_if_exists=True  &
        #ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel+mujoco/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=35 save_interval=5 train_dataset.keypoint_preprocesser_kwargs.jitter_keypoints_px=5 extra_str="_keypoint_jitter=5" eval_type=train eval_epoch=$epoch eval_cropped=$eval_cropped skip_if_exists=True &

        #ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel+mujoco/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=35 save_interval=5 train_dataset.keypoint_preprocesser_kwargs.jitter_keypoints_px=7 extra_str="_keypoint_jitter=7" eval_type=test eval_epoch=$epoch eval_cropped=$eval_cropped skip_if_exists=True  &
        #ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel+mujoco/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=35 save_interval=5 train_dataset.keypoint_preprocesser_kwargs.jitter_keypoints_px=7 extra_str="_keypoint_jitter=7" eval_type=train eval_epoch=$epoch eval_cropped=$eval_cropped skip_if_exists=True &

        # with internal scale augmentation during training and keypoint order randomization (with  MuJoCo)
        ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel+mujoco/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=35 save_interval=5 train_dataset.keypoint_preprocesser_kwargs.apply_internal_scale_aug=True extra_str="_internal_scale_aug"  eval_type=test eval_epoch=$epoch eval_cropped=$eval_cropped skip_if_exists=False display_to_file=True &
        #ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel+mujoco/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=35 save_interval=5 train_dataset.keypoint_preprocesser_kwargs.apply_internal_scale_aug=True extra_str="_internal_scale_aug"  eval_type=train eval_epoch=$epoch eval_cropped=$eval_cropped skip_if_exists=True  &

    done

done

for epoch in  "" ; do # _010 _020 _030 _040 _050 _060 _070 _080 _090
    for eval_cropped in False True; do
        # with internal scale augmentation during training and keypoint order randomization (without  MuJoCo)
        ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=100 save_interval=5 train_dataset.keypoint_preprocesser_kwargs.apply_internal_scale_aug=True extra_str="_internal_scale_aug"  eval_type=test eval_epoch=$epoch eval_cropped=$eval_cropped skip_if_exists=False display_to_file=True  &
        #ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=100 save_interval=5 train_dataset.keypoint_preprocesser_kwargs.apply_internal_scale_aug=True extra_str="_internal_scale_aug"  eval_type=train eval_epoch=$epoch eval_cropped=$eval_cropped skip_if_exists=True  &
    done
done

wait_or_interrupt
rm $GPUS_FILE