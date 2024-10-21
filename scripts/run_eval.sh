#!/bin/bash

# include some config and utils
source $(ccc file_cfg)
source $(ccc file_utils)

export USE_CONDA_ENV=unified-io-2
export DISABLE_X11=0

#GPUS_FILE=$(ccc gpus --on_cluster=$CLUSTER_INFO_FILE --gpus=1 --tasks=1 --hosts="crushinator(1+2+3+4)" --gpus_as_single_host=True)
GPUS_FILE=$(ccc gpus --on_cluster=$CLUSTER_INFO_FILE --gpus=1 --tasks=3 --hosts="crushinator,kif,zapp" --gpus_as_single_host=True)

#ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=300 save_interval=25 eval_type=test eval_epoch="" &
#ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=300 save_interval=25 eval_type=test eval_epoch=_200 &
#ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=300 save_interval=25 eval_type=test eval_epoch=_100 &
#ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=300 save_interval=25 eval_type=test eval_epoch=_050 &

ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=300 save_interval=25 eval_type=test eval_epoch=_025 &
ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=300 save_interval=25 eval_type=test eval_epoch=_075 &
ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=300 save_interval=25 eval_type=test eval_epoch=_150 &

#ccc run $GPUS_FILE python -m test_vicos_towel ../config/vicos_towel/train.py --config train_dataset.batch_size=24 train_dataset.workers=16 model.lr=1e-3 n_epochs=300 save_interval=25 eval_type=train eval_epoch="" &

wait_or_interrupt
rm $GPUS_FILE
