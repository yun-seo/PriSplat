#!/usr/bin/env bash
# LaCT training example.
#
# 1. Prepare DL3DV training data (see data_preprocess/):
#      python data_preprocess/dl3dv_train_download.py
#      python data_preprocess/dl3dv_train_format_converter.py
#    The final layout is `data_train/dl3dv_processed/` with a
#    `dl3dv_sample_data_path.json` index at `data_train/`.
#
# 2. Download a pretrained scene checkpoint (used as init) into `weight/`:
#      mkdir -p weight
#      wget https://huggingface.co/airsplay/lact_nvs/resolve/main/scene_res512x512.pt \
#           -O weight/scene_res512x512.pt
#
# 3. Run this script (from inside `lact_utils/train/`):
#      cd lact_utils/train
#      bash train.sh

set -e

CONFIG=${CONFIG:-config/lact_l24_d768_ttt2x.yaml}
LOAD=${LOAD:-./weight/scene_res512x512.pt}
DATA=${DATA:-data_train/dl3dv_sample_data_path.json}
NGPU=${NGPU:-4}
BS=${BS:-4}
LR=${LR:-1e-5}
EXPNAME=${EXPNAME:-lact_train_512x512_v4}

torchrun \
    --nproc_per_node="${NGPU}" \
    --standalone \
    train_masked.py \
    --config "${CONFIG}" \
    --actckpt \
    --load "${LOAD}" \
    --data_path "${DATA}" \
    --bs_per_gpu "${BS}" \
    --lr "${LR}" \
    --lpips_weight 0.1 \
    --validation_every 10 \
    --scene_pose_normalize \
    --image_size 512 512 \
    --num_target_views 1 \
    --num_input_views 4 \
    --num_all_views 120 \
    --expname "${EXPNAME}"
