#!/usr/bin/env bash
# Example training script for PriSplat on NeRF On-the-go dataset.
#
# Usage:
#   bash scripts/train.sh <scene>
#
# Expected data layout under ./data/<scene>:
#   images/                  (RGB training images)
#   sparse/                  (COLMAP reconstruction: cameras.bin, images.bin, points3D.bin)
#   opencv_cameras.json      (produced by lact_utils/colmap_to_opencv_json.py)
#   train_list.txt           (image names for training)
#   test_list.txt            (image names for evaluation)

set -e

SCENE=${1:-mountain}
DATA_ROOT=${DATA_ROOT:-./data}
OUT_ROOT=${OUT_ROOT:-./output}
PORT=${PORT:-1111}

DATA_DIR="${DATA_ROOT}/${SCENE}"
MODEL_DIR="${OUT_ROOT}/${SCENE}"
mkdir -p "${MODEL_DIR}"

# --- Hyperparameters ---
FISHER_THR=0.4         # Relative threshold
FINAL_SCORE=0.3        # Absolute threshold
INPAINT_START=10000    # iteration at which TTT-based inpainting kicks in
INPAINT_END=20000      # iteration after which the inpainting loss is disabled
MASK_DILATE=15         # mask dilation for LaCT inputs
EXP_START=3000         # iteration to start per-image exposure

python train.py \
    --port "${PORT}" \
    -s "${DATA_DIR}" \
    -r 1 \
    --model_path "${MODEL_DIR}" \
    --use_inpaint \
    --select_strategy fisherrf \
    --fisherrf_thr "${FISHER_THR}" \
    --final_score "${FINAL_SCORE}" \
    --use_inpaint_iter "${INPAINT_START}" \
    --use_inpaint_iter_end "${INPAINT_END}" \
    --mask_dilate "${MASK_DILATE}" \
    --exp_iter_start "${EXP_START}" \
    --densify_from_iter 10000 \
    --densify_until_iter 20000 \
    --exposure_lr_init 0.005 \
    --lact_config lact_utils/config/lact_l24_d768_ttt2x.yaml \
    --lact_ckpt lact_utils/ckpts/model_0003000.pth

# Render test views
python render.py -m "${MODEL_DIR}"

# Compute PSNR/SSIM/LPIPS
python metrics.py --result_dir "${MODEL_DIR}"
