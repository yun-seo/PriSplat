# LaCT NVS training

Self-contained training code for the **LaCT (Large-Chunk Test-time-training)**
novel-view-synthesis backbone that PriSplat uses at splat-training time.

This is a lightly adapted copy of the training pipeline released with the
[LaCT paper](https://tianyuanzhang.com/projects/ttt-done-right/) — kept here
for reproducibility. If you only want to *use* a pretrained LaCT model,
skip this folder and place the checkpoint at `lact_utils/ckpts/` directly
(see the top-level `README.md`).

## Layout

```
lact_utils/train/
├── train_masked.py             # main training script (masked NVS)
├── train_masked_uncertainty.py # uncertainty variant
├── inference.py                # object/scene NVS inference (paper-style)
├── inference_masked.py         # masked NVS inference (paper-style)
├── data.py                     # training dataset (NVSDataset)
├── data_inference.py           # validation dataset (VAL NVSDataset)
├── mask_tools.py               # random training-mask generator
├── model.py                    # LaCTLVSM (training variant)
├── lact_ttt.py                 # fast-weight TTT operator (training variant)
├── download.py                 # DL3DV download helper
├── config/lact_l24_d768_ttt2x.yaml   # model config (24 layers, dim 768, patch 8)
├── data_preprocess/            # DL3DV download / format conversion, COLMAP -> json
├── weight/                     # init checkpoint(s), see below
├── train.sh                    # launch script
└── requirements.txt            # training-only extras (lpips, transformers, ...)
```

`model.py` / `lact_ttt.py` here are the *training* variants of
`lact_utils/pipe/model.py` / `lact_utils/pipe/lact_ttt.py`. Both define the
same `LaCTLVSM` architecture and parameter names, so checkpoints are
interchangeable: the config in `config/` refers to the local modules
(`model.*`, `lact_ttt.*`) while `lact_utils/config/` refers to
`lact_utils.pipe.*`. The training `forward` returns `(rendering, ret)`; the
pipe `forward` returns only `rendering`.

All scripts in this folder use local imports and relative data paths, so run
them **from inside `lact_utils/train/`**.

## Environment

Use the top-level `prisplat` environment (see the top-level `README.md`). Its
install step already includes `requirements.txt` from this folder. If you set
the environment up separately:

```bash
pip install -r requirements.txt      # lpips, transformers, huggingface-hub, ...
sudo apt install ffmpeg              # only if you want to save mp4 previews
```

## Data preparation

### Training set

The default training set is [DL3DV-10K](https://dl3dv-10k.github.io/DL3DV-10K/).

```bash
cd lact_utils/train

# 1) download images + poses
python download.py --odir DL3DV-10K --subset 2K --resolution 960P --file_type images+poses

mkdir -p data_train/dl3dv_benchmark
mv DL3DV-10K/2K/* data_train/dl3dv_benchmark/

# 2) convert to the LaCT unified format
python data_preprocess/dl3dv_train_format_converter.py
```

This produces `data_train/dl3dv_processed/` and an index file
`data_train/dl3dv_sample_data_path.json` that the training loop expects.

The index is a JSON list of per-scene camera files, relative to the index
file; each camera file is a list of frames in the OpenCV convention
(`fx, fy, cx, cy, w, h, w2c, file_path`), the same format as PriSplat's
`opencv_cameras.json`. Any scene folder in that format can be added to the
list. Every training scene must contain **more than `--num_all_views`
frames** (default 120): `data.py` groups a scene's cameras into consecutive
chunks of `num_all_views` and drops the last (incomplete) chunk, so a scene
with fewer frames has no valid group.

### Validation set (required)

`train_masked.py` runs a fixed validation every `--validation_every` steps
and reads the path `data_example/robustnerf_sample_data_path.json` (relative
to the working directory). Create it in the same index format, pointing at a
scene with `opencv_cameras.json` and `images/`, e.g.

```
data_example/
├── robustnerf_sample_data_path.json   # ["robustnerf/patio-high/opencv_cameras.json"]
└── robustnerf/patio-high/
    ├── opencv_cameras.json
    └── images/
```

Validation takes the first three groups of 8 views, renders each group from
itself and prints `Average PSNR` (all pixels) and `Average PSNR_no_mask`
(non-black pixels only). Training fails at the first validation if this
file is missing.

## Pretrained init weights

Download the scene-level checkpoint(s) into `./weight/`:

```bash
mkdir -p weight
wget https://huggingface.co/airsplay/lact_nvs/resolve/main/scene_res512x512.pt \
     -O weight/scene_res512x512.pt
```

Other resolutions are listed in the LaCT model card:
<https://huggingface.co/airsplay/lact_nvs>. The released file only contains
model weights (no optimizer state); `--load` accepts it and starts from
iteration 0.

## Training

Simple launcher (4 GPUs, batch 4 per GPU, 80 000 steps by default):

```bash
cd lact_utils/train
bash train.sh
# overridable: NGPU, BS, LR, CONFIG, LOAD, DATA, EXPNAME
NGPU=1 bash train.sh
```

Or invoke `torchrun` directly:

```bash
torchrun --nproc_per_node=4 --standalone \
    train_masked.py \
    --config config/lact_l24_d768_ttt2x.yaml \
    --actckpt \
    --load ./weight/scene_res512x512.pt \
    --data_path data_train/dl3dv_sample_data_path.json \
    --bs_per_gpu 4 --lr 1e-5 \
    --lpips_weight 0.1 --validation_every 10 \
    --scene_pose_normalize --image_size 512 512 \
    --num_target_views 1 --num_input_views 4 --num_all_views 120 \
    --expname lact_train_512x512_v4
```

Each step samples `num_input_views` input views and `num_target_views`
target views from a group of `num_all_views` cameras, punches random holes
(`mask_tools.py`) into the inputs and trains the model to reconstruct the
unmasked inputs (MSE + LPIPS). The log line per step reports PSNR over all
pixels, over the known pixels (`PSNR_no_mask`), over the hole pixels
(`PSNR_masked`) and the masked fraction.

Key options:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--steps` | 80000 | total optimizer steps (cosine LR schedule ends here) |
| `--save_every` | 200 | write `outputs/<expname>/model_<step>.pth` |
| `--validation_every` | 100 | run the validation described above |
| `--log_every` | 1 | print / TensorBoard interval |
| `--bs_per_gpu` | 8 | scenes per GPU per step (4 fits one 48 GB A6000 at 512×512 with `--actckpt`) |
| `--lpips_weight`, `--lpips_start` | 1.0, 0 | LPIPS loss weight / first step it is applied |
| `--warmup`, `--weight_decay` | 0, 0.05 | scheduler warmup steps, AdamW weight decay |
| `--actckpt` | off | activation checkpointing (needed for long token sequences) |
| `--compile` | off | `torch.compile` the model (1.4–1.5× faster after a 30 s–2 min warm-up) |
| `--load` | none | init checkpoint (`.pt`/`.pth` with a `"model"` key) |

Outputs go to `outputs/<expname>/` (checkpoints with model, optimizer,
scheduler and step; TensorBoard logs under `tb_logs/`). If that folder
already contains a `model_*.pth`, training resumes from the latest one and
ignores `--load`.

Quick smoke test on one GPU (about 5 min; runs two validations and writes
one checkpoint):

```bash
torchrun --nproc_per_node=1 --standalone train_masked.py \
    --config config/lact_l24_d768_ttt2x.yaml --actckpt \
    --load ./weight/scene_res512x512.pt \
    --data_path data_train/dl3dv_sample_data_path.json \
    --bs_per_gpu 4 --lr 1e-5 --lpips_weight 0.1 --validation_every 10 \
    --scene_pose_normalize --image_size 512 512 \
    --num_target_views 1 --num_input_views 4 --num_all_views 120 \
    --steps 25 --save_every 20 --expname smoke
```

Notes:
- `--actckpt` enables activation checkpointing (needed to fit long input
  token sequences into GPU memory).
- `--compile` gives a 1.4–1.5× speed-up but the first step takes 30 s–2 min
  to compile; drop it for debugging. It needs a C compiler visible to
  Triton (the conda `gcc` from the top-level install works).
- Use `train_masked.py` for the fixed-mask training objective and
  `train_masked_uncertainty.py` for the uncertainty-guided variant.

## Using the trained checkpoint with PriSplat

After training, copy the resulting checkpoint into the top-level PriSplat
checkpoint folder and point `train.py` at it:

```bash
cp outputs/<expname>/model_<step>.pth ../../lact_utils/ckpts/
```

then run PriSplat as usual (`python train.py --lact_ckpt lact_utils/ckpts/model_<step>.pth ...`).
The bundled `lact_utils/ckpts/model_0003000.pth` is a step-3000 checkpoint
in exactly this format (model + optimizer + scheduler + `now_iters`).
