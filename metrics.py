import os
import json
import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


psnr = PeakSignalNoiseRatio(data_range=1.0)
ssim = SSIM(data_range=1.0, size_average=True, channel=3)
lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

transform = transforms.Compose([transforms.ToTensor()])


def _to_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, np.generic):
        return float(x.item())
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return x


def _safe(results_dict):
    if isinstance(results_dict, dict):
        return {k: _safe(v) for k, v in results_dict.items()}
    if isinstance(results_dict, (list, tuple)):
        return [_safe(v) for v in results_dict]
    if torch.is_tensor(results_dict):
        return _to_float(results_dict)
    if isinstance(results_dict, np.generic):
        return _to_float(results_dict)
    return results_dict


def evaluate(result_dir, iteration=None):
    """Evaluate PSNR/SSIM/LPIPS on the test split rendered by render.py."""
    if iteration is None:
        # Auto-detect available iteration under `test/ours_*`
        test_root = os.path.join(result_dir, "test")
        if not os.path.isdir(test_root):
            raise FileNotFoundError(f"Test directory not found: {test_root}. "
                                    f"Run render.py before metrics.py.")
        iters = [d for d in os.listdir(test_root) if d.startswith("ours_")]
        if not iters:
            raise FileNotFoundError(f"No 'ours_*' directory in {test_root}.")
        iteration = sorted(iters, key=lambda x: int(x.split("_")[-1]))[-1].split("_")[-1]

    eval_path = os.path.join(result_dir, "test", f"ours_{iteration}", "renders")
    gt_path = os.path.join(result_dir, "test", f"ours_{iteration}", "gt")

    eval_list = sorted(os.listdir(eval_path))
    datanum = len(eval_list)

    _psnr, _ssim, _lpips = 0.0, 0.0, 0.0
    for name in tqdm(eval_list, desc="Evaluating"):
        img = Image.open(os.path.join(eval_path, name)).convert("RGB")
        gt = Image.open(os.path.join(gt_path, name)).convert("RGB")
        img_tensor = transform(img)[None, ...]
        gt_tensor = transform(gt)[None, ...]

        _psnr += psnr(gt_tensor, img_tensor)
        _ssim += ssim(gt_tensor, img_tensor)
        _lpips += lpips(gt_tensor, img_tensor)

    _psnr /= datanum
    _ssim /= datanum
    _lpips /= datanum
    print(f" psnr: {_psnr:.2f}, ssim: {_ssim:.4f}, lpips: {_lpips:.4f}, "
          f"datanum: {datanum}")

    results = _safe({
        "paths": {"renders": eval_path, "gt": gt_path},
        "counts": {"evaluated": int(datanum)},
        "iteration": int(iteration),
        "metrics": {
            "psnr": _to_float(_psnr),
            "ssim": _to_float(_ssim),
            "lpips": _to_float(_lpips),
        },
    })

    save_path = os.path.join(result_dir, "results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Saved results to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PriSplat evaluation")
    parser.add_argument("--result_dir", type=str, required=True,
                        help="model output directory produced by train.py / render.py")
    parser.add_argument("--iteration", type=str, default=None,
                        help="iteration to evaluate (e.g., 30000). "
                             "If omitted, uses the highest available.")
    args = parser.parse_args()

    evaluate(args.result_dir, args.iteration)
