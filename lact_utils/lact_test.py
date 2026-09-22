import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import imageio
import omegaconf
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace

from lact_utils.data_inference import MaskNVSDataset
from lact_utils.pipe.model import LaCTLVSM


def model_size_report(model):
    # 1) params / buffers
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buffers = sum(b.numel() for b in model.buffers())

    # 2) parameter memory
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_bytes + buffer_bytes

    def fmt(n):
        for unit in ["B","KB","MB","GB","TB"]:
            if n < 1024:
                return f"{n:.2f}{unit}"
            n /= 1024
        return f"{n:.2f}PB"

    print(f"#params: {n_params:,} (trainable {n_trainable:,})")
    print(f"#buffers: {n_buffers:,}")
    print(f"param bytes:  {fmt(param_bytes)}")
    print(f"buffer bytes: {fmt(buffer_bytes)}")
    print(f"TOTAL (weights+buffers): {fmt(total_bytes)}")

def img_to_dict(img_path):
    mask_dict = dict()
    for imgname in os.listdir(img_path):
        imgpath = f"{img_path}/{imgname}"
        image = imageio.imread(imgpath)
        alpha = image[...,-1] / 255.
        alpha = torch.tensor(alpha).cuda()
        savename = imgname.replace('png','JPG')
        mask_dict[savename] = alpha
    H,W = alpha.shape
    return mask_dict,H,W

def main(args):
    # LaCT eval
    # Model configuration
    model_config = omegaconf.OmegaConf.load(args.lact_config)

    # Seed everything
    seed = 95
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # LaCT model
    lact_model = LaCTLVSM(**model_config).cuda()
    lact_checkpoint = torch.load(args.lact_ckpt, map_location="cpu")
    lact_model.load_state_dict(lact_checkpoint["model"])

    model_size_report(lact_model)

    inpaint_mask_dict,H,W = img_to_dict(args.maskdir)

    # Dataset
    lact_dataset = MaskNVSDataset(
    args.source_path,
    args.num_views,
    mask_dict=inpaint_mask_dict,
    mask_dilate=args.mask_dilate,
    select_strategy=args.select_strategy,
    seed=seed)

    # Dataloader
    dataloader_seed_generator = torch.Generator()
    dataloader_seed_generator.manual_seed(seed)
    lact_dataloader = DataLoader(
    lact_dataset,
    batch_size=1,
    shuffle=False,
    generator=dataloader_seed_generator,    # This ensures deterministic dataloader
    )

    # Eval
    inpainted_dict = dict()
    for data_dict in lact_dataloader:
        name_list = data_dict["name"]
        data_dict = {key: value.cuda() for key, value in data_dict.items() if isinstance(value, torch.Tensor)}
        import time
        for idx in range(data_dict["image"].shape[1] // args.num_views):
            istart = time.time()
            input_indices = torch.arange(idx * args.num_views, (idx + 1) * args.num_views)
            target_data_dict = {k: v[:, input_indices[0]:input_indices[0]+1] for k, v in data_dict.items()}
            input_masked_data_dict = {key: value[:, input_indices] for key, value in data_dict.items()}

            with torch.autocast(dtype=torch.bfloat16, device_type="cuda", enabled=True) and torch.no_grad():
                rendering, _ = lact_model(input_masked_data_dict, target_data_dict, args.lact_original)
            out = F.interpolate(rendering.squeeze(0), size=(H,W), mode='bicubic', align_corners=False).squeeze(0)
            np_out = out.permute(1,2,0).detach().cpu().numpy().clip(0,1)
            save_path = f"{args.outdir}/lact_results"
            os.makedirs(save_path, exist_ok=True)
            plt.imsave(f"{save_path}/{name_list[input_indices[0].item()][0]}", np_out)    
            ptime = time.time()
            print(ptime-istart)                
            

if __name__ == "__main__":

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--source_path", type=str, default="wg_datasets/mountain")
    parser.add_argument("--outdir", type=str, default="lact_utils/test")
    parser.add_argument("--maskdir", type=str)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--mask_dilate", type=int, default=None)
    parser.add_argument("--select_strategy", type=str, default="knn")
    parser.add_argument("--lact_config", type=str, default="lact_utils/config/lact_l24_d768_ttt2x.yaml")
    parser.add_argument("--lact_ckpt", type=str, default="lact_utils/ckpts/model_0003000.pth")
    parser.add_argument("--lact_original", action='store_true')


    args = parser.parse_args(sys.argv[1:])
    main(args)


# Example:
# export PYTHONPATH=$(pwd):$PYTHONPATH
# python lact_utils/lact_test.py \
#     --source_path data/mountain \
#     --maskdir output/mountain/mask_10000 \
#     --outdir output/mountain/lact_results \
#     --mask_dilate 15 \
#     --select_strategy select_knn
