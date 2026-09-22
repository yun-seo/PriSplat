import argparse
import random
import os

import imageio
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup
from PIL import Image

from data_inference import NVSDataset
from model import LaCTLVSM
from mask_tools import GeneratorParams, generate_mask

def get_turntable_cameras_with_zoom_in(
    batch_size=1,
    hfov=50,
    num_views=8,
    w=256,
    h=256,
    min_radius=1.7,
    max_radius=3.0,
    elevation=30,
    up_vector=np.array([0, 0, 1]),
    device=torch.device("cuda"),
):
    '''
    rotate the camera around the object, and change the radius and elevation periodically
    '''
    fx = w / (2 * np.tan(np.deg2rad(hfov) / 2.0))
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    fxfycxcy = np.array([fx, fy, cx, cy]).reshape(1, 4).repeat(num_views, axis=0) # [num_views, 4]
    azimuths = np.linspace(0, 360, num_views, endpoint=False)
    elevations = np.ones_like(azimuths) * (elevation + 15 * np.sin(np.linspace(0, 2*np.pi, num_views)))
    radius = (min_radius + max_radius) / 2.0 + (max_radius - min_radius) / 2.0 * np.sin(np.linspace(0, 2*np.pi, num_views))
    c2ws = []

    for cur_radius, elev, azim in zip(radius, elevations, azimuths):
        elev, azim = np.deg2rad(elev), np.deg2rad(azim)
        z = cur_radius * np.sin(elev)
        base = cur_radius * np.cos(elev)
        x = base * np.cos(azim)
        y = base * np.sin(azim)
        cam_pos = np.array([x, y, z])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, up_vector)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        R = np.stack((right, -up, forward), axis=1)
        c2w = np.eye(4)
        c2w[:3, :4] = np.concatenate((R, cam_pos[:, None]), axis=1)
        c2ws.append(c2w)
    c2ws = np.stack(c2ws, axis=0)  # [num_views, 4, 4]

    # Expand from [num_views, *] to [batch_size, num_views, *]
    fxfycxcy = fxfycxcy[None, ...].repeat(batch_size, axis=0) # [batch_size, num_views, 4]
    c2ws = c2ws[None, ...].repeat(batch_size, axis=0)
    return {
        "w": w,
        "h": h,
        "num_views": num_views,
        "fxfycxcy": torch.from_numpy(fxfycxcy).to(device).float(),
        "c2w": torch.from_numpy(c2ws).to(device).float(),
    }


def get_interpolated_cameras(
    cameras,
    num_views,
):
    """
    For each consecutive pair of cameras, add num_views linearly interpolated views.
    """
    fxfycxcy = cameras['fxfycxcy']  # [batch_size, num_input_views, 4]
    c2w = cameras['c2w']  # [batch_size, num_input_views, 4, 4]
    
    batch_size, num_input_views = fxfycxcy.shape[:2]
    
    interpolated_fxfycxcy = []
    interpolated_c2w = []
    
    for b in range(batch_size):
        batch_fxfycxcy = []
        batch_c2w = []
        
        for i in range(num_input_views - 1):
            # Add the current view
            batch_fxfycxcy.append(fxfycxcy[b, i])
            batch_c2w.append(c2w[b, i])
            
            curr_fxfycxcy = fxfycxcy[b, i]
            next_fxfycxcy = fxfycxcy[b, i + 1]
            curr_c2w = c2w[b, i]
            next_c2w = c2w[b, i + 1]
            
            # Create alpha values for all interpolations at once
            alphas = torch.linspace(1 / (num_views + 1), num_views / (num_views + 1), num_views, device=fxfycxcy.device)
            
            # Batch interpolation for camera intrinsics
            interp_fxfycxcy = (1 - alphas[:, None]) * curr_fxfycxcy[None, :] + alphas[:, None] * next_fxfycxcy[None, :]
            batch_fxfycxcy.extend(interp_fxfycxcy)
            
            # Batch interpolation for camera poses
            # For rotation, we should use SLERP, but for simplicity using linear interpolation
            interp_c2w = (1 - alphas[:, None, None]) * curr_c2w[None, :, :] + alphas[:, None, None] * next_c2w[None, :, :]
            batch_c2w.extend(interp_c2w)
        
        # Add the last view
        batch_fxfycxcy.append(fxfycxcy[b, -1])
        batch_c2w.append(c2w[b, -1])
        
        interpolated_fxfycxcy.append(torch.stack(batch_fxfycxcy))
        interpolated_c2w.append(torch.stack(batch_c2w))
    
    return {
        'fxfycxcy': torch.stack(interpolated_fxfycxcy),
        'c2w': torch.stack(interpolated_c2w)
    }

import torch._dynamo
torch._dynamo.config.suppress_errors = True

parser = argparse.ArgumentParser()
# Basic info
parser.add_argument("--config", type=str, default="config/lact_l24_d768_ttt2x.yaml")
parser.add_argument("--load", type=str, default="weight/obj_res256.pt")
parser.add_argument("--data_path", type=str, default="data_example/gso_sample_data_path.json")
parser.add_argument("--output_dir", type=str, default="output/")
parser.add_argument("--num_all_views", type=int, default=32)

parser.add_argument("--num_input_views", type=int, default=20)
parser.add_argument("--num_target_views", type=int, default=None)
parser.add_argument("--scene_inference", action="store_true")
parser.add_argument("--image_size", nargs=2, type=int, default=[256, 256], help="Image size H, W")
parser.add_argument("--random_mask", action="store_true")
parser.add_argument("--mask_dilate", action="store_true")


args = parser.parse_args()
if args.num_target_views is None:
    args.num_target_views = args.num_all_views - args.num_input_views
model_config = omegaconf.OmegaConf.load(args.config)
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
# os.makedirs(os.path.join(output_dir, "rendered"), exist_ok=True)
# os.makedirs(os.path.join(output_dir, "input_masked"), exist_ok=True)

# Seed everything
seed = 95
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
model = LaCTLVSM(**model_config).cuda()

# Load checkpoint
print(f"Loading checkpoint from {args.load}...")
checkpoint = torch.load(args.load, map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()

# Data
dataset = NVSDataset(args.data_path, args.num_all_views, tuple(args.image_size),
                    sorted_indices=args.scene_inference, scene_pose_normalize=args.scene_inference,
                    mask_dilate=args.mask_dilate)
dataloader_seed_generator = torch.Generator()
dataloader_seed_generator.manual_seed(seed)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    generator=dataloader_seed_generator,    # This ensures deterministic dataloader
)
now_iters = 0
for sample_idx, data_dict in enumerate(dataloader):

    name_list = data_dict["name"]
    size_original_list = data_dict["size_original"]
    data_dict = {key: value.cuda() for key, value in data_dict.items() if isinstance(value, torch.Tensor)}
    os.makedirs(os.path.join(output_dir, f"{sample_idx}", "rendered"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f"{sample_idx}", "original"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f"{sample_idx}", "input_masked"), exist_ok=True)
    
    for idx in range(data_dict["image"].shape[1] // args.num_all_views):
        input_indices = torch.arange(idx * args.num_all_views, (idx + 1) * args.num_all_views)
        input_name_list = name_list[idx * args.num_all_views:(idx + 1) * args.num_all_views]
        input_size_original_list = size_original_list[idx * args.num_all_views:(idx + 1) * args.num_all_views]
        input_data_dict = {key: value[:, input_indices] for key, value in data_dict.items()}
        target_data_dict = {k: v[:, input_indices[0]:input_indices[0]+1] for k, v in data_dict.items()}
        input_masked_data_dict = {key: value[:, input_indices] for key, value in data_dict.items()}
            
        B, V, C, H, W = input_masked_data_dict["image"].shape
        mask_list = []
        rng = np.random.RandomState(now_iters)  # deterministic per iteration
        now_iters += 1

        gen_params = GeneratorParams(
            height=H,
            width=W,
            num_components_hist={"1": 1, "2": 1, "3": 1},
            component_area_quantiles={"q05": 0.002, "q25": 0.004, "q50": 0.008, "q75": 0.015, "q95": 0.03},
            bbox_aspect_quantiles={"q05": 0.5, "q25": 0.8, "q50": 1.2, "q75": 1.8, "q95": 2.5},
            hole_probability=0.9,      # --hole-prob 0.5
            smooth_sigma=0.6,
            area_scale=3.0,           # --area-scale 0.15
        )

        mask_generated = []
        for b in range(B):
            mask_views = []
            for v in range(V):
                py_rng = random.Random(int(now_iters * 1000 + b * 100 + v))
                mask_np = generate_mask(gen_params, py_rng)  # (H, W), uint8
                mask_torch = torch.from_numpy(mask_np).float().to(input_masked_data_dict["image"].device)
                mask_views.append(mask_torch)
            mask_views = torch.stack(mask_views, dim=0)
            mask_generated.append(mask_views)
        mask_generated = torch.stack(mask_generated, dim=0)
        mask_generated = mask_generated.unsqueeze(2)
        if args.random_mask:
            input_masked_data_dict["image"] = input_masked_data_dict["image"] * (1 - mask_generated)
        psnr_log_path = os.path.join(output_dir, f"{sample_idx}", "psnr_log.txt") if output_dir else None
        with torch.no_grad():
            rendering, _, _, _ = model(input_masked_data_dict, input_data_dict)
            target = input_data_dict["image"]
            input_masked = input_masked_data_dict["image"]
            eps = 1e-8
            mse_full = F.mse_loss(rendering, target)
            psnr = -10.0 * torch.log10(mse_full + eps).item()
            mask_hole = (mask_generated > 0.5)        # (B, V, 1, H, W)
            mask_hole = mask_hole.expand_as(target)
            mask_valid = ~mask_hole

            if mask_valid.any():
                mse_no_mask = F.mse_loss(rendering[mask_valid], target[mask_valid])
                psnr_no_mask = -10.0 * torch.log10(mse_no_mask + eps).item()
            else:
                psnr_no_mask = float("nan")

            if mask_hole.any():
                mse_mask = F.mse_loss(rendering[mask_hole], target[mask_hole])
                psnr_mask = -10.0 * torch.log10(mse_mask + eps).item()
            else:
                psnr_mask = float("nan")

            if psnr_log_path is not None:
                os.makedirs(os.path.dirname(psnr_log_path), exist_ok=True)
                with open(psnr_log_path, "a") as f:
                    f.write(f"{sample_idx}\t{psnr:.6f}\t{psnr_no_mask:.6f}\t{psnr_mask:.6f}\n")

            print(f"Sample {sample_idx}: PSNR = {psnr:.2f}, PSNR_no_mask = {psnr_no_mask:.2f}, PSNR_mask = {psnr_mask:.2f}")
            
            # Save rendered images if output directory is specified
            if output_dir:
                
                def save_image_rgb(tensor, filepath, size_original):
                    """Save tensor as RGB image."""
                    numpy_image = tensor.permute(1, 2, 0).cpu().numpy()
                    numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)
                    # Ensure size_original is a tuple of ints, not a torch tensor
                    # size_original = size_original[::-1]
                    if isinstance(size_original, torch.Tensor):
                        if size_original.numel() == 2:
                            size_original = tuple(int(x) for x in size_original.tolist())
                        else:
                            size_original = (int(size_original.item()), int(size_original.item()))
                    Image.fromarray(numpy_image, mode='RGB').resize(size_original).save(filepath)

                batch_size, num_views = rendering.shape[:2]
                
                for batch_idx in range(batch_size):
                    for view_idx in range(num_views):
                        # Save rendered and target images
                        for img_type, img_tensor in [("rendered", rendering[batch_idx, view_idx]), 
                                                        ("input_masked", input_masked[batch_idx, view_idx])]:
                            
                            size_original = input_size_original_list[view_idx]
                            if view_idx == 0:
                                input_name = input_name_list[view_idx][batch_idx].split('.')[0]
                                filename = f"{img_type}/{input_name}.png"
                                tar_tensor = target[0][0]
                                save_image_rgb(tar_tensor, os.path.join(output_dir, f"{sample_idx}", filename.replace('rendered','original')), size_original)
                                save_image_rgb(img_tensor, os.path.join(output_dir, f"{sample_idx}", filename), size_original)
                
                # print(f"Saved images for sample {sample_idx} to {output_dir}")
        
            