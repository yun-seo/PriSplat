import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
import math
import json
import random
import imageio
import numpy as np
from PIL import Image, ImageFilter

from typing import Dict, List, Tuple, Iterable
import math

def random_indices_for_each_view(images_info, k: int, include_self: bool = True, rng: random.Random | None = None):
    N = len(images_info)
    if N == 0:
        return []

    if rng is None:
        rng = random

    neighbors_per_view = []
    all_idx = list(range(N))

    for i in range(N):
        if include_self:
            others = [j for j in all_idx if j != i]
            m = min(k - 1, len(others))
            sampled = rng.sample(others, m)
            neigh = [i] + sampled
            while len(neigh) < k:
                pick = rng.choice(others) if others else i
                neigh.append(pick)
        else:
            pool = [j for j in all_idx if j != i]
            m = min(k, len(pool))
            neigh = rng.sample(pool, m)
            while len(neigh) < k:
                pick = rng.choice(pool) if pool else i
                neigh.append(pick)

        neighbors_per_view.append(neigh)

    return neighbors_per_view

def resize_and_crop(image, target_size, fxfycxcy, resize_mode='lanczos'):
    """
    Resize and crop image to target_size, adjusting camera parameters accordingly.
    
    Args:
        image: PIL Image
        target_size: (height, width) tuple
        fxfycxcy: [fx, fy, cx, cy] list
    
    Returns:
        tuple: (resized_cropped_image, adjusted_fxfycxcy)
    """
    original_width, original_height = image.size  # PIL image is (width, height)
    target_height, target_width = target_size
    
    fx, fy, cx, cy = fxfycxcy
    
    # Calculate scale factor to fill target size (resize to cover)
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    # scale = max(scale_x, scale_y)  # Use larger scale to ensure it covers the target area
    
    # Resize image
    # new_width = int(round(original_width * scale))
    # new_height = int(round(original_height * scale))
    new_width = target_width
    new_height = target_height
    if resize_mode == 'lanczos':
        resized_image = image.resize((new_width, new_height), Image.LANCZOS) #lanczos
    elif resize_mode == 'nearest':
        resized_image = image.resize((new_width, new_height), Image.Resampling.NEAREST) #nearest neighbor
    else:
        raise ValueError(f"Invalid resize mode: {resize_mode}")
    
    # Crop image
    # cropped_image = resized_image.crop((left, top, right, bottom))
    cropped_image = resized_image
    
    # Adjust camera parameters
    # Scale focal lengths and principal points
    new_fx = fx * scale_x   
    new_fy = fy * scale_y
    new_cx = cx * scale_x
    new_cy = cy * scale_y
    
    return cropped_image, [new_fx, new_fy, new_cx, new_cy]

def normalize(x):
    return x / x.norm()

def normalize_with_mean_pose(c2ws: torch.Tensor):
    # This is a historical code for scene camera normalization;
    #  thanks to the authors (might mostly credit to Zexiang Xu)

    # Get the mean parameters
    center = c2ws[:, :3, 3].mean(0)
    vec2 = c2ws[:, :3, 2].mean(0)
    up = c2ws[:, :3, 1].mean(0)

    # Get the view matrix.
    vec2 = normalize(vec2)
    vec0 = normalize(torch.cross(up, vec2))
    vec1 = normalize(torch.cross(vec2, vec0))
    m = torch.stack([vec0, vec1, vec2, center], 1)

    # Extend the view matrix to 4x4.
    avg_pos = c2ws.new_zeros(4, 4)
    avg_pos[3, 3] = 1.0
    avg_pos[:3] = m

    # Align coordinate system to the mean camera
    c2ws = torch.linalg.inv(avg_pos) @ c2ws

    # Scale the scene to the range of [-1, 1].
    scene_scale = torch.max(torch.abs(c2ws[:, :3, 3]))
    c2ws[:, :3, 3] /= scene_scale

    return c2ws

def knn_indices_for_each_view(images_info, k: int, include_self: bool = True):
    """
    For each view, return indices of its k nearest neighbors under the combined
    (position + direction) distance. Optionally include the view itself.

    Args:
        images_info: List of per-view metadata dicts
        k: Number of neighbors per view to return
        include_self: If True, neighbors may include the query view itself

    Returns:
        List[List[int]] of shape [N][k], where N = len(images_info)
    """
    N = len(images_info)
    if N == 0:
        return []

    # Build c2w for all views without loading images
    c2ws = []
    for info in images_info:
        w2c = torch.tensor(info["w2c"], dtype=torch.float32)
        c2ws.append(torch.inverse(w2c))
    c2ws = torch.stack(c2ws, dim=0)  # [N, 4, 4]

    positions = c2ws[:, :3, 3]
    directions = F.normalize(c2ws[:, :3, 2], dim=1)

    # Pairwise position distance and angular distance (1 - cos sim)
    pos_dist = torch.cdist(positions, positions)  # [N, N]
    cos_sim = torch.clamp(directions @ directions.t(), -1.0, 1.0)
    ang_dist = 1.0 - cos_sim

    # Normalize scales to balance contributions
    pos_mask = pos_dist > 0
    ang_mask = ang_dist > 0
    pos_scale = float(torch.median(pos_dist[pos_mask])) if torch.any(pos_mask) else 1.0
    ang_scale = float(torch.median(ang_dist[ang_mask])) if torch.any(ang_mask) else 1.0
    combined = (pos_dist / pos_scale) + (ang_dist / ang_scale)

    neighbors_per_view = []
    # tmp!
    neighbors_per_view_distance = []
    for i in range(N):
        distances = combined[i]
        order = torch.argsort(distances)
        if not include_self:
            # Filter out self index i
            order = order[order != i]
        # Take first k
        neighbors = order[:k].tolist()
        # If k > available (e.g., N==1), pad with self or repeat nearests
        if len(neighbors) < k:
            fill_with = i if include_self else (neighbors[0] if neighbors else i)
            neighbors = neighbors + [fill_with] * (k - len(neighbors))

        # tmp!
        neighbors_per_view_distance.append(distances[order][:k].tolist())


        neighbors_per_view.append(neighbors)

    # return neighbors_per_view
    return neighbors_per_view, neighbors_per_view_distance

class MaskNVSDataset(Dataset):
    def __init__(self, 
        data_path,
        num_views,
        mask_dict,
        mask_dilate=None,
        select_strategy="fisherrf_knn",
        selection_dict=None,
        inpainted_dict=None,
        save_path=None,
        seed=95,
        num_map=None,
    ):
        self.data_path = data_path
        self.base_dir = os.path.dirname(data_path)

        self.images_info = json.load(open(f"{data_path}/opencv_cameras.json", "r"))
        self.image_size = (512,512)

        self.num_views = num_views
        self.mask_dict = mask_dict
        self.mask_dilate = mask_dilate

        self.select_strategy = select_strategy
        self.selection_dict = selection_dict
        self.inpainted_dict =inpainted_dict
        self.num_map = num_map
        self.save_path = save_path
        self.rng = random.Random(seed)

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        
        # Select only PNG files first, then choose views that are spatially close and look in similar directions
        png_indices = [i for i, info in enumerate(self.images_info) if os.path.splitext(info["file_path"])[1].lower() == ".png"]
        candidate_infos = [self.images_info[i] for i in png_indices] if len(png_indices) > 0 else self.images_info
        
        # Compute per-view k-NN indices on candidate infos
        if self.select_strategy == "random":
            neighbors_per_view = random_indices_for_each_view(candidate_infos, self.num_views, include_self=True, rng=self.rng)
            indices = [j for sub in neighbors_per_view for j in sub]     
        elif self.select_strategy == "fisherrf":
            idx_by_img = {info['file_path'].split('/')[-1]: i for i, info in enumerate(candidate_infos)}
            neighbors_per_view = []
            for idx, info in enumerate(candidate_infos):
                imgname = info['file_path'].split('/')[-1]
                indices = [
                    idx_by_img[name]
                    for name, num_gs in self.selection_dict[imgname].items()
                    ]
                indices.insert(0, idx)
                if len(indices) < self.num_views:
                    continue                
                neighbors_per_view.append(indices[:self.num_views])
            if len(neighbors_per_view) == 0:
                return None
            indices = [j for sub in neighbors_per_view for j in sub]
        else:
            neighbors_per_view, _ = knn_indices_for_each_view(candidate_infos, self.num_views, include_self=True)
            neighbors_flat = [j for sub in neighbors_per_view for j in sub]
            indices = neighbors_flat
        
        if len(indices) == 0: return None

        fxfycxcy_list = []
        c2w_list = []
        image_list = []
        name_list = []
        size_original_list = []
        mask_list = []

        for n, index in enumerate(indices):
            info = candidate_infos[index]
            fxfycxcy = [info["fx"], info["fy"], info["cx"], info["cy"]]
            
            w2c = torch.tensor(info["w2c"])
            c2w = torch.inverse(w2c)
            c2w_list.append(c2w)
            
            # Load image from file_path using PIL and convert to torch tensor
            imgname = info["file_path"].split('/')[-1]
            if self.inpainted_dict is not None:
                exist_inp = self.inpainted_dict.get(imgname, None)
                if exist_inp is not None:
                    image_path = f"./{self.save_path}/lact_results/{imgname}"
                else:
                    image_path = os.path.join(self.data_path, info["file_path"])
            else:
                image_path = os.path.join(self.data_path, info["file_path"])
            image = imageio.imread(image_path)
            size_original = image.shape[:2]
            image = Image.fromarray(image)

            alpha = self.mask_dict[imgname].squeeze().detach().cpu().numpy()
            image_mask = Image.fromarray(alpha)

            image, fxfycxcy = resize_and_crop(image, self.image_size, fxfycxcy)
            image_mask, _ = resize_and_crop(image_mask, self.image_size, fxfycxcy, resize_mode='nearest')
            if self.mask_dilate is not None:
                image_mask = image_mask.convert('L')
                image_mask = image_mask.filter(ImageFilter.MinFilter(size=self.mask_dilate))

            image = np.array(image)
            image_mask = np.array(image_mask)
            image[image_mask == 0] = 0
            image = Image.fromarray(image)
            image = image.convert('RGB')
            image_mask = Image.fromarray(image_mask)

            size_original_list.append(size_original)
            fxfycxcy_list.append(fxfycxcy)
            image_list.append(transforms.ToTensor()(image))
            name_list.append(os.path.basename(info["file_path"]))
            mask_list.append(transforms.ToTensor()(image_mask))
        
        c2ws = torch.stack(c2w_list)
        print("Normalizing scene poses...")
        c2ws = normalize_with_mean_pose(c2ws)        
        return {
            "fxfycxcy": torch.tensor(fxfycxcy_list),
            "c2w": c2ws,
            "image": torch.stack(image_list),
            "name": name_list,
            "size_original": size_original_list,
            "mask": torch.stack(mask_list),
        }
