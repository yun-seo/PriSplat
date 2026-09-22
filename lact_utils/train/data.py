import json
import os
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F


def resize_and_crop(image, target_size, fxfycxcy):
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
    scale = max(scale_x, scale_y)  # Use larger scale to ensure it covers the target area
    
    # Resize image
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate crop box for center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    # Crop image
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    # Adjust camera parameters
    # Scale focal lengths and principal points
    new_fx = fx * scale
    new_fy = fy * scale
    new_cx = cx * scale - left
    new_cy = cy * scale - top
    
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

def group_all_views_into_groups_of_size(images_info, k: int):
    """
    Group all views into contiguous groups of size k along a nearest-neighbor path
    under a combined (position + direction) distance. Ensures every camera is included.
    If the total count N is not divisible by k, the last group may be smaller.
    Returns (groups_rel, order_rel), where indices are relative to images_info.
    """
    N = len(images_info)
    if N == 0:
        return [], []
    # Effective group size (at least 1)
    g = max(1, k)
    # Build c2w for all views without loading images
    c2ws = []
    for info in images_info:
        w2c = torch.tensor(info["w2c"], dtype=torch.float32)
        c2ws.append(torch.inverse(w2c))
    c2ws = torch.stack(c2ws, dim=0)  # [N, 4, 4]
    positions = c2ws[:, :3, 3]
    directions = F.normalize(c2ws[:, :3, 2], dim=1)
    # Pairwise distances
    pos_dist = torch.cdist(positions, positions)  # [N, N]
    cos_sim = torch.clamp(directions @ directions.t(), -1.0, 1.0)
    ang_dist = 1.0 - cos_sim
    # Normalize scales
    pos_mask = pos_dist > 0
    ang_mask = ang_dist > 0
    pos_scale = float(torch.median(pos_dist[pos_mask])) if torch.any(pos_mask) else 1.0
    ang_scale = float(torch.median(ang_dist[ang_mask])) if torch.any(ang_mask) else 1.0
    combined = (pos_dist / pos_scale) + (ang_dist / ang_scale)
    # Build a nearest-neighbor path (TSP heuristic)
    total_cost = combined.sum(dim=1)
    start = int(torch.argmin(total_cost).item())
    visited = torch.zeros(N, dtype=torch.bool)
    order = []
    cur = start
    for _ in range(N):
        order.append(cur)
        visited[cur] = True
        # Mask visited nodes with large cost
        row = combined[cur].clone()
        row[visited] = float('inf')
        # pick nearest unvisited
        if visited.all():
            break
        nxt = int(torch.argmin(row).item())
        cur = nxt
    order_rel = order
    # Split the path into contiguous groups of size g
    groups_rel = [order_rel[i:i+g] for i in range(0, N, g)]
    return groups_rel, order_rel

class NVSDataset(Dataset):
    def __init__(self, 
        data_path, num_views, image_size, 
        sorted_indices=False, 
        scene_pose_normalize=False,
    ):
        """
        image_size is (h, w) or just a int (as size).
        """
        self.base_dir = os.path.dirname(data_path)
        self.data_point_paths = json.load(open(data_path, "r"))
        self.sorted_indices = sorted_indices
        self.scene_pose_normalize = scene_pose_normalize

        self.num_views = num_views
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

    def __len__(self):
        return len(self.data_point_paths)
    
    def __getitem__(self, index):
        
        data_point_path = os.path.join(self.base_dir, self.data_point_paths[index])
        data_point_base_dir = os.path.dirname(data_point_path)
        with open(data_point_path, "r") as f:
            images_info = json.load(f)
        
        # # # If the num_views is larger than the number of images, use all images
        # indices = random.sample(range(len(images_info)), min(self.num_views, len(images_info)))
        
        num_candidate_views = min(self.num_views * 1, len(images_info))
        groups_rel, order_rel = group_all_views_into_groups_of_size(images_info, num_candidate_views)
        try:
            indices = groups_rel[:-1][random.randint(0, len(groups_rel) - 2)]
        except:
            print(groups_rel)
            print(data_point_base_dir)
            
        
        if self.sorted_indices:
            indices = sorted(indices)        
        fxfycxcy_list = []
        c2w_list = []
        image_list = []
        
        for index in indices:
            info = images_info[index]
            
            fxfycxcy = [info["fx"], info["fy"], info["cx"], info["cy"]]
            
            w2c = torch.tensor(info["w2c"])
            c2w = torch.inverse(w2c)
            c2w_list.append(c2w)
            
            # Load image from file_path using PIL and convert to torch tensor
            image_path = os.path.join(data_point_base_dir, info["file_path"])
            image = Image.open(image_path)
            
            image, fxfycxcy = resize_and_crop(image, self.image_size, fxfycxcy)
            

            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                # Create a white background and paste the RGBA image on it
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = rgb_image
            elif image.mode != 'RGB':
                # Convert any other mode to RGB
                image = image.convert('RGB')
            
            fxfycxcy_list.append(fxfycxcy)
            image_list.append(transforms.ToTensor()(image))
        
        c2ws = torch.stack(c2w_list)
        if self.scene_pose_normalize:
            # print("Normalizing scene poses...")
            c2ws = normalize_with_mean_pose(c2ws)

        return {
            "fxfycxcy": torch.tensor(fxfycxcy_list),
            "c2w": c2ws,
            "image": torch.stack(image_list),
        }