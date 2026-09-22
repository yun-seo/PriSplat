import json
import os
import random
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import imageio
import numpy as np

def resize_and_crop_original(image, target_size, fxfycxcy, resize_mode='lanczos'):
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
    if resize_mode == 'lanczos':
        resized_image = image.resize((new_width, new_height), Image.LANCZOS) #lanczos
    elif resize_mode == 'nearest':
        resized_image = image.resize((new_width, new_height), Image.Resampling.NEAREST) #nearest neighbor
    else:
        raise ValueError(f"Invalid resize mode: {resize_mode}")
    
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

import random

def random_indices_for_each_view(images_info, k: int, include_self: bool = True, rng: random.Random | None = None):
    """
    각 뷰 i에 대해, 자기 자신(옵션)을 포함하여 k개의 인덱스를 랜덤으로 반환.
    리턴: List[List[int]] of shape [N][k]
    """
    N = len(images_info)
    if N == 0:
        return []

    if rng is None:
        rng = random

    neighbors_per_view = []
    all_idx = list(range(N))

    for i in range(N):
        if include_self:
            # 자기 자신 고정 + 나머지에서 k-1개 샘플 (중복 없이)
            others = [j for j in all_idx if j != i]
            m = min(k - 1, len(others))
            sampled = rng.sample(others, m)
            neigh = [i] + sampled
            # k개 안 찼으면 (N < k 같은 경우) 다른 인덱스나 자기 자신으로 채움
            while len(neigh) < k:
                # others가 비면 자기 자신으로 채움
                pick = rng.choice(others) if others else i
                neigh.append(pick)
        else:
            # 자기 자신 제외하고 k개 샘플
            pool = [j for j in all_idx if j != i]
            m = min(k, len(pool))
            neigh = rng.sample(pool, m)
            while len(neigh) < k:
                # 부족하면 중복 허용으로 채움
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
    
    # # Calculate crop box for center crop
    # left = (new_width - target_width) // 2
    # top = (new_height - target_height) // 2
    # right = left + target_width
    # bottom = top + target_height
    
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

def select_close_directional_indices(images_info, k: int):
    """
    Select k view indices that are spatially close and look in similar directions.
    Deterministic: chooses a central view (medoid) under a combined distance and
    returns the k closest views to it.
    """
    N = len(images_info)
    if N == 0:
        return []
    if k >= N:
        return list(range(N))
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
    # Choose center (medoid) that minimizes total combined distance
    center = int(torch.argmin(combined.sum(dim=1)).item())
    # Select top-k closest to the center
    order = torch.argsort(combined[center])
    return order[:k].tolist()

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

def select_all_closest_views_for_each(images_info, k: int, max_neighbors_per_view: int = None):
    """
    For each view, find its k closest views (including itself), then return all unique views.
    This ensures that every view gets represented along with its closest neighbors.
    
    Args:
        images_info: List of image info dictionaries
        k: Number of closest views to find for each view (including the view itself)
        max_neighbors_per_view: Maximum neighbors per view (if None, uses k)
    
    Returns:
        List of unique view indices that represent all views and their closest neighbors
    """
    N = len(images_info)
    if N == 0:
        return []
    if k >= N:
        return list(range(N))
    
    # Use a smaller number of neighbors per view to be more selective
    neighbors_per_view = max_neighbors_per_view if max_neighbors_per_view is not None else min(k, max(2, k // 4))
    
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
    
    # For each view, find its closest views
    selected_views = set()
    for i in range(N):
        # Get distances from view i to all other views
        distances = combined[i]
        # Find neighbors_per_view closest views (including itself since distance to self is 0)
        closest_indices = torch.argsort(distances)[:neighbors_per_view]
        selected_views.update(closest_indices.tolist())
    
    result = sorted(list(selected_views))
    print(f"Debug: N={N}, k={k}, neighbors_per_view={neighbors_per_view}, selected={len(result)} views")
    return result

def select_representative_views_with_neighbors(images_info, target_count: int, neighbors_per_representative: int = 3):
    """
    Select representative views distributed across the scene, then add their closest neighbors.
    This approach is more selective than including neighbors for every single view.
    
    Args:
        images_info: List of image info dictionaries  
        target_count: Target number of views to select
        neighbors_per_representative: Number of neighbors to add for each representative view
    
    Returns:
        List of view indices representing a good coverage of the scene
    """
    N = len(images_info)
    if N == 0:
        return []
    if target_count >= N:
        return list(range(N))
    
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
    
    # Step 1: Select representative views using farthest point sampling
    representatives = []
    remaining = set(range(N))
    
    # Start with the view that has maximum total distance to others (most isolated)
    total_distances = combined.sum(dim=1)
    first = int(torch.argmax(total_distances).item())
    representatives.append(first)
    remaining.remove(first)
    
    # Estimate how many representatives we need
    num_representatives = max(1, target_count // (neighbors_per_representative + 1))
    
    # Iteratively add the view that is farthest from all selected representatives
    for _ in range(min(num_representatives - 1, len(remaining))):
        if not remaining:
            break
        
        # Find the view with maximum minimum distance to any representative
        max_min_dist = -1
        best_candidate = None
        
        for candidate in remaining:
            min_dist_to_reps = min(combined[candidate, rep].item() for rep in representatives)
            if min_dist_to_reps > max_min_dist:
                max_min_dist = min_dist_to_reps
                best_candidate = candidate
        
        if best_candidate is not None:
            representatives.append(best_candidate)
            remaining.remove(best_candidate)
    
    # Step 2: For each representative, add its closest neighbors
    selected_views = set(representatives)
    
    for rep in representatives:
        distances = combined[rep]
        # Find closest neighbors (excluding already selected ones when possible)
        closest_indices = torch.argsort(distances)
        added = 0
        for idx in closest_indices:
            if added >= neighbors_per_representative:
                break
            idx_val = int(idx.item())
            if idx_val not in selected_views or len(selected_views) < target_count:
                selected_views.add(idx_val)
                added += 1
    
    result = sorted(list(selected_views))[:target_count]  # Limit to target count
    print(f"Debug: N={N}, target={target_count}, representatives={len(representatives)}, final_selected={len(result)} views")
    return result

def _get_intrinsics(info):
    """
    Extract intrinsics from a per-view dict.
    Supports either a full K matrix or (fx, fy, cx, cy) with (H, W).
    """
    if "K" in info:
        K = torch.tensor(info["K"], dtype=torch.float32)
        fx, fy = K[0, 0].item(), K[1, 1].item()
        cx, cy = K[0, 2].item(), K[1, 2].item()
    else:
        fx, fy = float(info["fx"]), float(info["fy"])
        cx, cy = float(info["cx"]), float(info["cy"])

    H, W = int(info["h"]), int(info["w"])
    return fx, fy, cx, cy, H, W

@torch.no_grad()
def frustum_overlap_neighbors(images_info, k: int, include_self: bool = True,
                              grid: int = 32, device: str = "cpu"):
    """
    For each view i, select k views whose frustums overlap the most with view i.
    Overlap is approximated by: cast a coarse pixel grid of rays from view i,
    rotate those directions into each view j, and count the fraction of rays that
    fall inside j's image plane and are in front of the camera (z>0).

    Args:
        images_info: List[dict], each dict must contain:
            - "w2c": 4x4 world-to-camera matrix (list/np/torch)
            - Either:
                * "K": 3x3 intrinsics matrix
              or * "fx","fy","cx","cy"
            - "H","W": image height/width
        k:            Number of neighbors to return for each view
        include_self: Whether to allow the view itself in its neighbor list
        grid:         Grid resolution per axis for ray sampling (e.g., 32 -> 32x32 rays)
        device:       Torch device string or torch.device

    Returns:
        neighbors: List[List[int]] of length N; each row contains k indices
                   sorted by decreasing frustum-overlap with the row's view.
    """
    N = len(images_info)
    if N == 0:
        return []
    k = max(1, k)

    # Collect extrinsics (w2c) and compute c2w
    w2c = torch.stack(
        [torch.as_tensor(v["w2c"], dtype=torch.float32) for v in images_info],
        dim=0
    ).to(device)                                     # [N, 4, 4]
    c2w = torch.linalg.inv(w2c)                      # [N, 4, 4]

    # Rotations only (directions are enough; origins are not needed here)
    R_wc = c2w[:, :3, :3]                            # world <- camera
    R_cw = w2c[:, :3, :3]                            # camera <- world

    # Collect intrinsics and image sizes
    intr = [_get_intrinsics(info) for info in images_info]
    fx = torch.tensor([t[0] for t in intr], device=device)  # [N]
    fy = torch.tensor([t[1] for t in intr], device=device)
    cx = torch.tensor([t[2] for t in intr], device=device)
    cy = torch.tensor([t[3] for t in intr], device=device)
    Hs = torch.tensor([t[4] for t in intr], device=device)
    Ws = torch.tensor([t[5] for t in intr], device=device)

    # Overlap[i, j] = fraction of i's sampled rays that land inside j
    overlap = torch.zeros((N, N), device=device)

    for i in range(N):
        # Build a coarse pixel grid for view i (adapt to each view's size)
        H_i, W_i = int(Hs[i].item()), int(Ws[i].item())
        gi = min(grid, H_i)
        gj = min(grid, W_i)
        ys = torch.linspace(0, H_i - 1, steps=gi, device=device)
        xs = torch.linspace(0, W_i - 1, steps=gj, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")      # [gi, gj]

        # Rays in camera-i space at unit depth (pixel-center approx)
        dir_cam = torch.stack(
            [(xx - cx[i]) / fx[i], (yy - cy[i]) / fy[i], torch.ones_like(xx)],
            dim=-1
        )                                                   # [gi, gj, 3]
        dir_cam = F.normalize(dir_cam, dim=-1).reshape(-1, 3)  # [M, 3], M=gi*gj

        # Rotate to world space (directions only)
        dir_w = dir_cam @ R_wc[i].T                        # [M, 3]

        # Rotate world directions into each camera j: dir_camj = R_cw[j] * dir_w
        # Result shape: [N, M, 3]
        dir_camj = torch.einsum('md,ndk->nmk', dir_w, R_cw.transpose(1, 2))

        z = dir_camj[..., 2]                               # [N, M]
        x = dir_camj[..., 0] / (z + 1e-8)
        y = dir_camj[..., 1] / (z + 1e-8)

        # Project to pixel coordinates in each view j
        u = fx[:, None] * x + cx[:, None]                  # [N, M]
        v = fy[:, None] * y + cy[:, None]                  # [N, M]

        # Inside FOV and in front of the camera
        in_front = z > 0
        in_u = (u >= 0) & (u <= (Ws[:, None] - 1))
        in_v = (v >= 0) & (v <= (Hs[:, None] - 1))
        inside = in_front & in_u & in_v                    # [N, M]

        overlap[i] = inside.float().mean(dim=1)            # [N]

    # Optionally exclude self
    if not include_self:
        overlap.fill_diagonal_(-1.0)

    # Pick top-k by overlap (larger is better)
    k_eff = min(k, N if include_self else max(N - 1, 1))
    _, idxs = torch.topk(overlap, k=k_eff, largest=True, dim=1)  # [N, k_eff]

    # Pad rare edge cases and return as Python lists
    neighbors = []
    for i in range(N):
        row = idxs[i].tolist()
        if len(row) < k:
            fill = i if include_self else (row[0] if row else i)
            row = row + [fill] * (k - len(row))
        neighbors.append(row[:k])
    return neighbors

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
        neighbors_per_view.append(neighbors)

    return neighbors_per_view


class NVSDataset(Dataset):
    def __init__(self, 
        data_path, num_views, image_size, 
        sorted_indices=False, 
        scene_pose_normalize=False,
        mask_dilate=False,
        select_strategy="knn",  # <- "knn" | "random"
        seed=95    
    ):
        """
        image_size is (h, w) or just a int (as size).
        """
        self.base_dir = os.path.dirname(data_path)
        self.data_point_paths = json.load(open(data_path, "r"))
        self.sorted_indices = sorted_indices
        self.scene_pose_normalize = scene_pose_normalize

        self.num_views = num_views
        self.mask_dilate = mask_dilate
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

        self.select_strategy = select_strategy
        self.rng = random.Random(seed)  # 랜덤 선택 재현성

    def __len__(self):
        return len(self.data_point_paths)
    
    def __getitem__(self, index):
        data_point_path = os.path.join(self.base_dir, self.data_point_paths[index])
        data_point_base_dir = os.path.dirname(data_point_path)
        with open(data_point_path, "r") as f:
            images_info = json.load(f)
        
        
        # Select only PNG files first, then choose views that are spatially close and look in similar directions
        png_indices = [i for i, info in enumerate(images_info) if os.path.splitext(info["file_path"])[1].lower() == ".png"]
        candidate_infos = [images_info[i] for i in png_indices] if len(png_indices) > 0 else images_info
        
        # 내가 하고싶은거: 모든 뷰들에 대해서, 각자 뷰에서 가장 가까운 뷰들을 찾아서 그 뷰들을 모두 선택하고 싶다.
        # Option 1: For each view, find its closest neighbors (more selective)
        # indices = select_all_closest_views_for_each(candidate_infos, self.num_views, max_neighbors_per_view=3)
        
        # Compute per-view k-NN indices on candidate infos
        if self.select_strategy == "random":
            neighbors_per_view = random_indices_for_each_view(candidate_infos, self.num_views, include_self=True, rng=self.rng)
            indices = [j for sub in neighbors_per_view for j in sub]
        else:
            neighbors_per_view = knn_indices_for_each_view(candidate_infos, self.num_views, include_self=True)
            neighbors_flat = [j for sub in neighbors_per_view for j in sub]
            indices = neighbors_flat
        
        # Group all candidates into self.num_views adjacent groups and flatten the order
        # groups_rel, order_rel = group_all_views_into_groups_of_size(candidate_infos, self.num_views)
        # indices = order_rel
        # if self.sorted_indices:
            # indices = sorted(indices)
        
        fxfycxcy_list = []
        c2w_list = []
        image_list = []
        name_list = []
        size_original_list = []
        
        
        for index in indices:
            # info = images_info[index]
            info = candidate_infos[index]
            
            fxfycxcy = [info["fx"], info["fy"], info["cx"], info["cy"]]
            
            w2c = torch.tensor(info["w2c"])
            c2w = torch.inverse(w2c)
            c2w_list.append(c2w)
            
            # Load image from file_path using PIL and convert to torch tensor
            image_path = os.path.join(data_point_base_dir, info["file_path"])
            if 'robust' in image_path:
                # image_path = image_path.replace(image_path.split('/')[-1], 'images/'+image_path.split('/')[-1])
                # if os.path.exists(image_path):
                #     image = Image.open(image_path)
                # else:
                #     image_path = image_path.replace('.JPG', '.png')
                image = imageio.imread(image_path)
                size_original = image.shape[:2]
                if image.shape[-1] == 4:
                    rgb, alpha = image[..., :3], image[..., 3:]
                    
                    alpha = alpha[:, :, 0]
                    alpha = np.stack([alpha, alpha, alpha], axis=-1)
                    # alpha = alpha / 255.0  # [0,1] 범위로 정규화
                    # breakpoint()
                    # rgb[alpha.squeeze() == 0] = 0
                    image = rgb
                    image = Image.fromarray(image)
                    image_mask = Image.fromarray(alpha)
                    # breakpoint()
                    image, fxfycxcy = resize_and_crop(image, self.image_size, fxfycxcy)
                    image_mask, _ = resize_and_crop(image_mask, self.image_size, fxfycxcy, resize_mode='nearest')
                    if self.mask_dilate:
                        image_mask = image_mask.convert('L')
                        image_mask = image_mask.filter(ImageFilter.MinFilter(size=7))

                    image = np.array(image)
                    image_mask = np.array(image_mask)
                    image[image_mask == 0] = 0
                    image = Image.fromarray(image)
                else:
                    image = Image.fromarray(image)
                    image, fxfycxcy = resize_and_crop(image, self.image_size, fxfycxcy)
            else:
                image = Image.open(image_path)
                size_original = image.size
                image, fxfycxcy = resize_and_crop(image, self.image_size, fxfycxcy)
            size_original_list.append(size_original)
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
            name_list.append(os.path.basename(info["file_path"]))
        
        c2ws = torch.stack(c2w_list)
        if self.scene_pose_normalize:
            print("Normalizing scene poses...")
            c2ws = normalize_with_mean_pose(c2ws)

        # # Map relative group indices to absolute indices for reference/debugging
        # if len(png_indices) > 0:
        #     view_groups = [[png_indices[i] for i in grp] for grp in groups_rel]
        # else:
        #     view_groups = groups_rel

        return {
            "fxfycxcy": torch.tensor(fxfycxcy_list),
            "c2w": c2ws,
            "image": torch.stack(image_list),
            "name": name_list,
            "size_original": size_original_list,
            # "view_groups": view_groups,
        }

class MaskNVSDataset(Dataset):
    def __init__(self, 
        data_path, num_views, image_size, 
        sorted_indices=False, 
        scene_pose_normalize=False,
        mask_dilate=False,
        select_strategy="knn",  # <- "knn" | "random"
        seed=95, selected_filepath=None,
    ):
        """
        image_size is (h, w) or just a int (as size).
        """
        self.base_dir = os.path.dirname(data_path)
        self.data_point_paths = json.load(open(data_path, "r"))
        self.sorted_indices = sorted_indices
        self.scene_pose_normalize = scene_pose_normalize

        self.num_views = num_views
        self.mask_dilate = mask_dilate
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

        self.select_strategy = select_strategy
        self.rng = random.Random(seed)
        self.selected_filepath = selected_filepath

    def __len__(self):
        return len(self.data_point_paths)
    
    def __getitem__(self, index):
        data_point_path = os.path.join(self.base_dir, self.data_point_paths[index])
        data_point_base_dir = os.path.dirname(data_point_path)
        with open(data_point_path, "r") as f:
            images_info = json.load(f)
        import pdb; pdb.set_trace()
        
        # Select only PNG files first, then choose views that are spatially close and look in similar directions
        png_indices = [i for i, info in enumerate(images_info) if os.path.splitext(info["file_path"])[1].lower() == ".png"]
        candidate_infos = [images_info[i] for i in png_indices] if len(png_indices) > 0 else images_info
        
        # 내가 하고싶은거: 모든 뷰들에 대해서, 각자 뷰에서 가장 가까운 뷰들을 찾아서 그 뷰들을 모두 선택하고 싶다.
        # Option 1: For each view, find its closest neighbors (more selective)
        # indices = select_all_closest_views_for_each(candidate_infos, self.num_views, max_neighbors_per_view=3)
        
        # Compute per-view k-NN indices on candidate infos
        if self.select_strategy == "random":
            neighbors_per_view = random_indices_for_each_view(candidate_infos, self.num_views, include_self=True, rng=self.rng)
            indices = [j for sub in neighbors_per_view for j in sub]
        elif self.select_strategy == "fisherrf":
            idx_by_img = {info['file_path'].split('/')[-1]: i for i, info in enumerate(candidate_infos)}
            with open(self.selected_filepath, "r", encoding="utf-8") as f:
                selected_dict = json.load(f)
            neighbors_per_view = []
            for idx, info in enumerate(candidate_infos):
                imgname = info['file_path'].split('/')[-1]
                indices = [idx_by_img[name] for name in (selected_dict.get(imgname, {}) or {}) if name != imgname]
                indices.insert(0, idx)
                if len(indices) < self.num_views:
                    continue
                neighbors_per_view.append(indices[:self.num_views])

            indices = [j for sub in neighbors_per_view for j in sub]        
        elif self.select_strategy == "fisherrf_and_knn":
            idx_by_img = {info['file_path'].split('/')[-1]: i for i, info in enumerate(candidate_infos)}
            with open(self.selected_filepath, "r", encoding="utf-8") as f:
                selected_dict = json.load(f)
            knn_indices = frustum_overlap_neighbors(candidate_infos, self.num_views*2, include_self=True)
            # knn_indices = knn_indices_for_each_view(candidate_infos, self.num_views*2, include_self=True)
            neighbors_per_view = []
            for idx, info in enumerate(candidate_infos):
                imgname = info['file_path'].split('/')[-1]
                indices = [idx_by_img[name] for name in (selected_dict.get(imgname, {}) or {})]
                indices = [x for x in knn_indices[idx] if x in set(indices)]                
                indices.insert(0, idx)
                if len(indices) < self.num_views:
                    continue
                neighbors_per_view.append(indices[:self.num_views])
                # neighbors_per_view.append(indices[:self.num_views//2] + knn_indices[idx][1:self.num_views//2+1])
            indices = [j for sub in neighbors_per_view for j in sub]             
        elif self.select_strategy == "opacity":
            idx_by_img = {info['file_path'].split('/')[-1]: i for i, info in enumerate(candidate_infos)}
            with open(self.selected_filepath, "r", encoding="utf-8") as f:
                selected_dict = json.load(f)
            neighbors_per_view = []
            for idx, info in enumerate(candidate_infos):
                imgname = info['file_path'].split('/')[-1]
                indices = [
                    idx_by_img[name]
                    for name, num_gs in (selected_dict.get(imgname, {}) or {}).items()
                    if num_gs > 0.01 and name in idx_by_img
                ]
                indices.insert(0, idx)
                if len(indices) < self.num_views:
                    continue
                neighbors_per_view.append(indices[:self.num_views])
            indices = [j for sub in neighbors_per_view for j in sub]
        elif self.select_strategy == "opacity_and_knn":
            idx_by_img = {info['file_path'].split('/')[-1]: i for i, info in enumerate(candidate_infos)}
            with open(self.selected_filepath, "r", encoding="utf-8") as f:
                selected_dict = json.load(f)
            neighbors_per_view = []
            # knn_indices = frustum_overlap_neighbors(candidate_infos, self.num_views*2, include_self=True)
            knn_indices = knn_indices_for_each_view(candidate_infos, self.num_views, include_self=True)
            for idx, info in enumerate(candidate_infos):
                imgname = info['file_path'].split('/')[-1]
                indices = [
                    idx_by_img[name]
                    for name, num_gs in selected_dict[imgname].items()
                     if num_gs > 0.005
                     ]
                indices = [x for x in knn_indices[idx][1:] if x in set(indices)]   
                indices.insert(0, idx)
                if len(indices) < self.num_views:
                    continue                
                # indices = knn_indices[idx][:self.num_views//2] + indices
                neighbors_per_view.append(indices[:self.num_views])
            indices = [j for sub in neighbors_per_view for j in sub]
        else:
            neighbors_per_view = knn_indices_for_each_view(candidate_infos, self.num_views, include_self=True)
            # # tmp!
            # save_dict = dict()
            # idx_by_img = {i: info['file_path'].split('/')[-1] for i, info in enumerate(candidate_infos)}
            # for name_list in neighbors_per_view:
            #     save_dict[idx_by_img[name_list[0]]] = [idx_by_img[i] for i in name_list]
            # with open(f"tmp.txt", "w", encoding="utf-8") as f:
            #     json.dump(save_dict, f, ensure_ascii=False, indent=2)            
            # import pdb; pdb.set_trace()

            neighbors_flat = [j for sub in neighbors_per_view for j in sub]
            indices = neighbors_flat
        
        # Group all candidates into self.num_views adjacent groups and flatten the order
        # groups_rel, order_rel = group_all_views_into_groups_of_size(candidate_infos, self.num_views)
        # indices = order_rel
        # if self.sorted_indices:
            # indices = sorted(indices)
        
        fxfycxcy_list = []
        c2w_list = []
        image_list = []
        name_list = []
        size_original_list = []
        mask_list = []
        
        
        for index in indices:
            # info = images_info[index]
            info = candidate_infos[index]
            
            fxfycxcy = [info["fx"], info["fy"], info["cx"], info["cy"]]
            
            w2c = torch.tensor(info["w2c"])
            c2w = torch.inverse(w2c)
            c2w_list.append(c2w)
            
            # Load image from file_path using PIL and convert to torch tensor
            image_path = os.path.join(data_point_base_dir, info["file_path"])
            if 'robust' in image_path:
                # image_path = image_path.replace(image_path.split('/')[-1], 'images/'+image_path.split('/')[-1])
                # if os.path.exists(image_path):
                #     image = Image.open(image_path)
                # else:
                #     image_path = image_path.replace('.JPG', '.png')
                image = imageio.imread(image_path)
                size_original = image.shape[:2]
                if image.shape[-1] == 4:
                    rgb, alpha = image[..., :3], image[..., 3:]
                    
                    alpha = alpha[:, :, 0]
                    alpha = np.stack([alpha, alpha, alpha], axis=-1)
                    # alpha = alpha / 255.0  # [0,1] 범위로 정규화
                    # breakpoint()
                    # rgb[alpha.squeeze() == 0] = 0
                    image = rgb
                    image = Image.fromarray(image)
                    image_mask = Image.fromarray(alpha[:,:,0])
                    # breakpoint()
                    image, fxfycxcy = resize_and_crop(image, self.image_size, fxfycxcy)
                    image_mask, _ = resize_and_crop(image_mask, self.image_size, fxfycxcy, resize_mode='nearest')
                    if self.mask_dilate:
                        image_mask = image_mask.convert('L')
                        image_mask = image_mask.filter(ImageFilter.MinFilter(size=15))

                    image = np.array(image)
                    image_mask = np.array(image_mask)
                    image[image_mask == 0] = 0
                    image = Image.fromarray(image)
                    image_mask = Image.fromarray(image_mask)
                else:
                    image = Image.fromarray(image)
                    image, fxfycxcy = resize_and_crop(image, self.image_size, fxfycxcy)
            else:
                image = Image.open(image_path)
                size_original = image.size
                image, fxfycxcy = resize_and_crop(image, self.image_size, fxfycxcy)
            size_original_list.append(size_original)
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
            name_list.append(os.path.basename(info["file_path"]))
            if 'robust' in image_path:
                mask_list.append(transforms.ToTensor()(image_mask))
        
        c2ws = torch.stack(c2w_list)
        if self.scene_pose_normalize:
            print("Normalizing scene poses...")
            c2ws = normalize_with_mean_pose(c2ws)

        # # Map relative group indices to absolute indices for reference/debugging
        # if len(png_indices) > 0:
        #     view_groups = [[png_indices[i] for i in grp] for grp in groups_rel]
        # else:
        #     view_groups = groups_rel

        if 'robust' in image_path:
            return {
                "fxfycxcy": torch.tensor(fxfycxcy_list),
                "c2w": c2ws,
                "image": torch.stack(image_list),
                "name": name_list,
                "size_original": size_original_list,
                "mask": torch.stack(mask_list),
                # "view_groups": view_groups,
            }
        else:
            return {
                "fxfycxcy": torch.tensor(fxfycxcy_list),
                "c2w": c2ws,
                "image": torch.stack(image_list),
                "name": name_list,
                "size_original": size_original_list,
                # "view_groups": view_groups,
            }


