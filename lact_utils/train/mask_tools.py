#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


# -----------------------------
# Utilities
# -----------------------------

def load_mask(path: Path) -> np.ndarray:
	"""Load an image as a binary mask (values {0,1}). Any nonzero pixel becomes 1.
	Returns array of shape (H, W), dtype=uint8.
	"""
	img = Image.open(path).convert("L")
	arr = np.array(img)
	# Threshold: treat any > 0 as foreground
	mask = (arr > 0).astype(np.uint8)
	return mask


def get_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
	"""Return (min_row, min_col, max_row_inclusive, max_col_inclusive) for foreground pixels.
	If mask is empty, returns (0,0,-1,-1)."""
	rows = np.any(mask, axis=1)
	cols = np.any(mask, axis=0)
	if not rows.any() or not cols.any():
		return 0, 0, -1, -1
	min_r = int(np.argmax(rows))
	max_r = int(len(rows) - 1 - np.argmax(rows[::-1]))
	min_c = int(np.argmax(cols))
	max_c = int(len(cols) - 1 - np.argmax(cols[::-1]))
	return min_r, min_c, max_r, max_c


def perimeter_4conn(mask: np.ndarray) -> int:
	"""Approximate perimeter using 4-connectivity edge transitions.
	Counts edges between foreground and background in 4-neighborhood."""
	if mask.size == 0:
		return 0
	# Horizontal transitions
	h_diff = np.abs(mask[:, 1:] - mask[:, :-1])
	# Vertical transitions
	v_diff = np.abs(mask[1:, :] - mask[:-1, :])
	perim = int(h_diff.sum() + v_diff.sum())
	# Add outer border contributions
	perim += int(mask[0, :].sum())  # top border
	perim += int(mask[-1, :].sum())  # bottom border
	perim += int(mask[:, 0].sum())  # left border
	perim += int(mask[:, -1].sum())  # right border
	return perim


class UnionFind:
	def __init__(self, n: int) -> None:
		self.parent = list(range(n))
		self.rank = [0] * n

	def find(self, x: int) -> int:
		while self.parent[x] != x:
			self.parent[x] = self.parent[self.parent[x]]
			x = self.parent[x]
		return x

	def union(self, a: int, b: int) -> None:
		ra = self.find(a)
		rb = self.find(b)
		if ra == rb:
			return
		if self.rank[ra] < self.rank[rb]:
			self.parent[ra] = rb
		elif self.rank[rb] < self.rank[ra]:
			self.parent[rb] = ra
		else:
			self.parent[rb] = ra
			self.rank[ra] += 1


def connected_components_4(mask: np.ndarray) -> Tuple[np.ndarray, int, Dict[int, int]]:
	"""Label connected components (4-connected) on binary mask {0,1}.
	Returns (labels, num_labels, label_to_area). Background is 0 in labels.
	Two-pass algorithm with union-find."""
	h, w = mask.shape
	labels = np.zeros((h, w), dtype=np.int32)
	uf = UnionFind(h * w // 2 + 1)  # heuristic capacity
	current_label = 1
	# First pass
	for r in range(h):
		for c in range(w):
			if mask[r, c] == 0:
				continue
			up = labels[r - 1, c] if r > 0 else 0
			left = labels[r, c - 1] if c > 0 else 0
			if up == 0 and left == 0:
				labels[r, c] = current_label
				current_label += 1
			elif up != 0 and left == 0:
				labels[r, c] = up
			elif up == 0 and left != 0:
				labels[r, c] = left
			else:
				# both nonzero, choose min and union
				lab = up if up < left else left
				labels[r, c] = lab
				if up != left:
					uf.union(up, left)
	# Second pass: resolve equivalences
	label_map: Dict[int, int] = {}
	next_label = 1
	for r in range(h):
		for c in range(w):
			lab = labels[r, c]
			if lab == 0:
				continue
			root = uf.find(lab)
			if root not in label_map:
				label_map[root] = next_label
				next_label += 1
			labels[r, c] = label_map[root]
	# Areas
	areas: Dict[int, int] = {}
	for r in range(h):
		for c in range(w):
			lab = labels[r, c]
			if lab == 0:
				continue
			areas[lab] = areas.get(lab, 0) + 1
	return labels, next_label - 1, areas


def count_holes(mask: np.ndarray) -> int:
	"""Count holes: connected components of background that do NOT touch image border."""
	inv = 1 - mask
	labels, nlab, _areas = connected_components_4(inv)
	if nlab == 0:
		return 0
	# Determine which labels touch border
	h, w = inv.shape
	border_labels = set(np.unique(np.concatenate([
		labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
	])))
	border_labels.discard(0)
	all_labels = set(np.unique(labels))
	all_labels.discard(0)
	interior = all_labels - border_labels
	return len(interior)


@dataclass
class MaskFeatures:
	filename: str
	height: int
	width: int
	area: int
	area_fraction: float
	num_components: int
	component_areas: List[int]
	perimeter: int
	compactness: float  # 4*pi*A / P^2
	bbox_height: int
	bbox_width: int
	bbox_aspect: float
	holes: int
	image_euler_number: int  # components - holes


@dataclass
class AggregateStats:
	num_images: int
	height: int
	width: int
	area_fraction_mean: float
	area_fraction_std: float
	num_components_hist: Dict[str, int]
	component_area_quantiles: Dict[str, float]
	hole_probability: float
	compactness_quantiles: Dict[str, float]
	bbox_aspect_quantiles: Dict[str, float]


def compute_features(mask: np.ndarray, filename: str) -> MaskFeatures:
	h, w = mask.shape
	area = int(mask.sum())
	area_fraction = area / float(h * w) if h * w > 0 else 0.0
	labels, ncomp, areas = connected_components_4(mask)
	comp_areas = sorted([areas[k] for k in areas])
	perim = perimeter_4conn(mask)
	compactness = (4.0 * math.pi * area / (perim * perim)) if perim > 0 else 0.0
	min_r, min_c, max_r, max_c = get_bounding_box(mask)
	bbox_h = max(0, max_r - min_r + 1)
	bbox_w = max(0, max_c - min_c + 1)
	bbox_aspect = (bbox_w / bbox_h) if bbox_h > 0 else 0.0
	holes = count_holes(mask)
	return MaskFeatures(
		filename=filename,
		height=h,
		width=w,
		area=area,
		area_fraction=area_fraction,
		num_components=ncomp,
		component_areas=comp_areas,
		perimeter=perim,
		compactness=compactness,
		bbox_height=bbox_h,
		bbox_width=bbox_w,
		bbox_aspect=bbox_aspect,
		holes=holes,
		image_euler_number=ncomp - holes,
	)


def aggregate_stats(features: List[MaskFeatures]) -> AggregateStats:
	if not features:
		raise ValueError("No features provided")
	h = features[0].height
	w = features[0].width
	area_fracs = np.array([f.area_fraction for f in features], dtype=np.float64)
	ncomp_hist: Dict[str, int] = {}
	for f in features:
		key = str(f.num_components)
		ncomp_hist[key] = ncomp_hist.get(key, 0) + 1
	# Collect all component areas across images (normalize by image area)
	all_comp_area_fracs: List[float] = []
	for f in features:
		if f.width * f.height == 0:
			continue
		for a in f.component_areas:
			all_comp_area_fracs.append(a / float(f.width * f.height))
	comp_q = np.quantile(all_comp_area_fracs, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist() if all_comp_area_fracs else [0, 0, 0, 0, 0]
	comp_q_named = {"q05": comp_q[0], "q25": comp_q[1], "q50": comp_q[2], "q75": comp_q[3], "q95": comp_q[4]}
	compactness_vals = np.array([f.compactness for f in features], dtype=np.float64)
	compct_q = np.quantile(compactness_vals, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
	compct_q_named = {"q05": compct_q[0], "q25": compct_q[1], "q50": compct_q[2], "q75": compct_q[3], "q95": compct_q[4]}
	bbox_aspects = np.array([f.bbox_aspect for f in features], dtype=np.float64)
	bbox_q = np.quantile(bbox_aspects, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
	bbox_q_named = {"q05": bbox_q[0], "q25": bbox_q[1], "q50": bbox_q[2], "q75": bbox_q[3], "q95": bbox_q[4]}
	hole_prob = float(np.mean([1 if f.holes > 0 else 0 for f in features]))
	return AggregateStats(
		num_images=len(features),
		height=h,
		width=w,
		area_fraction_mean=float(area_fracs.mean()),
		area_fraction_std=float(area_fracs.std(ddof=0)),
		num_components_hist=ncomp_hist,
		component_area_quantiles=comp_q_named,
		hole_probability=hole_prob,
		compactness_quantiles=compct_q_named,
		bbox_aspect_quantiles=bbox_q_named,
	)


# -----------------------------
# Generator
# -----------------------------

@dataclass
class GeneratorParams:
	height: int
	width: int
	num_components_hist: Dict[str, int]
	component_area_quantiles: Dict[str, float]
	bbox_aspect_quantiles: Dict[str, float]
	hole_probability: float
	smooth_sigma: float = 0.6
	area_scale: float = 1.0

	@staticmethod
	def from_aggregate(stats: AggregateStats) -> "GeneratorParams":
		return GeneratorParams(
			height=stats.height,
			width=stats.width,
			num_components_hist=stats.num_components_hist,
			component_area_quantiles=stats.component_area_quantiles,
			bbox_aspect_quantiles=stats.bbox_aspect_quantiles,
			hole_probability=stats.hole_probability,
			smooth_sigma=0.6,
			area_scale=1.0,
		)


def sample_from_hist(hist: Dict[str, int]) -> int:
	items = sorted(((int(k), v) for k, v in hist.items()), key=lambda x: x[0])
	total = sum(v for _, v in items)
	if total <= 0:
		return 1
	r = random.uniform(0, total)
	acc = 0.0
	for k, v in items:
		acc += v
		if r <= acc:
			return k
	return items[-1][0]


def sample_between_quantiles(qs: Dict[str, float]) -> float:
	# Piecewise between quantiles with equal probability mass per segment
	q05, q25, q50, q75, q95 = qs["q05"], qs["q25"], qs["q50"], qs["q75"], qs["q95"]
	segments = [(q05, q25), (q25, q50), (q50, q75), (q75, q95)]
	seg = random.choice(segments)
	return random.uniform(seg[0], seg[1])


def gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
	if sigma <= 0:
		return np.array([1.0], dtype=np.float64)
	x = np.arange(-radius, radius + 1, dtype=np.float64)
	k = np.exp(-(x * x) / (2 * sigma * sigma))
	k /= k.sum()
	return k


def blur_and_threshold(mask_f: np.ndarray, sigma: float, thr: float = 0.5) -> np.ndarray:
	"""Blur a float mask [0,1] and threshold to binary."""
	if sigma <= 0:
		return (mask_f >= thr).astype(np.uint8)
	radius = max(1, int(3 * sigma))
	k = gaussian_kernel1d(sigma, radius)
	# separable convolution, reflect pad
	def conv1d_along_axis(a: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
		pad = len(k) // 2
		ap = np.pad(a, [(pad, pad) if ax == axis else (0, 0) for ax in range(a.ndim)], mode="reflect")
		# roll and weighted sum
		out = np.zeros_like(a, dtype=np.float64)
		for i, w in enumerate(k):
			shift = i - pad
			out += w * np.take(ap, indices=range(pad + shift, pad + shift + a.shape[axis]), axis=axis)
		return out
	b = conv1d_along_axis(mask_f, k, axis=1)
	b = conv1d_along_axis(b, k, axis=0)
	return (b >= thr).astype(np.uint8)


def draw_superellipse(height: int, width: int, center: Tuple[float, float], axes: Tuple[float, float], angle_rad: float, exponent: float) -> np.ndarray:
	"""Rasterize a rotated superellipse: (|x/a|)^n + (|y/b|)^n <= 1.
	Returns float mask in [0,1]."""
	y = np.arange(height, dtype=np.float64)
	x = np.arange(width, dtype=np.float64)
	xx, yy = np.meshgrid(x, y)
	cx, cy = center
	a, b = axes
	cos_t = math.cos(angle_rad)
	sin_t = math.sin(angle_rad)
	xr = (xx - cx) * cos_t + (yy - cy) * sin_t
	yr = -(xx - cx) * sin_t + (yy - cy) * cos_t
	val = (np.abs(xr) / max(a, 1e-6)) ** exponent + (np.abs(yr) / max(b, 1e-6)) ** exponent
	inside = (val <= 1.0).astype(np.float64)
	return inside


def generate_mask(params: GeneratorParams, rng: random.Random) -> np.ndarray:
	h, w = params.height, params.width
	comp_count = max(1, sample_from_hist(params.num_components_hist))
	canvas = np.zeros((h, w), dtype=np.float64)
	for _ in range(comp_count):
		# Sample component area as fraction of image
		area_frac = max(1e-4, sample_between_quantiles(params.component_area_quantiles)) * params.area_scale
		target_area = area_frac * (h * w)
		# Sample aspect ratio for superellipse major/minor axes
		aspect = max(0.2, sample_between_quantiles(params.bbox_aspect_quantiles))
		# Solve for axes a,b of superellipse approximating target_area: A ~ 4ab * Gamma(1+1/n)^2 / Gamma(1+2/n)
		n = rng.uniform(1.8, 3.5)  # shape exponent, 2 -> ellipse; >2 more squarish
		# approximate ellipse area for simplicity (n=2): A = pi * a * b
		A = max(target_area, 1.0)
		b = math.sqrt(A / (math.pi * aspect))
		a = aspect * b
		# Random orientation
		angle = rng.uniform(0, math.pi)
		# Random center avoiding overflow
		cx = rng.uniform(a + 2, w - a - 2) if w > 2 * a + 4 else rng.uniform(0, w - 1)
		cy = rng.uniform(b + 2, h - b - 2) if h > 2 * b + 4 else rng.uniform(0, h - 1)
		blob = draw_superellipse(h, w, (cx, cy), (a, b), angle, n)
		canvas = np.maximum(canvas, blob)
	# Optional holes
	if rng.random() < max(0.0, min(1.0, params.hole_probability)):
		n_holes = 1 if rng.random() < 0.8 else 2
		for _ in range(n_holes):
			# Subtract a smaller superellipse somewhere inside existing mask
			area_frac = max(5e-5, 0.5 * sample_between_quantiles(params.component_area_quantiles))
			target_area = area_frac * (h * w)
			aspect = max(0.3, sample_between_quantiles(params.bbox_aspect_quantiles))
			b = math.sqrt(target_area / (math.pi * aspect))
			a = aspect * b
			angle = rng.uniform(0, math.pi)
			cx = rng.uniform(0, w - 1)
			cy = rng.uniform(0, h - 1)
			hole = draw_superellipse(h, w, (cx, cy), (a, b), angle, rng.uniform(2.0, 4.0))
			canvas = np.clip(canvas - hole, 0.0, 1.0)
	# Smooth and binarize
	bin_mask = blur_and_threshold(canvas, sigma=params.smooth_sigma, thr=0.5)
	return bin_mask.astype(np.uint8)


# -----------------------------
# CLI
# -----------------------------


def analyze_dir(src_dir: Path) -> Tuple[List[MaskFeatures], AggregateStats]:
	paths = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
	features: List[MaskFeatures] = []
	for p in paths:
		mask = load_mask(p)
		features.append(compute_features(mask, p.name))
	agg = aggregate_stats(features)
	return features, agg


def save_json(obj, path: Path) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)


def cmd_analyze(args: argparse.Namespace) -> None:
	src = Path(args.src)
	features, agg = analyze_dir(src)
	out = Path(args.out)
	out.parent.mkdir(parents=True, exist_ok=True)
	save_json([asdict(f) for f in features], out.with_suffix(".per_image.json"))
	save_json(asdict(agg), out)
	print(f"Analyzed {len(features)} masks. Stats saved to {out} and per-image to {out.with_suffix('.per_image.json')}")


def cmd_generate(args: argparse.Namespace) -> None:
	# Load params from stats JSON or default
	if args.params and Path(args.params).exists():
		with open(args.params, "r", encoding="utf-8") as f:
			agg_data = json.load(f)
			stats = AggregateStats(**agg_data)
			params = GeneratorParams.from_aggregate(stats)
	else:
		# Fallback default size
		params = GeneratorParams(
			height=args.height,
			width=args.width,
			num_components_hist={"1": 1, "2": 1, "3": 1},
			component_area_quantiles={"q05": 0.002, "q25": 0.004, "q50": 0.008, "q75": 0.015, "q95": 0.03},
			bbox_aspect_quantiles={"q05": 0.5, "q25": 0.8, "q50": 1.2, "q75": 1.8, "q95": 2.5},
			hole_probability=0.2,
			smooth_sigma=0.6,
			area_scale=1.0,
		)
	if args.height and args.width:
		params.height = args.height
		params.width = args.width
	# Apply generator knobs
	params.area_scale = args.area_scale if hasattr(args, "area_scale") and args.area_scale is not None else params.area_scale
	if hasattr(args, "hole_prob") and args.hole_prob is not None:
		params.hole_probability = float(args.hole_prob)
	outdir = Path(args.outdir)
	outdir.mkdir(parents=True, exist_ok=True)
	rng = random.Random(args.seed)
	for i in range(args.n):
		mask = generate_mask(params, rng)
		img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
		img.save(outdir / f"gen_{i:03d}.png")
	print(f"Generated {args.n} masks to {outdir}")


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Analyze masks and generate similar random masks")
	sub = p.add_subparsers(dest="cmd", required=True)
	pa = sub.add_parser("analyze", help="Analyze a directory of masks")
	pa.add_argument("--src", type=str, required=True, help="Directory containing masks")
	pa.add_argument("--out", type=str, required=True, help="Output JSON path for aggregate stats")
	pa.set_defaults(func=cmd_analyze)
	pg = sub.add_parser("generate", help="Generate random masks")
	pg.add_argument("--params", type=str, default="", help="Path to aggregate stats JSON (from analyze)")
	pg.add_argument("--outdir", type=str, required=True, help="Output directory for generated masks")
	pg.add_argument("-n", type=int, default=16, help="Number of masks to generate")
	pg.add_argument("--height", type=int, default=512, help="Mask height if no params provided")
	pg.add_argument("--width", type=int, default=512, help="Mask width if no params provided")
	pg.add_argument("--seed", type=int, default=42, help="Random seed")
	pg.add_argument("--area-scale", type=float, default=1.0, help="Multiply component area fractions to control coverage")
	pg.add_argument("--hole-prob", type=float, default=None, help="Override hole probability [0,1]")
	pg.set_defaults(func=cmd_generate)
	return p


def main(argv: List[str]) -> None:
	parser = build_parser()
	args = parser.parse_args(argv)
	args.func(args)


if __name__ == "__main__":
	main(sys.argv[1:]) 