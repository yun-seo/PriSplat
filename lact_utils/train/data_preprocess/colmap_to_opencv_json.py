#!/usr/bin/env python3
import os
import sys
import struct
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Reference for COLMAP binary format: https://colmap.github.io/format.html

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = qvec
    return np.array([
        [1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],
        [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q0 * q1],
        [2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1 * q1 - 2 * q2 * q2],
    ])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + format_char_sequence, data)


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id = read_next_bytes(f, 4, "i")[0]
            model_id = read_next_bytes(f, 4, "i")[0]
            width = read_next_bytes(f, 8, "Q")[0]
            height = read_next_bytes(f, 8, "Q")[0]
            num_params = {
                0: 3,   # SIMPLE_PINHOLE: f, cx, cy
                1: 4,   # PINHOLE: fx, fy, cx, cy
                2: 4,   # SIMPLE_RADIAL: f, cx, cy, k
                3: 5,   # RADIAL: f, cx, cy, k1, k2
                4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
                5: 12,  # FULL_OPENCV
                6: 5,   # FOV: fx, fy, cx, cy, omega
                7: 3,   # SIMPLE_RADIAL_FISHEYE: f, cx, cy
                8: 4,   # RADIAL_FISHEYE: f, cx, cy, k
                9: 12,  # THIN_PRISM_FISHEYE
            }.get(model_id, None)
            if num_params is None:
                raise ValueError(f"Unsupported camera model id {model_id}")
            params = list(read_next_bytes(f, 8 * num_params, "d" * num_params))
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_reg_images = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_reg_images):
            image_id = read_next_bytes(f, 4, "i")[0]
            qvec = np.array(read_next_bytes(f, 8 * 4, "dddd"))
            tvec = np.array(read_next_bytes(f, 8 * 3, "ddd"))
            camera_id = read_next_bytes(f, 4, "i")[0]
            # Read null-terminated name
            name_bytes = bytearray()
            while True:
                ch = f.read(1)
                if ch == b"\x00" or ch == b"":
                    break
                name_bytes.extend(ch)
            name = name_bytes.decode("utf-8")
            # Skip 2D-3D correspondences for speed
            num_points2D = read_next_bytes(f, 8, "Q")[0]
            _ = f.read(num_points2D * (8 * 2 + 8))
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
    return images


def intrinsics_from_colmap(camera_model_id: int, params: list):
    # Return fx, fy, cx, cy for common models
    if camera_model_id == 0:  # SIMPLE_PINHOLE: [f, cx, cy]
        f, cx, cy = params[0], params[1], params[2]
        return f, f, cx, cy
    if camera_model_id == 1:  # PINHOLE: [fx, fy, cx, cy]
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return fx, fy, cx, cy
    if camera_model_id in (2, 3):  # SIMPLE_RADIAL or RADIAL: take f/cx/cy
        f, cx, cy = params[0], params[1], params[2]
        return f, f, cx, cy
    if camera_model_id == 4:  # OPENCV: [fx, fy, cx, cy, k1, k2, p1, p2]
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return fx, fy, cx, cy
    if camera_model_id == 5:  # FULL_OPENCV: first 4 are fx, fy, cx, cy
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return fx, fy, cx, cy
    if camera_model_id in (6, 7, 8, 9):
        # FOV / fisheye variants: first 4 params when available or fallback
        if len(params) >= 4:
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            return fx, fy, cx, cy
        elif len(params) >= 3:
            f, cx, cy = params[0], params[1], params[2]
            return f, f, cx, cy
    raise ValueError(f"Unsupported or unexpected camera model {camera_model_id} with params {params}")


def build_w2c_from_qt(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    # COLMAP stores world-to-camera as R, t such that x_cam = R * x_world + t
    R = qvec2rotmat(qvec)
    t = tvec.reshape(3, 1)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3:4] = t
    return w2c


def main():
    if len(sys.argv) != 2:
        print("Usage: colmap_to_opencv_json.py /abs/path/to/robustnerf/scene")
        sys.exit(1)
    scene_dir = Path(sys.argv[1])
    sparse_dir = scene_dir / "sparse" / "0"
    images_dir = scene_dir / "images"
    cameras_bin = sparse_dir / "cameras.bin"
    images_bin = sparse_dir / "images.bin"
    if not cameras_bin.exists() or not images_bin.exists():
        print(f"Missing COLMAP binaries in {sparse_dir}")
        sys.exit(2)

    cameras = read_cameras_binary(str(cameras_bin))
    images = read_images_binary(str(images_bin))

    # Cache image sizes to fix any zero H/W from binary parsing
    size_cache = {}
    for p in images_dir.glob("*.*"):
        try:
            with Image.open(p) as im:
                w, h = im.size
            size_cache[p.name] = (w, h)
        except Exception:
            pass

    results = []
    for img_id in sorted(images.keys()):
        im = images[img_id]
        cam = cameras[im["camera_id"]]
        fx, fy, cx, cy = intrinsics_from_colmap(cam["model_id"], cam["params"])
        w2c = build_w2c_from_qt(im["qvec"], im["tvec"]).tolist()
        name = im["name"]
        stem = Path(name).stem
        # Prefer PNG in images/
        if (images_dir / f"{stem}.png").exists():
            img_name = f"{stem}.png"
        else:
            img_name = Path(name).name
        rel_path = f"images/{img_name}"
        # Target width/height from actual image if available; otherwise camera record
        tgt_w, tgt_h = size_cache.get(img_name, (int(cam["width"]), int(cam["height"])) )
        src_w, src_h = int(cam["width"]), int(cam["height"]) 
        if tgt_w and tgt_h and (tgt_w != src_w or tgt_h != src_h):
            sx = tgt_w / float(src_w)
            sy = tgt_h / float(src_h)
            fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
            width, height = tgt_w, tgt_h
        else:
            width, height = src_w, src_h
        results.append({
            "w": int(width),
            "h": int(height),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "w2c": w2c,
            "file_path": rel_path,
        })

    out_file = scene_dir / "opencv_cameras.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Wrote {out_file} with {len(results)} entries")


if __name__ == "__main__":
    main() 