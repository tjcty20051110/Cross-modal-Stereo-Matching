#!/usr/bin/env python3
"""
Generate manifest CSV for the MS2 dataset.

Supported stereo pair modes:
  rgb      — RGB_left  + RGB_right   (same-modal,   baseline ~299 mm)
  nir_rgb  — NIR_left  + RGB_right   (cross-modal,  baseline ~353 mm) ← RECOMMENDED cross-modal
  rgb_nir  — RGB_left  + NIR_right   (cross-modal,  baseline ~3.6 mm) ← too small, not useful
  cross    — RGB_left  + NIR_left    (cross-modal,  baseline ~54 mm)

Usage:
    python3 scripts/generate_manifest.py [--mode rgb|nir_rgb|rgb_nir|cross] [--max-samples N]
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MS2 dataset manifest")
    parser.add_argument("--data-root", type=str, default="data/MS2")
    parser.add_argument("--output", type=str, default="data/MS2/manifest.csv")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rgb", "nir_rgb", "rgb_nir", "cross"],
        default="rgb",
        help=(
            "Stereo pair mode:\n"
            "  rgb     : RGB_left  + RGB_right  (same-modal,  ~299 mm)\n"
            "  nir_rgb : NIR_left  + RGB_right  (cross-modal, ~353 mm) [recommended]\n"
            "  rgb_nir : RGB_left  + NIR_right  (cross-modal, ~3.6 mm) [too small]\n"
            "  cross   : RGB_left  + NIR_left   (cross-modal, ~54 mm)"
        ),
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    return parser.parse_args()


def extract_calibration(data_root: Path, mode: str) -> dict:
    """Compute focal length, baseline and reference intrinsic for each mode."""
    calib_npy = data_root / "sync_data" / "calib.npy"
    if not calib_npy.exists():
        print(f"ERROR: {calib_npy} not found", file=sys.stderr)
        sys.exit(1)

    c = np.load(str(calib_npy), allow_pickle=True).item()
    R_nir2rgb = c["R_nir2rgb"]
    T_nir2rgb = c["T_nir2rgb"].flatten()   # NIR_left origin in RGB_left frame (mm)
    T_nirR    = c["T_nirR"].flatten()       # NIR_right in NIR_left frame (mm)
    T_rgbR    = c["T_rgbR"].flatten()       # RGB_right in RGB_left frame (mm)

    if mode == "rgb":
        # Reference: RGB_left, matching: RGB_right
        focal     = float(c["K_rgbL"][0, 0])
        baseline  = abs(float(T_rgbR[0])) / 1000.0
        intrinsic = c["K_rgbL"]

    elif mode == "nir_rgb":
        # Reference: NIR_left, matching: RGB_right
        # RGB_right in NIR_left frame = R_nir2rgb^-1 * (T_rgbR - T_nir2rgb)
        R_inv = np.linalg.inv(R_nir2rgb)
        T_rgbR_in_nir = R_inv @ (T_rgbR - T_nir2rgb)
        focal     = float(c["K_nirL"][0, 0])
        baseline  = abs(float(T_rgbR_in_nir[0])) / 1000.0
        intrinsic = c["K_nirL"]

    elif mode == "rgb_nir":
        # Reference: RGB_left, matching: NIR_right
        # NIR_right in RGB_left frame = R_nir2rgb * T_nirR + T_nir2rgb
        T_nirR_in_rgb = R_nir2rgb @ T_nirR + T_nir2rgb
        focal     = float(c["K_rgbL"][0, 0])
        baseline  = abs(float(T_nirR_in_rgb[0])) / 1000.0
        intrinsic = c["K_rgbL"]

    else:  # cross: RGB_left + NIR_left
        focal     = float(c["K_rgbL"][0, 0])
        baseline  = abs(float(T_nir2rgb[0])) / 1000.0
        intrinsic = c["K_rgbL"]

    return {
        "focal_length": focal,
        "baseline": baseline,
        "intrinsic": intrinsic.flatten().tolist(),
    }


def write_calib_yaml(calib_params: dict, output_path: Path):
    with open(output_path, "w") as f:
        f.write(f"focal_length: {calib_params['focal_length']}\n")
        f.write(f"baseline: {calib_params['baseline']}\n")
        f.write("intrinsic:\n")
        for v in calib_params["intrinsic"]:
            f.write(f"  - {v}\n")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    sync_data = data_root / "sync_data"
    proj_depth = data_root / "proj_depth"

    # Determine left / right image directories and GT depth directory
    if args.mode == "rgb":
        left_dir  = sync_data / "rgb" / "img_left"
        right_dir = sync_data / "rgb" / "img_right"
        gt_dir    = proj_depth / "rgb" / "depth_filtered"

    elif args.mode == "nir_rgb":
        # Reference camera is NIR_left → GT depth in NIR viewpoint
        left_dir  = sync_data / "nir" / "img_left"
        right_dir = sync_data / "rgb" / "img_right"
        gt_dir    = proj_depth / "nir" / "depth_filtered"

    elif args.mode == "rgb_nir":
        left_dir  = sync_data / "rgb" / "img_left"
        right_dir = sync_data / "nir" / "img_right"
        gt_dir    = proj_depth / "rgb" / "depth_filtered"

    else:  # cross
        left_dir  = sync_data / "rgb" / "img_left"
        right_dir = sync_data / "nir" / "img_left"
        gt_dir    = proj_depth / "rgb" / "depth_filtered"

    for d, name in [(left_dir, "left"), (right_dir, "right"), (gt_dir, "GT depth")]:
        if not d.exists():
            print(f"ERROR: {name} directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    left_files  = sorted(left_dir.glob("*.png"))
    right_files = sorted(right_dir.glob("*.png"))
    gt_files    = sorted(gt_dir.glob("*.png"))

    print(f"Found {len(left_files)} left, {len(right_files)} right, {len(gt_files)} GT")

    left_stems  = {f.stem: f for f in left_files}
    right_stems = {f.stem: f for f in right_files}
    gt_stems    = {f.stem: f for f in gt_files}

    common_stems = sorted(
        set(left_stems) & set(right_stems) & set(gt_stems)
    )
    print(f"Matched {len(common_stems)} complete samples")

    if args.max_samples > 0:
        common_stems = common_stems[: args.max_samples]
        print(f"Limited to {len(common_stems)} samples")

    calib_params   = extract_calibration(data_root, args.mode)
    calib_yaml_path = data_root / f"calib_{args.mode}.yaml"
    write_calib_yaml(calib_params, calib_yaml_path)
    print(f"Calibration -> {calib_yaml_path}")
    print(f"  focal_length : {calib_params['focal_length']:.4f} px")
    print(f"  baseline     : {calib_params['baseline']*1000:.2f} mm  ({calib_params['baseline']:.6f} m)")

    n_train = int(len(common_stems) * args.train_ratio)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_path.parent

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["left_path", "right_path", "gt_disparity_path", "calib_path", "split"])
        for i, stem in enumerate(common_stems):
            split = "train" if i < n_train else "val"
            writer.writerow([
                os.path.relpath(left_stems[stem],  manifest_dir),
                os.path.relpath(right_stems[stem], manifest_dir),
                os.path.relpath(gt_stems[stem],    manifest_dir),
                os.path.relpath(calib_yaml_path,   manifest_dir),
                split,
            ])

    print(f"Manifest -> {output_path}  (train={n_train}, val={len(common_stems)-n_train})")


if __name__ == "__main__":
    main()
