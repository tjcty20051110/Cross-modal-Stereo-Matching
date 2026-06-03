#!/usr/bin/env python3
"""Generate comparison plots for Census vs DL stereo matching results.

Produces:
  - loss_curve.png        : training loss vs. epoch
  - metrics_comparison.png: Census vs DL bar chart (MSE / EPE / D1-all)
  - qualitative_*.png     : side-by-side visualization per sample
"""

import argparse
import csv
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-log", default="logs/train_resnet.log")
    p.add_argument("--census-csv", default="output/val_census/metrics.csv")
    p.add_argument("--dl-csv", default="output/val_dl/metrics.csv")
    p.add_argument("--manifest", default="data/MS2/manifest_val.csv")
    p.add_argument("--census-pred-dir", default="output/val_census")
    p.add_argument("--dl-pred-dir", default="output/val_dl")
    p.add_argument("--out-dir", default="output/plots")
    return p.parse_args()


def parse_train_log(log_path: Path):
    """Extract per-epoch average loss and per-batch loss trajectory."""
    epoch_losses = []         # [(epoch, avg_loss)]
    batch_losses = []         # [(global_batch_idx, loss, epoch)]
    global_idx = 0
    current_epoch = 0
    pattern_batch = re.compile(r"\[epoch (\d+)/\d+\] batch (\d+) loss=([\d.eE+\-]+)")
    pattern_avg = re.compile(r"\[epoch (\d+)\] batches=\d+ avg_loss=([\d.eE+\-]+)")
    if not log_path.exists():
        return epoch_losses, batch_losses
    with open(log_path) as f:
        for line in f:
            m = pattern_batch.search(line)
            if m:
                current_epoch = int(m.group(1))
                global_idx += 1
                batch_losses.append((global_idx, float(m.group(3)), current_epoch))
                continue
            m = pattern_avg.search(line)
            if m:
                epoch_losses.append((int(m.group(1)), float(m.group(2))))
    return epoch_losses, batch_losses


def plot_loss_curve(epoch_losses, batch_losses, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if batch_losses:
        xs = [b[0] for b in batch_losses]
        ys = [b[1] for b in batch_losses]
        axes[0].plot(xs, ys, linewidth=0.6, alpha=0.6, color="tab:blue", label="batch loss")
        # smoothed curve
        if len(ys) >= 20:
            window = 20
            smoothed = np.convolve(ys, np.ones(window) / window, mode="valid")
            axes[0].plot(range(window, window + len(smoothed)), smoothed,
                         linewidth=1.5, color="tab:red", label=f"moving avg (w={window})")
        axes[0].set_xlabel("global batch index")
        axes[0].set_ylabel("triplet margin loss")
        axes[0].set_title("Training loss per batch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No batch data", ha="center", va="center")

    if epoch_losses:
        xs = [e[0] for e in epoch_losses]
        ys = [e[1] for e in epoch_losses]
        axes[1].plot(xs, ys, marker="o", color="tab:green")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("average loss")
        axes[1].set_title("Training loss per epoch")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No epoch data", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_metrics_csv(csv_path: Path):
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    # Separate aggregate row (sample_id == 'AVERAGE' or similar) from per-sample.
    per_sample = [r for r in rows if r.get("sample_id", "").lower() != "average"]
    agg = None
    for r in rows:
        if r.get("sample_id", "").lower() == "average":
            agg = r
            break
    if agg is None and per_sample:
        # compute mean from per-sample rows
        def mean_key(k):
            vals = [float(r[k]) for r in per_sample if r.get(k) and r[k] != "nan"]
            return sum(vals) / len(vals) if vals else float("nan")
        agg = {"mse": mean_key("mse"), "epe": mean_key("epe"), "d1_all": mean_key("d1_all")}
    return {"per_sample": per_sample, "average": agg}


def plot_metrics_comparison(census, dl, out_path: Path):
    labels = ["MSE", "EPE", "D1-all (%)"]
    census_vals = []
    dl_vals = []
    for key in ["mse", "epe", "d1_all"]:
        c = float(census["average"][key]) if census and census["average"] else float("nan")
        d = float(dl["average"][key]) if dl and dl["average"] else float("nan")
        census_vals.append(c)
        dl_vals.append(d)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, census_vals, width, label="Census + SGM", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, dl_vals, width, label="Siamese (ResNet18) + SGM", color="#DD8452")
    for bar, val in zip(list(bars1) + list(bars2), census_vals + dl_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("metric value (lower is better)")
    ax.set_title("Census vs DL feature on validation set")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def disp_to_color(disp: np.ndarray, max_disp: float = 128.0) -> np.ndarray:
    valid = disp > 0
    norm = np.clip(disp / max(max_disp, 1e-3), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = [0, 0, 0]
    return color


def load_disparity(pred_dir: Path, sample_id: str):
    path = pred_dir / f"{sample_id}_disp.png"
    if not path.exists():
        return None
    disp = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if disp is None:
        return None
    return disp.astype(np.float32) / 256.0


def load_ground_truth(manifest_row, focal, baseline):
    gt_path = manifest_row["gt_path"]
    depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    depth_m = depth.astype(np.float32) / 256.0
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.where(depth_m > 0, focal * baseline / np.maximum(depth_m, 1e-3), 0.0)
    disp[depth_m <= 0] = 0
    return disp


def parse_manifest(manifest_path: Path):
    entries = []
    manifest_dir = manifest_path.parent
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            def resolve(p):
                p = Path(p)
                return str((manifest_dir / p).resolve()) if not p.is_absolute() else str(p)
            entry = {
                "left_path": resolve(row["left_path"]),
                "right_path": resolve(row["right_path"]),
                "gt_path": resolve(row["gt_disparity_path"]),
                "calib_path": resolve(row["calib_path"]),
                "sample_id": f"{Path(row['left_path']).stem}_{i}",
            }
            entries.append(entry)
    return entries


def parse_calib(calib_path: str):
    focal = 1.0
    baseline = 1.0
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("focal_length:"):
                focal = float(line.split(":", 1)[1].strip())
            elif line.startswith("baseline:"):
                baseline = float(line.split(":", 1)[1].strip())
    return focal, baseline


def make_qualitative_figure(entry, census_pred, dl_pred, gt, left_img, max_disp, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    axes[0, 0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Left image  ({entry['sample_id']})")
    axes[0, 0].axis("off")

    if gt is not None:
        gt_color = cv2.cvtColor(disp_to_color(gt, max_disp), cv2.COLOR_BGR2RGB)
        axes[0, 1].imshow(gt_color)
        axes[0, 1].set_title("Ground truth disparity")
    else:
        axes[0, 1].text(0.5, 0.5, "GT unavailable", ha="center", va="center")
    axes[0, 1].axis("off")

    if census_pred is not None:
        c_color = cv2.cvtColor(disp_to_color(census_pred, max_disp), cv2.COLOR_BGR2RGB)
        axes[1, 0].imshow(c_color)
        axes[1, 0].set_title("Census + SGM")
    else:
        axes[1, 0].text(0.5, 0.5, "Census pred missing", ha="center", va="center")
    axes[1, 0].axis("off")

    if dl_pred is not None:
        d_color = cv2.cvtColor(disp_to_color(dl_pred, max_disp), cv2.COLOR_BGR2RGB)
        axes[1, 1].imshow(d_color)
        axes[1, 1].set_title("Siamese (ResNet18) + SGM")
    else:
        axes[1, 1].text(0.5, 0.5, "DL pred missing", ha="center", va="center")
    axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loss curve
    epoch_losses, batch_losses = parse_train_log(Path(args.train_log))
    plot_loss_curve(epoch_losses, batch_losses, out_dir / "loss_curve.png")
    print(f"[plot] loss curve -> {out_dir / 'loss_curve.png'}")

    # 2. Metrics comparison
    census = parse_metrics_csv(Path(args.census_csv))
    dl = parse_metrics_csv(Path(args.dl_csv))
    plot_metrics_comparison(census, dl, out_dir / "metrics_comparison.png")
    print(f"[plot] metrics bar chart -> {out_dir / 'metrics_comparison.png'}")

    if census and census["average"]:
        print("  Census avg:", census["average"])
    if dl and dl["average"]:
        print("  DL avg:    ", dl["average"])

    # 3. Qualitative per-sample figures (up to 5)
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        entries = parse_manifest(manifest_path)[:5]
        for entry in entries:
            focal, baseline = parse_calib(entry["calib_path"])
            gt = load_ground_truth(entry, focal, baseline)
            census_pred = load_disparity(Path(args.census_pred_dir), entry["sample_id"])
            dl_pred = load_disparity(Path(args.dl_pred_dir), entry["sample_id"])
            left_img = cv2.imread(entry["left_path"])
            if left_img is None:
                continue
            max_disp = 128.0
            if gt is not None and gt.max() > 0:
                max_disp = max(64.0, float(gt.max()) * 1.1)
            out_path = out_dir / f"qualitative_{entry['sample_id']}.png"
            make_qualitative_figure(entry, census_pred, dl_pred, gt, left_img, max_disp, out_path)
            print(f"[plot] qualitative -> {out_path}")


if __name__ == "__main__":
    main()
