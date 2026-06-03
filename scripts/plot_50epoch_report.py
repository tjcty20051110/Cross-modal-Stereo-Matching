#!/usr/bin/env python3
"""Generate training-result plots for the 50-epoch report.

Produces:
  - resnet_loss_curve.png : Siamese ResNet18 triplet loss vs. epoch/batch
  - cfm_loss_curve.png    : CFM end-to-end NLL+L1 loss vs. epoch/batch
  - metrics_comparison.png: Census vs Siamese vs CFM bar chart (MSE / EPE / D1-all)
  - per_sample_epe.png    : per-sample EPE comparison line plot
  - qualitative_*.png     : 4-panel qualitative figures for selected samples
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
    p.add_argument("--project-root",
                   default="/home/cty/work/cv/project5/cross-modal-stereo-matching")
    p.add_argument("--out-dir",
                   default="/home/cty/work/cv/project5/report_assets_50epoch")
    return p.parse_args()


def parse_train_log(log_path: Path):
    """Extract per-epoch average loss and per-batch loss trajectory."""
    epoch_losses = []
    batch_losses = []
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


def plot_loss_curve(epoch_losses, batch_losses, out_path: Path, title_prefix: str,
                    loss_label: str = "triplet margin loss"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if batch_losses:
        xs = [b[0] for b in batch_losses]
        ys = [b[1] for b in batch_losses]
        axes[0].plot(xs, ys, linewidth=0.6, alpha=0.55, color="tab:blue", label="batch loss")
        if len(ys) >= 20:
            window = max(10, min(50, len(ys) // 20))
            smoothed = np.convolve(ys, np.ones(window) / window, mode="valid")
            axes[0].plot(range(window, window + len(smoothed)), smoothed,
                         linewidth=1.6, color="tab:red",
                         label=f"moving avg (w={window})")
        axes[0].set_xlabel("global batch index")
        axes[0].set_ylabel(loss_label)
        axes[0].set_title(f"{title_prefix} — loss per batch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    if epoch_losses:
        xs = [e[0] for e in epoch_losses]
        ys = [e[1] for e in epoch_losses]
        axes[1].plot(xs, ys, marker="o", color="tab:green", markersize=4)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel(f"avg {loss_label}")
        axes[1].set_title(f"{title_prefix} — loss per epoch")
        axes[1].grid(True, alpha=0.3)
        # annotate first and last
        if len(ys) >= 2:
            axes[1].annotate(f"{ys[0]:.3f}", (xs[0], ys[0]),
                             textcoords="offset points", xytext=(6, 6), fontsize=9)
            axes[1].annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]),
                             textcoords="offset points", xytext=(-32, 6), fontsize=9,
                             color="tab:red")

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
    per_sample = [r for r in rows if r.get("sample_id", "").lower() != "average"]
    agg = None
    for r in rows:
        if r.get("sample_id", "").lower() == "average":
            agg = r
            break
    if agg is None and per_sample:
        def mean_key(k):
            vals = [float(r[k]) for r in per_sample if r.get(k) and r[k] != "nan"]
            return sum(vals) / len(vals) if vals else float("nan")
        agg = {"mse": mean_key("mse"),
               "epe": mean_key("epe"),
               "d1_all": mean_key("d1_all")}
    return {"per_sample": per_sample, "average": agg}


def plot_metrics_comparison(methods, out_path: Path):
    """methods: list of (name, metrics_dict) where metrics_dict has average {mse,epe,d1_all}."""
    labels = ["MSE (px²)", "EPE (px)", "D1-all (%)"]
    keys = ["mse", "epe", "d1_all"]
    n_methods = len(methods)
    x = np.arange(len(labels))
    width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    vals_per_method = []
    for i, (name, mdict) in enumerate(methods):
        vals = []
        for k in keys:
            if mdict and mdict.get("average"):
                vals.append(float(mdict["average"][k]))
            else:
                vals.append(float("nan"))
        vals_per_method.append(vals)
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=name,
                      color=colors[i % len(colors)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("metric value (lower is better)")
    ax.set_title("NIR(L)+RGB(R) cross-modal validation — method comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_yscale("log")  # values span multiple magnitudes
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_sample_epe(methods, out_path: Path):
    """Line plot comparing per-sample EPE across methods."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    sample_ids = None
    for i, (name, mdict) in enumerate(methods):
        if not mdict or not mdict.get("per_sample"):
            continue
        rows = mdict["per_sample"]
        xs = list(range(len(rows)))
        ys = [float(r["epe"]) for r in rows]
        ax.plot(xs, ys, marker="o", label=name,
                color=colors[i % len(colors)], markersize=5)
        if sample_ids is None:
            sample_ids = [r["sample_id"] for r in rows]

    if sample_ids:
        step = max(1, len(sample_ids) // 10)
        ax.set_xticks(list(range(0, len(sample_ids), step)))
        ax.set_xticklabels([sample_ids[i] for i in range(0, len(sample_ids), step)],
                           rotation=45, fontsize=8)
    ax.set_xlabel("sample id")
    ax.set_ylabel("EPE (px)")
    ax.set_title("per-sample EPE — NIR+RGB cross-modal validation")
    ax.legend()
    ax.grid(True, alpha=0.3)
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
        disp = np.where(depth_m > 0,
                        focal * baseline / np.maximum(depth_m, 1e-3),
                        0.0)
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
            entries.append({
                "left_path": resolve(row["left_path"]),
                "right_path": resolve(row["right_path"]),
                "gt_path": resolve(row["gt_disparity_path"]),
                "calib_path": resolve(row["calib_path"]),
                "sample_id": f"{Path(row['left_path']).stem}_{i}",
            })
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


def make_qualitative_figure(entry, preds_by_name, gt, left_img, max_disp,
                            out_path: Path):
    """preds_by_name: dict name -> disparity (or None)."""
    n_methods = len(preds_by_name)
    # 2x(1+ceil(n/2)) layout: first row shows left image + GT; rest show preds
    n_cols = 2
    n_rows = 1 + int(np.ceil((n_methods) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    axes = np.atleast_2d(axes)

    axes[0, 0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Left image (NIR)  [{entry['sample_id']}]")
    axes[0, 0].axis("off")

    if gt is not None:
        axes[0, 1].imshow(cv2.cvtColor(disp_to_color(gt, max_disp),
                                       cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Ground truth disparity")
    else:
        axes[0, 1].text(0.5, 0.5, "GT unavailable", ha="center", va="center")
    axes[0, 1].axis("off")

    # Layout remaining preds
    idx = 0
    for r in range(1, n_rows):
        for c in range(n_cols):
            if idx >= n_methods:
                axes[r, c].axis("off")
                continue
            name = list(preds_by_name.keys())[idx]
            pred = preds_by_name[name]
            if pred is not None:
                axes[r, c].imshow(cv2.cvtColor(disp_to_color(pred, max_disp),
                                               cv2.COLOR_BGR2RGB))
                axes[r, c].set_title(name)
            else:
                axes[r, c].text(0.5, 0.5, f"{name} missing",
                                ha="center", va="center")
            axes[r, c].axis("off")
            idx += 1

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    root = Path(args.project_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loss curves
    resnet_log = root / "logs" / "train_cross_gpu.log"
    cfm_log = root / "logs" / "train_cfm.log"

    resnet_epoch, resnet_batch = parse_train_log(resnet_log)
    cfm_epoch, cfm_batch = parse_train_log(cfm_log)

    plot_loss_curve(resnet_epoch, resnet_batch,
                    out_dir / "resnet_loss_curve.png",
                    "Siamese ResNet18 (NIR+RGB triplet)",
                    loss_label="triplet margin loss")
    print(f"[plot] resnet loss curve -> {out_dir / 'resnet_loss_curve.png'}")

    plot_loss_curve(cfm_epoch, cfm_batch,
                    out_dir / "cfm_loss_curve.png",
                    "CFM Pipeline (NIR+RGB end-to-end)",
                    loss_label="L1 + 0.5·L_cfm + 0.25·L_mdp")
    print(f"[plot] cfm loss curve -> {out_dir / 'cfm_loss_curve.png'}")

    # 2. Combined loss curve (normalized per model)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if resnet_epoch:
        x1 = [e[0] for e in resnet_epoch]
        y1 = [e[1] for e in resnet_epoch]
        axes[0].plot(x1, y1, marker="o", color="tab:blue", markersize=4,
                     label="Siamese ResNet18 (triplet)")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("triplet margin loss")
        axes[0].set_title("Siamese ResNet18 per-epoch loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    if cfm_epoch:
        x2 = [e[0] for e in cfm_epoch]
        y2 = [e[1] for e in cfm_epoch]
        axes[1].plot(x2, y2, marker="s", color="tab:orange", markersize=4,
                     label="CFM (L1 + NLL)")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("composite loss")
        axes[1].set_title("CFM Pipeline per-epoch loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "combined_loss_curve.png", dpi=120)
    plt.close(fig)
    print(f"[plot] combined loss -> {out_dir / 'combined_loss_curve.png'}")

    # 3. Metrics comparison (cross-modal)
    census = parse_metrics_csv(root / "output" / "cross_val_census" / "metrics.csv")
    siamese = parse_metrics_csv(root / "output" / "cross_val_dl" / "metrics.csv")
    cfm = parse_metrics_csv(root / "output" / "cfm_val" / "metrics.csv")

    methods = [
        ("Census + SGM", census),
        ("Siamese ResNet18 + SGM", siamese),
        ("CFM Pipeline", cfm),
    ]
    plot_metrics_comparison(methods, out_dir / "metrics_comparison.png")
    print(f"[plot] metrics comparison -> {out_dir / 'metrics_comparison.png'}")

    # Per-sample EPE (only use shared sample range)
    # Siamese & Census use 10 samples, CFM uses 20; trim to common 10
    trimmed = []
    for name, m in methods:
        if m and m.get("per_sample"):
            ps = m["per_sample"][:10]
            trimmed.append((name, {"per_sample": ps, "average": m["average"]}))
    plot_per_sample_epe(trimmed, out_dir / "per_sample_epe.png")
    print(f"[plot] per-sample epe -> {out_dir / 'per_sample_epe.png'}")

    # Print aggregates
    for name, m in methods:
        if m and m["average"]:
            a = m["average"]
            print(f"  {name:35s}  MSE={float(a['mse']):.2f}  "
                  f"EPE={float(a['epe']):.2f}  D1={float(a['d1_all']):.2f}%")

    # 4. Qualitative figures (first 5 samples)
    manifest_path = root / "data" / "MS2" / "manifest_nir_rgb_val.csv"
    if manifest_path.exists():
        entries = parse_manifest(manifest_path)[:5]
        for entry in entries:
            focal, baseline = parse_calib(entry["calib_path"])
            gt = load_ground_truth(entry, focal, baseline)
            census_pred = load_disparity(root / "output" / "cross_val_census",
                                         entry["sample_id"])
            siam_pred = load_disparity(root / "output" / "cross_val_dl",
                                       entry["sample_id"])
            cfm_pred = load_disparity(root / "output" / "cfm_val",
                                      entry["sample_id"])
            left_img = cv2.imread(entry["left_path"])
            if left_img is None:
                continue
            max_disp = 128.0
            if gt is not None and gt.max() > 0:
                max_disp = max(64.0, float(gt.max()) * 1.1)
            preds = {
                "Census + SGM": census_pred,
                "Siamese ResNet18 + SGM": siam_pred,
                "CFM Pipeline": cfm_pred,
            }
            out_path = out_dir / f"qualitative_{entry['sample_id']}.png"
            make_qualitative_figure(entry, preds, gt, left_img, max_disp, out_path)
            print(f"[plot] qualitative -> {out_path}")


if __name__ == "__main__":
    main()
