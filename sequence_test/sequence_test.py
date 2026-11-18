"""HeLa sequence-order experiment using the original three metric protocols.

IoU uses the IAIoU evaluator path from oaims_test_iou2.py, Dice uses IADisc,
and NoC uses IANoc.
This file intentionally preserves those separate protocols for comparison with
the existing paper tables.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "20261106")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import label

from model import initialize_model
from pi_seg.inference.oaims_adaptor_KD_EWC_IADisc import OAIMSAdaptor as DiceAdaptor
from pi_seg.inference.oaims_adaptor_KD_EWC_IANoc import OAIMSAdaptor as NocAdaptor
from pi_seg.inference.oaims_adaptor_KD_EWC_IAIoU import OAIMSAdaptor as IoUAdaptor


IMAGE_DIR = ROOT / "pi_seg/data/datasets/Skeletal_muscle_test/domain_2_cells/test/input"
MASK_DIR = ROOT / "pi_seg/data/datasets/Skeletal_muscle_test/domain_2_cells/test/target"
MODEL_PATH = ROOT / "save/skeletal_muscle_run_cells/model_latest.pth"
OUTPUT_ROOT = ROOT / "顺序实验/results"

MAX_CLICKS = 15
TARGET_CLICKS = [1, 3, 5, 10, 15]
TARGET_IOUS = [0.85, 0.88, 0.90]
POINT_SLOTS = 10
DEFAULT_SEED = 20261106
DEFAULT_NUM_RUNS = 5


def set_reproducible(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)


def natural_sort_key(value):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"([0-9]+)", str(value))]


def collect_images():
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        paths.extend(IMAGE_DIR.glob(ext))
    paths = sorted(set(paths), key=natural_sort_key)
    if not paths:
        raise FileNotFoundError(f"No images found: {IMAGE_DIR}")
    return paths


def mask_path(image_path):
    masks = sorted(MASK_DIR.glob(f"{image_path.stem}.*"), key=natural_sort_key)
    if not masks:
        raise FileNotFoundError(f"No mask for {image_path.name}: {MASK_DIR}")
    return masks[0]


def load_image(path, device):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    tensor = torch.tensor(image, dtype=torch.float32, device=device)[None, None] / 255.0
    return image, tensor


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    return (mask > 127).astype(np.uint8)


def empty_points(device):
    return torch.full((1, POINT_SLOTS * 2, 3), -1.0, device=device)


def append_click(points, click):
    y, x, positive = click
    result = points.clone()
    slots = result.shape[1] // 2
    start = 0 if positive == 1 else slots
    for index in range(start, start + slots):
        if result[0, index, 0] == -1:
            result[0, index, 0] = y
            result[0, index, 1] = x
            result[0, index, 2] = 1.0
            break
    return result


class RobotClicker:
    def __init__(self, gt):
        self.gt = gt.astype(np.uint8)
        self.clicked = np.zeros_like(self.gt, dtype=np.uint8)

    def next_click(self, pred):
        error = (pred.astype(np.uint8) != self.gt).astype(np.uint8)
        error[self.clicked == 1] = 0
        components, count = label(error)
        if count == 0:
            return None
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        component_id = int(np.argmax(sizes))
        if sizes[component_id] < 5:
            return None
        component = (components == component_id).astype(np.uint8)
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        distance[self.clicked == 1] = 0
        if distance.max() < 1:
            return None
        y, x = np.unravel_index(np.argmax(distance), distance.shape)
        self.clicked[y, x] = 1
        return int(y), int(x), int(self.gt[y, x])


def iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection / union) if union else 1.0


def dice(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    denominator = pred.sum() + gt.sum()
    return float(2.0 * intersection / denominator) if denominator else 1.0


def predict(adaptor, image_tensor, points):
    with torch.no_grad():
        output = adaptor.model(image_tensor, points, training=False)
        logits = output["instances"] if isinstance(output, dict) else output
        return (torch.sigmoid(logits) > 0.5).cpu().numpy()[0, 0].astype(np.uint8)


def load_model(device):
    checkpoint = torch.load(str(MODEL_PATH), map_location=device)
    model = initialize_model()
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def run_iou(order_paths, mode_name, device, seed, refresh_after_pi=False):
    """Run the same IoU protocol as oaims_test_iou2.py.

    The adaptor is created once per order and is kept across images so that
    online updates are transferred along the selected sequence order.
    """
    set_reproducible(seed)
    model = load_model(device)
    adaptor = IoUAdaptor(model)
    adaptor.reset_weights()
    rows = []

    for position, image_path in enumerate(order_paths, 1):
        mp = mask_path(image_path)
        _, image_tensor = load_image(image_path, device)
        gt = load_mask(mp)

        points = empty_points(device)
        clicker = RobotClicker(gt)
        clicks = []
        pred = predict(adaptor, image_tensor, points)
        current_iou = iou(pred, gt)
        records = {}

        for click_count in range(1, MAX_CLICKS + 1):
            click = clicker.next_click(pred)
            if click is not None:
                clicks.append(click)
                points = append_click(points, click)
                if mode_name == "OAIS-Net":
                    adaptor.adapt_MI(image_tensor, points, clicks)
                pred = predict(adaptor, image_tensor, points)
                current_iou = iou(pred, gt)

            if click_count in TARGET_CLICKS:
                records[click_count] = current_iou

            # Match oaims_test_iou2.py: trigger PI after the current
            # prediction reaches the threshold and keep the updated model
            # state for the next image in the sequence.
            if mode_name == "OAIS-Net" and current_iou >= 0.65:
                adaptor.adapt_PI(image_tensor, points, gt)
                if refresh_after_pi:
                    pred = predict(adaptor, image_tensor, points)
                    current_iou = iou(pred, gt)

        for count in TARGET_CLICKS:
            records.setdefault(count, current_iou)

        row = {
            "position": position,
            "image": image_path.name,
            "final_iou": current_iou,
            **{f"iou_at_{count}": records[count] for count in TARGET_CLICKS},
        }
        rows.append(row)

    del adaptor, model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def run_dice(order_paths, mode_name, device, seed, refresh_after_pi=False):
    set_reproducible(seed)
    model = load_model(device)
    adaptor = DiceAdaptor(model)
    adaptor.reset_weights()
    rows = []
    for position, image_path in enumerate(order_paths, 1):
        mp = mask_path(image_path)
        _, image_tensor = load_image(image_path, device)
        gt = load_mask(mp)
        points = empty_points(device)
        clicker = RobotClicker(gt)
        clicks = []
        pred = predict(adaptor, image_tensor, points)
        current_dice = dice(pred, gt)
        records = {}
        for click_count in range(1, MAX_CLICKS + 1):
            click = clicker.next_click(pred)
            if click is not None:
                clicks.append(click)
                points = append_click(points, click)
                if mode_name == "OAIS-Net":
                    adaptor.adapt_MI(image_tensor, points, clicks)
                pred = predict(adaptor, image_tensor, points)
                current_dice = dice(pred, gt)
            if click_count in TARGET_CLICKS:
                records[click_count] = current_dice
            # This placement matches oaims_test_Disc.py.
            if mode_name == "OAIS-Net" and current_dice >= 0.65:
                adaptor.adapt_PI(image_tensor, points, gt, max_clicks=10)
                if refresh_after_pi:
                    pred = predict(adaptor, image_tensor, points)
                    current_dice = dice(pred, gt)
        for count in TARGET_CLICKS:
            records.setdefault(count, current_dice)
        rows.append({"position": position, "image": image_path.name,
                     **{f"dice_at_{k}": records[k] for k in TARGET_CLICKS},
                     "final_dice": current_dice})
    del adaptor, model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def run_noc(order_paths, mode_name, target_iou, device, seed, refresh_after_pi=False):
    set_reproducible(seed)
    model = load_model(device)
    adaptor = NocAdaptor(model)
    adaptor.reset_weights()
    rows = []
    for position, image_path in enumerate(order_paths, 1):
        mp = mask_path(image_path)
        _, image_tensor = load_image(image_path, device)
        gt = load_mask(mp)
        points = empty_points(device)
        clicker = RobotClicker(gt)
        clicks = []
        pred = predict(adaptor, image_tensor, points)
        current_iou = iou(pred, gt)
        clicks_used = 0
        while current_iou < target_iou and clicks_used < MAX_CLICKS:
            click = clicker.next_click(pred)
            if click is None:
                break
            clicks.append(click)
            points = append_click(points, click)
            clicks_used += 1
            if mode_name == "OAIS-Net":
                adaptor.adapt_MI(image_tensor, points, clicks)
            pred = predict(adaptor, image_tensor, points)
            current_iou = iou(pred, gt)
            if mode_name == "OAIS-Net" and current_iou >= 0.65:
                adaptor.adapt_PI(image_tensor, points, gt, max_clicks=10)
                if refresh_after_pi:
                    pred = predict(adaptor, image_tensor, points)
                    current_iou = iou(pred, gt)
        rows.append({"position": position, "image": image_path.name,
                     "target_iou": target_iou,
                     "noc": clicks_used if current_iou >= target_iou else MAX_CLICKS,
                     "final_iou": current_iou})
    del adaptor, model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(order_name, random_seed, iou_rows, dice_rows, noc_rows):
    rows = []
    for mode in ("Base", "OAIS-Net"):
        iou_rows_m = [r for r in iou_rows if r["mode"] == mode]
        dice_rows_m = [r for r in dice_rows if r["mode"] == mode]
        noc_rows_m = [r for r in noc_rows if r["mode"] == mode]
        out = {"order": order_name, "random_seed": random_seed, "mode": mode,
               "num_images": len(iou_rows_m)}
        for count in TARGET_CLICKS:
            m, s = mean_std([r[f"iou_at_{count}"] for r in iou_rows_m])
            out[f"iou_at_{count}_mean"], out[f"iou_at_{count}_std"] = m, s
            m, s = mean_std([r[f"dice_at_{count}"] for r in dice_rows_m])
            out[f"dice_at_{count}_mean"], out[f"dice_at_{count}_std"] = m, s
        m, s = mean_std([r["final_iou"] for r in iou_rows_m])
        out["final_iou_mean"], out["final_iou_std"] = m, s
        m, s = mean_std([r["final_dice"] for r in dice_rows_m])
        out["final_dice_mean"], out["final_dice_std"] = m, s
        for target in TARGET_IOUS:
            values = [r["noc"] for r in noc_rows_m if r["target_iou"] == target]
            m, s = mean_std(values)
            out[f"noc_at_{target:.2f}_mean"], out[f"noc_at_{target:.2f}_std"] = m, s
        rows.append(out)
    return rows


def plot_order(order_name, iou_rows, noc_rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=220)
    colors = {"Base": "#d62728", "OAIS-Net": "#1f77b4"}
    for mode in ("Base", "OAIS-Net"):
        a = sorted([r for r in iou_rows if r["mode"] == mode], key=lambda r: r["position"])
        b = sorted([r for r in noc_rows if r["mode"] == mode and r["target_iou"] == 0.90], key=lambda r: r["position"])
        axes[0].plot([r["position"] for r in a], [r["iou_at_10"] * 100 for r in a],
                     marker="s" if mode == "Base" else "o", linestyle="--" if mode == "Base" else "-",
                     color=colors[mode], linewidth=2.2, label=mode)
        axes[1].plot([r["position"] for r in b], [r["noc"] for r in b],
                     marker="s" if mode == "Base" else "o", linestyle="--" if mode == "Base" else "-",
                     color=colors[mode], linewidth=2.2, label=mode)
    axes[0].set_title(f"{order_name}: IoU@10", fontweight="bold")
    axes[0].set_ylabel("IoU (%)")
    axes[1].set_title(f"{order_name}: NoC@IoU 0.90", fontweight="bold")
    axes[1].set_ylabel("Number of clicks")
    for ax in axes:
        ax.set_xlabel("Image Sequence Index")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[1].set_ylim(0, MAX_CLICKS + 1)
    fig.tight_layout()
    fig.savefig(out_dir / f"{order_name}_iou_noc_curve.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--random-seeds", default=str(DEFAULT_SEED))
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument(
        "--protocol",
        choices=("table1", "refreshed"),
        default="table1",
        help=(
            "table1 reproduces the original oaims_test protocols; "
            "refreshed recomputes the prediction after every PI update "
            "for all orders and all metrics."
        ),
    )
    args = parser.parse_args()
    refresh_after_pi = args.protocol == "refreshed"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random_seeds = [int(x.strip()) for x in args.random_seeds.split(",") if x.strip()]
    paths = collect_images()
    output_dir = Path(args.output_root) / (
        f"HeLa_table12_sequence_{args.protocol}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; Images: {len(paths)}; Model: {MODEL_PATH}")
    print(f"Protocol: {args.protocol}; Refresh after PI: {refresh_after_pi}")
    print(f"Fixed clicks: {TARGET_CLICKS}; NoC thresholds: {TARGET_IOUS}; Max clicks: {MAX_CLICKS}")

    summary = []
    orders = [("original", "", list(paths)), ("reverse", "", list(reversed(paths)))]
    for seed in random_seeds:
        shuffled = list(paths)
        random.Random(seed).shuffle(shuffled)
        orders.append((f"random_{seed}", str(seed), shuffled))

    for order_name, random_seed, order_paths in orders:
        print(f"\n===== {order_name} =====")
        print("Sequence:", " -> ".join(p.stem for p in order_paths))
        order_dir = output_dir / order_name
        order_dir.mkdir(parents=True, exist_ok=True)
        iou_rows, dice_rows, noc_rows = [], [], []
        for mode in ("Base", "OAIS-Net"):
            run_count = 1 if mode == "Base" else args.num_runs
            for run_index in range(run_count):
                seed = DEFAULT_SEED + run_index + (int(random_seed) if random_seed else 0)
                print(f"[{order_name}] {mode} run {run_index + 1}/{run_count}: IoU")
                rows = run_iou(order_paths, mode, device, seed, refresh_after_pi)
                for row in rows:
                    row.update({"order": order_name, "random_seed": random_seed, "mode": mode, "run": run_index + 1})
                iou_rows.extend(rows)
                print(f"[{order_name}] {mode} run {run_index + 1}/{run_count}: Dice")
                rows = run_dice(order_paths, mode, device, seed, refresh_after_pi)
                for row in rows:
                    row.update({"order": order_name, "random_seed": random_seed, "mode": mode, "run": run_index + 1})
                dice_rows.extend(rows)
                for target in TARGET_IOUS:
                    print(f"[{order_name}] {mode} run {run_index + 1}/{run_count}: NoC@{target:.2f}")
                    rows = run_noc(order_paths, mode, target, device, seed, refresh_after_pi)
                    for row in rows:
                        row.update({"order": order_name, "random_seed": random_seed, "mode": mode, "run": run_index + 1})
                    noc_rows.extend(rows)
        write_csv(order_dir / "per_image_iou.csv", iou_rows)
        write_csv(order_dir / "per_image_dice.csv", dice_rows)
        write_csv(order_dir / "per_image_noc.csv", noc_rows)
        summary.extend(summarize(order_name, random_seed, iou_rows, dice_rows, noc_rows))
        plot_order(order_name, iou_rows, noc_rows, order_dir)

    write_csv(output_dir / "sequence_order_table1_metrics.csv", summary)
    print(f"\n[DONE] Summary: {output_dir / 'sequence_order_table1_metrics.csv'}")


if __name__ == "__main__":
    main()
