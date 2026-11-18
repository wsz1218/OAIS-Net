"""IoU-only sequence-order evaluation for the HeLa EM image sequence."""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "20261106")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from scipy.ndimage import label

from model import initialize_model
from pi_seg.inference.oaims_adaptor_KD_EWC_IAIoU import OAIMSAdaptor


# HeLa sequence paths used by the sequence-order experiment.
IMAGE_DIR = ROOT / "pi_seg/data/datasets/Skeletal_muscle_test/domain_2_cells/test/input"
MASK_DIR = ROOT / "pi_seg/data/datasets/Skeletal_muscle_test/domain_2_cells/test/target"
MODEL_PATH = ROOT / "save/skeletal_muscle_run_cells/model_latest.pth"
OUTPUT_ROOT = ROOT / "顺序实验/results"

MAX_CLICKS = 15
TARGET_CLICKS = (1, 3, 5, 10, 15)
POINT_SLOTS = 10
DEFAULT_NUM_RUNS = 5
DEFAULT_RANDOM_SEED = 20261106
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_reproducible(seed: int) -> None:
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


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"([0-9]+)", path.name)]


def collect_images():
    images = sorted(
        [path for path in IMAGE_DIR.iterdir() if path.is_file()],
        key=natural_key,
    )
    if not images:
        raise FileNotFoundError(f"No images found: {IMAGE_DIR}")
    return images


def build_orders(images, order_names, random_seed):
    orders = []
    for name in order_names:
        if name == "original":
            ordered = list(images)
            seed_text = ""
            order_name = "original"
        elif name == "reverse":
            ordered = list(reversed(images))
            seed_text = ""
            order_name = "reverse"
        elif name == "random":
            ordered = list(images)
            random.Random(random_seed).shuffle(ordered)
            seed_text = str(random_seed)
            order_name = f"random_{random_seed}"
        else:
            raise ValueError(f"Unsupported order: {name}")
        orders.append((order_name, seed_text, ordered))
    return orders


def mask_path(image_path: Path) -> Path:
    masks = sorted(MASK_DIR.glob(f"{image_path.stem}.*"), key=natural_key)
    if not masks:
        raise FileNotFoundError(f"No mask for {image_path.name}: {MASK_DIR}")
    return masks[0]


def load_image(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    tensor = torch.from_numpy(image).float().to(DEVICE)[None, None] / 255.0
    return image, tensor


def load_mask(mask_path_: Path):
    mask = cv2.imread(str(mask_path_), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path_}")
    return (mask > 127).astype(np.uint8)


def empty_points():
    return torch.full((1, POINT_SLOTS * 2, 3), -1.0, device=DEVICE)


def append_click(points, click):
    y, x, positive = click
    result = points.clone()
    slots = result.shape[1] // 2
    start = 0 if positive == 1 else slots
    for index in range(start, start + slots):
        if result[0, index, 0].item() < 0:
            result[0, index, 0] = y
            result[0, index, 1] = x
            result[0, index, 2] = 1.0
            break
    return result


def predict(adaptor, image_tensor, points):
    with torch.no_grad():
        output = adaptor.model(image_tensor, points, training=False)
        logits = output["instances"] if isinstance(output, dict) else output
        return (torch.sigmoid(logits) > 0.5).cpu().numpy()[0, 0].astype(np.uint8)


def calculate_iou(pred, target):
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(intersection / union) if union else 1.0


class RobotClicker:
    def __init__(self, target):
        self.target = target.astype(np.uint8)
        self.clicked = np.zeros_like(self.target, dtype=np.uint8)

    def next_click(self, pred):
        error = (pred != self.target).astype(np.uint8)
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
        return int(y), int(x), int(self.target[y, x])


def load_model():
    print(f"Loading model: {MODEL_PATH}")
    checkpoint = torch.load(str(MODEL_PATH), map_location=DEVICE)
    model = initialize_model()
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state = {key[7:] if key.startswith("module.") else key: value
             for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval()


def evaluate_run(adaptor, image_paths, mode_name, run_index, order_name):
    rows = []
    update_times = []

    for position, image_path in enumerate(image_paths, 1):
        _, image_tensor = load_image(image_path)
        target = load_mask(mask_path(image_path))
        points = empty_points()
        clicks = []
        clicker = RobotClicker(target)
        pred = predict(adaptor, image_tensor, points)
        initial_iou = calculate_iou(pred, target)
        current_iou = initial_iou
        records = {}

        for click_count in range(1, MAX_CLICKS + 1):
            click = clicker.next_click(pred)
            if click is not None:
                clicks.append(click)
                points = append_click(points, click)
                if mode_name == "OAIS-Net":
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    adaptor.adapt_MI(image_tensor, points, clicks)
                    pred = predict(adaptor, image_tensor, points)
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    update_times.append((time.perf_counter() - start) * 1000.0)
                else:
                    pred = predict(adaptor, image_tensor, points)

                current_iou = calculate_iou(pred, target)

            if click_count in TARGET_CLICKS:
                records[click_count] = current_iou

            # Match oaims_test_iou2.py: PI is triggered after the current
            # prediction reaches the threshold and the updated state is kept
            # for the next image in this order.
            if mode_name == "OAIS-Net" and current_iou >= 0.65:
                adaptor.adapt_PI(image_tensor, points, target)

        for click_count in TARGET_CLICKS:
            records.setdefault(click_count, current_iou)

        rows.append({
            "order": order_name,
            "run": run_index,
            "position": position,
            "image": image_path.name,
            "initial_iou": initial_iou,
            "iou_at_1": records[1],
            "iou_at_3": records[3],
            "iou_at_5": records[5],
            "iou_at_10": records[10],
            "iou_at_15": records[15],
            "final_iou": current_iou,
            "clicks_used": len(clicks),
            "mode": mode_name,
        })

        print(
            f"[{order_name}] {mode_name} run {run_index}, "
            f"{position:02d}/{len(image_paths)} {image_path.name}: "
            f"IoU@1={records[1]:.4f}, IoU@5={records[5]:.4f}, "
            f"IoU@15={records[15]:.4f}"
        )

    mean_latency = float(np.mean(update_times)) if update_times else 0.0
    return rows, mean_latency


def mean_std(rows, key):
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="HeLa sequence-order IoU evaluation")
    parser.add_argument(
        "--orders",
        default="original,reverse,random",
        help="Comma-separated orders: original, reverse, random, or all.",
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    set_reproducible(DEFAULT_RANDOM_SEED)
    images = collect_images()
    order_names = [item.strip().lower() for item in args.orders.split(",") if item.strip()]
    if order_names == ["all"]:
        order_names = ["original", "reverse", "random"]
    orders = build_orders(images, order_names, args.random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"HeLa_sequence_iou_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Images: {len(images)}")
    print(f"Orders: {', '.join(name for name, _, _ in orders)}")
    print(f"Adaptor: oaims_adaptor_KD_EWC_IAIoU")
    print(f"Output: {output_dir}")

    summary_rows = []
    for order_name, random_seed, image_paths in orders:
        order_dir = output_dir / order_name
        order_dir.mkdir(parents=True, exist_ok=True)
        all_rows = []

        for mode_name in ("Base", "OAIS-Net"):
            run_count = 1 if mode_name == "Base" else args.num_runs
            for run_index in range(1, run_count + 1):
                set_reproducible(DEFAULT_RANDOM_SEED + run_index)
                model = load_model()
                adaptor = OAIMSAdaptor(model)
                adaptor.reset_weights()
                rows, _ = evaluate_run(
                    adaptor, image_paths, mode_name, run_index, order_name
                )
                all_rows.extend(rows)
                del adaptor, model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

        write_csv(order_dir / "per_image_iou.csv", all_rows)

        for mode_name in ("Base", "OAIS-Net"):
            mode_rows = [row for row in all_rows if row["mode"] == mode_name]
            summary = {
                "order": order_name,
                "random_seed": random_seed,
                "mode": mode_name,
                "num_images": len(image_paths),
            }
            for click_count in TARGET_CLICKS:
                mean, std = mean_std(mode_rows, f"iou_at_{click_count}")
                summary[f"iou_at_{click_count}_mean"] = mean
                summary[f"iou_at_{click_count}_std"] = std
            mean, std = mean_std(mode_rows, "final_iou")
            summary["final_iou_mean"] = mean
            summary["final_iou_std"] = std
            summary_rows.append(summary)

    write_csv(output_dir / "sequence_order_iou_summary.csv", summary_rows)
    print(f"\nSaved summary: {output_dir / 'sequence_order_iou_summary.csv'}")


if __name__ == "__main__":
    main()
