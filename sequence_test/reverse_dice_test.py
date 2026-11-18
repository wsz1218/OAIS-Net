"""Evaluate sequence orders with the oaims_test_iou_last_all.py protocol.

IoU and Dice at each click are calculated from the same prediction mask.
"""

from __future__ import annotations

import csv
import argparse
import os
import random
import re
import sys
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

from oaims_test_iou_last_all import TABLE3_OAIS_CONFIG, make_table3_args
from oaisnet_ablation_sta_ltm1 import run_one_image
from pi_seg.inference.oaims_adaptor_ablation import OAIMSAdaptor
from model import initialize_model


# Same HeLa sequence paths used by the sequence-order experiment.
IMAGE_DIR = ROOT / "pi_seg/data/datasets/Skeletal_muscle_test/domain_2_cells/test/input"
MASK_DIR = ROOT / "pi_seg/data/datasets/Skeletal_muscle_test/domain_2_cells/test/target"
MODEL_PATH = ROOT / "save/skeletal_muscle_run_cells/model_latest.pth"
OUTPUT_ROOT = ROOT / "顺序实验/results"

MAX_CLICKS = 15
TARGET_CLICKS = (1, 3, 5, 10, 15)
POINT_SLOTS = 10
NUM_RUNS = 5
SEED = 20261106
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
        [p for p in IMAGE_DIR.iterdir() if p.is_file()],
        key=natural_key,
    )
    if not images:
        raise FileNotFoundError(f"No images found: {IMAGE_DIR}")
    return images


def build_orders(images, order_names, random_seed):
    orders = []
    for order_name in order_names:
        if order_name == "original":
            ordered = list(images)
            seed_text = ""
        elif order_name == "reverse":
            ordered = list(reversed(images))
            seed_text = ""
        elif order_name == "random":
            ordered = list(images)
            random.Random(random_seed).shuffle(ordered)
            seed_text = str(random_seed)
            order_name = f"random_{random_seed}"
        else:
            raise ValueError(f"Unsupported order: {order_name}")
        orders.append((order_name, seed_text, ordered))
    return orders


def get_mask(image_path: Path) -> Path:
    candidates = sorted(MASK_DIR.glob(f"{image_path.stem}.*"), key=natural_key)
    if not candidates:
        raise FileNotFoundError(f"No mask found for {image_path.name}: {MASK_DIR}")
    return candidates[0]


def load_image(image_path: Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    tensor = torch.from_numpy(image).float().to(DEVICE)[None, None] / 255.0
    return image, tensor


def load_mask(mask_path: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
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


def dice(pred, target):
    intersection = np.logical_and(pred, target).sum()
    denominator = pred.sum() + target.sum()
    return float(2.0 * intersection / denominator) if denominator else 1.0


def iou(pred, target):
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


def evaluate_run(adaptor, image_paths, run_index, order_name):
    rows = []
    update_times = []
    config = {
        "target_clicks": list(TARGET_CLICKS),
        "target_ious": [0.85, 0.88, 0.90],
        "max_clicks": MAX_CLICKS,
    }
    args = make_table3_args(config)
    for position, image_path in enumerate(image_paths, 1):
        result = run_one_image(
            adaptor,
            image_path,
            get_mask(image_path),
            TABLE3_OAIS_CONFIG,
            args,
            DEVICE.type,
        )
        row = {
            "run": run_index,
            "position": position,
            "image": image_path.name,
            **{f"iou_at_{click}": result["fixed_clicks"][click]["iou"] for click in TARGET_CLICKS},
            **{f"dice_at_{click}": result["fixed_clicks"][click]["dice"] for click in TARGET_CLICKS},
            "final_iou": result["final_iou"],
            "final_dice": result["final_dice"],
            "clicks_used": result["num_clicks_generated"],
            "order": order_name,
        }
        for click in TARGET_CLICKS:
            score = row[f"iou_at_{click}"]
            expected_dice = 2.0 * score / (1.0 + score)
            if not np.isclose(row[f"dice_at_{click}"], expected_dice, atol=1e-6):
                raise ValueError(f"IoU/Dice mismatch: {order_name} {image_path.name} @{click}")
        rows.append(row)
        update_times.append(result["click_latency_ms"])
        print(
            f"[{run_index}] {position:02d}/{len(image_paths)} {image_path.name}: "
            f"Dice@1={row['dice_at_1']:.4f}, Dice@5={row['dice_at_5']:.4f}, "
            f"Dice@15={row['dice_at_15']:.4f}, "
            f"IoU@1={row['iou_at_1']:.4f}, IoU@5={row['iou_at_5']:.4f}, "
            f"IoU@15={row['iou_at_15']:.4f}"
        )

    return rows, float(np.mean(update_times)) if update_times else 0.0


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


def compare_reference_iou(reference_root, order_name, rows):
    reference_file = reference_root / order_name / "per_image_iou.csv"
    if not reference_file.is_file():
        raise FileNotFoundError(f"Reference IoU CSV not found: {reference_file}")
    with reference_file.open(newline="", encoding="utf-8-sig") as handle:
        references = list(csv.DictReader(handle))
    indexed = {(int(r["run"]), r["image"]): r for r in references if r["mode"] == "OAIS-Net"}
    comparisons = []
    for row in rows:
        key = (row["run"], row["image"])
        if key not in indexed:
            raise ValueError(f"Missing reference IoU row: {order_name} run={key[0]} {key[1]}")
        reference = indexed[key]
        for click in TARGET_CLICKS:
            old_key = f"iou_at_{click}" if click != 15 else "final_iou"
            if old_key not in reference:
                raise ValueError(f"Missing reference field {old_key}: {reference_file}")
            comparisons.append({
                "order": order_name,
                "run": row["run"],
                "image": row["image"],
                "click": click,
                "new_iou": row[f"iou_at_{click}"],
                "reference_iou": float(reference[old_key]),
                "absolute_difference": abs(row[f"iou_at_{click}"] - float(reference[old_key])),
            })
    return comparisons


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aligned Dice and IoU sequence-order evaluation with the "
            "oaims_test_iou_last_all.py protocol. "
            "Both metrics use the same prediction masks."
        )
    )
    parser.add_argument(
        "--orders",
        default="original,reverse,random",
        help="Comma-separated orders: original, reverse, random, or all.",
    )
    parser.add_argument("--random-seed", type=int, default=SEED)
    parser.add_argument("--num-runs", type=int, default=NUM_RUNS)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument(
        "--reference-root",
        type=Path,
        help="Optional old sequence result directory for per-image IoU comparison.",
    )
    args = parser.parse_args()

    set_reproducible(SEED)
    images = collect_images()
    requested_orders = [x.strip().lower() for x in args.orders.split(",") if x.strip()]
    if requested_orders == ["all"]:
        requested_orders = ["original", "reverse", "random"]
    orders = build_orders(images, requested_orders, args.random_seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"HeLa_sequence_metrics_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Images: {len(images)}")
    print(f"Orders: {', '.join(order_name for order_name, _, _ in orders)}")
    print(f"Random seed: {args.random_seed}")
    print(f"Runs per order: {args.num_runs}")
    print("Adaptor: oaims_adaptor_ablation (oaims_test_iou_last_all.py protocol)")
    print("Metrics: aligned Dice and IoU from the same prediction masks")
    print(f"Output: {output_dir}")

    table_rows = []
    for order_name, random_seed, image_paths in orders:
        print(f"\n===== {order_name} =====")
        print("Sequence:", " -> ".join(path.stem for path in image_paths))
        order_dir = output_dir / order_name
        order_dir.mkdir(parents=True, exist_ok=True)
        all_rows = []
        latencies = []

        for run_index in range(1, args.num_runs + 1):
            set_reproducible(SEED)
            model = load_model()
            adaptor = OAIMSAdaptor(model)
            adaptor.reset_weights()
            rows, latency = evaluate_run(adaptor, image_paths, run_index, order_name)
            all_rows.extend(rows)
            latencies.append(latency)
            del adaptor, model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        write_csv(order_dir / "per_image_metrics.csv", all_rows)
        # Keep a Dice-named copy for compatibility with the previous output
        # layout; its rows now also contain the aligned IoU columns.
        write_csv(order_dir / "per_image_dice.csv", all_rows)

        if args.reference_root is not None:
            comparison = compare_reference_iou(args.reference_root, order_name, all_rows)
            write_csv(order_dir / "reference_iou_comparison.csv", comparison)
            maximum = max(r["absolute_difference"] for r in comparison)
            print(f"{order_name} maximum IoU difference from old result: {maximum:.6f}")
            if maximum > 1e-6:
                print("WARNING: IoU differs from the old run; do not pair this Dice with the old IoU table.")

        # The fixed-click IoU and Dice values come from the same masks.
        table_row = {
            "order": order_name,
            "random_seed": random_seed,
            "mode": "OAIS-Net",
            "num_images": len(image_paths),
        }
        for click_count in TARGET_CLICKS:
            mean, std = mean_std(all_rows, f"dice_at_{click_count}")
            table_row[f"dice_at_{click_count}_mean"] = mean
            table_row[f"dice_at_{click_count}_std"] = std
            mean, std = mean_std(all_rows, f"iou_at_{click_count}")
            table_row[f"iou_at_{click_count}_mean"] = mean
            table_row[f"iou_at_{click_count}_std"] = std
        final_mean, final_std = mean_std(all_rows, "final_dice")
        table_row["final_dice_mean"] = final_mean
        table_row["final_dice_std"] = final_std
        final_mean, final_std = mean_std(all_rows, "final_iou")
        table_row["final_iou_mean"] = final_mean
        table_row["final_iou_std"] = final_std
        table_rows.append(table_row)

        print(f"{order_name} Dice summary")
        for click_count in TARGET_CLICKS:
            mean, std = mean_std(all_rows, f"dice_at_{click_count}")
            print(f"@{click_count:>2} clicks: Dice={mean:.4f} +/- {std:.4f}")
        print(f"{order_name} IoU summary (aligned with the Dice predictions)")
        for click_count in TARGET_CLICKS:
            mean, std = mean_std(all_rows, f"iou_at_{click_count}")
            print(f"@{click_count:>2} clicks: IoU={mean:.4f} +/- {std:.4f}")
        print(f"Mean STA update time: {np.mean(latencies):.2f} ms")

    write_csv(output_dir / "sequence_order_metrics_summary.csv", table_rows)
    # Also save metric-specific summaries with the same values for convenient
    # downstream table generation.
    dice_rows = []
    iou_rows = []
    for row in table_rows:
        dice_row = {
            key: value for key, value in row.items()
            if key.startswith("dice_") or key in {"order", "random_seed", "mode", "num_images", "final_dice_mean", "final_dice_std"}
        }
        iou_row = {
            key: value for key, value in row.items()
            if key.startswith("iou_") or key in {"order", "random_seed", "mode", "num_images", "final_iou_mean", "final_iou_std"}
        }
        dice_rows.append(dice_row)
        iou_rows.append(iou_row)
    write_csv(output_dir / "sequence_order_dice_summary.csv", dice_rows)
    write_csv(output_dir / "sequence_order_iou_summary.csv", iou_rows)
    print(f"\nSaved aligned summary: {output_dir / 'sequence_order_metrics_summary.csv'}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
