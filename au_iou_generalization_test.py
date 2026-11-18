import argparse
import csv
import glob
import os
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import label

TARGET_CLICKS = [1, 3, 5, 10, 15]


def natural_sort_key(path):
    name = os.path.basename(path)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", name)]


def calculate_iou(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection / union) if union > 0 else 1.0


def load_model(model_path, device):
    from model import initialize_model

    print(f"[Load] model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model = initialize_model()
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Load] missing keys: {len(missing)}")
    if unexpected:
        print(f"[Load] unexpected keys: {len(unexpected)}")
    return model.to(device).eval()


class RobotClicker:
    def __init__(self, gt_mask):
        self.gt_mask = (gt_mask > 0).astype(np.uint8)
        self.clicked_mask = np.zeros_like(self.gt_mask, dtype=np.uint8)

    def get_next_click(self, pred_mask):
        pred_mask = (pred_mask > 0).astype(np.uint8)
        error_mask = (pred_mask != self.gt_mask).astype(np.uint8)
        error_mask[self.clicked_mask == 1] = 0

        labeled_array, num_features = label(error_mask)
        if num_features == 0:
            return None

        comp_sizes = np.bincount(labeled_array.ravel())
        comp_sizes[0] = 0
        max_id = int(np.argmax(comp_sizes))
        if comp_sizes[max_id] < 5:
            return None

        comp_mask = (labeled_array == max_id).astype(np.uint8)
        dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        dist[self.clicked_mask == 1] = 0
        if float(dist.max()) < 1.0:
            return None

        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        self.clicked_mask[y, x] = 1
        is_pos = 1 if self.gt_mask[y, x] == 1 else 0
        return int(y), int(x), is_pos


def append_click(points_tensor, click):
    y, x, is_pos = click
    points = points_tensor.clone()
    num_points = points.size(1) // 2
    offset = 0 if is_pos == 1 else num_points

    for i in range(offset, offset + num_points):
        if points[0, i, 0] == -1.0:
            points[0, i, 0] = y
            points[0, i, 1] = x
            points[0, i, 2] = 1.0
            break
    return points


def predict_mask(model, image_tensor, points_tensor):
    with torch.no_grad():
        model.eval()
        out = model(image_tensor, points_tensor, training=False)
        return (torch.sigmoid(out["instances"]) > 0.5).cpu().numpy()[0, 0].astype(np.uint8)


def find_mask(mask_dir, image_path):
    stem = Path(image_path).stem
    matches = glob.glob(str(Path(mask_dir) / f"{stem}.*"))
    if not matches:
        raise FileNotFoundError(f"No mask found for {image_path} under {mask_dir}")
    return matches[0]


def evaluate_one_image(
    adaptor,
    image_path,
    mask_path,
    device,
    max_clicks,
    use_online_learning,
    mi_update_below_iou,
    enable_pi,
    pi_iou_low,
    pi_iou_high,
):
    gt_mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    img_np = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_np is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    image_tensor = torch.tensor(img_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) / 255.0
    points_tensor = torch.full((1, 20, 3), -1.0, device=device)

    robot = RobotClicker(gt_mask)
    pred_mask = predict_mask(adaptor.model, image_tensor, points_tensor)
    current_iou = calculate_iou(pred_mask, gt_mask)
    records = {}
    latency = []
    pi_used = False
    mi_updates = 0

    for click_id in range(1, max_clicks + 1):
        click = robot.get_next_click(pred_mask)
        if click is not None:
            points_tensor = append_click(points_tensor, click)

            pred_mask = predict_mask(adaptor.model, image_tensor, points_tensor)
            current_iou = calculate_iou(pred_mask, gt_mask)

            should_update_mi = use_online_learning and current_iou < mi_update_below_iou
            if should_update_mi:
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
                adaptor.adapt_MI(image_tensor, points_tensor, [click])
                pred_mask = predict_mask(adaptor.model, image_tensor, points_tensor)
                current_iou = calculate_iou(pred_mask, gt_mask)
                if device == "cuda":
                    torch.cuda.synchronize()
                latency.append(time.time() - start)
                mi_updates += 1

            should_update_pi = (
                use_online_learning
                and enable_pi
                and not pi_used
                and pi_iou_low <= current_iou <= pi_iou_high
            )
            if should_update_pi:
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
                adaptor.adapt_PI(image_tensor, points_tensor, gt_mask, max_clicks=5)
                pred_mask = predict_mask(adaptor.model, image_tensor, points_tensor)
                current_iou = calculate_iou(pred_mask, gt_mask)
                if device == "cuda":
                    torch.cuda.synchronize()
                latency.append(time.time() - start)
                pi_used = True

        if click_id in TARGET_CLICKS:
            records[click_id] = current_iou

    return records, latency, mi_updates, int(pi_used)


def evaluate_dataset(
    adaptor,
    image_dir,
    mask_dir,
    device,
    max_clicks,
    use_online_learning,
    mi_update_below_iou,
    enable_pi,
    pi_iou_low,
    pi_iou_high,
    reset_per_image,
):
    image_paths = sorted(glob.glob(str(Path(image_dir) / "*.*")), key=natural_sort_key)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")

    if not reset_per_image:
        adaptor.reset_weights()

    per_click = {k: [] for k in TARGET_CLICKS}
    rows = []
    all_latency = []
    total_mi_updates = 0
    total_pi_updates = 0

    for img_idx, image_path in enumerate(image_paths):
        if reset_per_image:
            adaptor.reset_weights()

        mask_path = find_mask(mask_dir, image_path)
        records, latency, mi_updates, pi_updates = evaluate_one_image(
            adaptor=adaptor,
            image_path=image_path,
            mask_path=mask_path,
            device=device,
            max_clicks=max_clicks,
            use_online_learning=use_online_learning,
            mi_update_below_iou=mi_update_below_iou,
            enable_pi=enable_pi,
            pi_iou_low=pi_iou_low,
            pi_iou_high=pi_iou_high,
        )
        all_latency.extend(latency)
        total_mi_updates += mi_updates
        total_pi_updates += pi_updates

        for click_id, iou in records.items():
            per_click[click_id].append(iou)
            rows.append(
                {
                    "image": os.path.basename(image_path),
                    "clicks": click_id,
                    "iou": iou,
                    "mi_updates": mi_updates,
                    "pi_updates": pi_updates,
                }
            )

        print(
            f"[{img_idx + 1:03d}/{len(image_paths):03d}] {os.path.basename(image_path)} "
            f"IoU@15={records.get(15, np.nan):.4f}, MI={mi_updates}, PI={pi_updates}"
        )

    summary = {
        k: (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))
        for k, v in per_click.items()
    }
    avg_latency_ms = float(np.mean(all_latency) * 1000) if all_latency else 0.0
    return summary, rows, avg_latency_ms, total_mi_updates, total_pi_updates


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Au IoU-only Base vs guarded OAIMS evaluation.")
    parser.add_argument(
        "--image-dir",
        default="pi_seg/data/datasets/Au_generalization_330kx_884e_source/test_330kx_884e/input",
    )
    parser.add_argument(
        "--mask-dir",
        default="pi_seg/data/datasets/Au_generalization_330kx_884e_source/test_330kx_884e/target",
    )
    parser.add_argument("--model-path", default="save/skeletal_muscle_run_Au_884/model_latest.pth")
    parser.add_argument("--output-dir", default="test_result/au_iou_guarded")
    parser.add_argument("--max-clicks", type=int, default=15)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--mi-update-below-iou", type=float, default=0.85)
    parser.add_argument("--enable-pi", action="store_true", help="Enable conservative PI memory update.")
    parser.add_argument("--pi-iou-low", type=float, default=0.85)
    parser.add_argument("--pi-iou-high", type=float, default=0.93)
    parser.add_argument("--reset-per-image", action="store_true", help="Disable cross-image memory accumulation.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = Path(args.image_dir).parents[1].name
    run_dir = Path(args.output_dir) / f"{dataset_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_path, device)
    from pi_seg.inference.oaims_adaptor_KD_EWC_IAIoU import OAIMSAdaptor

    adaptor = OAIMSAdaptor(model)

    print("\n[Base] static interactive model")
    base_summary, base_rows, base_latency, _, _ = evaluate_dataset(
        adaptor=adaptor,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        device=device,
        max_clicks=args.max_clicks,
        use_online_learning=False,
        mi_update_below_iou=args.mi_update_below_iou,
        enable_pi=False,
        pi_iou_low=args.pi_iou_low,
        pi_iou_high=args.pi_iou_high,
        reset_per_image=args.reset_per_image,
    )

    write_csv(run_dir / "base_per_image_iou.csv", base_rows)

    oaims_summaries = []
    oaims_rows = []
    latencies = []
    mi_update_counts = []
    pi_update_counts = []

    for run_idx in range(args.num_runs):
        print(f"\n[OAIMS-lite] run {run_idx + 1}/{args.num_runs}")
        adaptor.reset_weights()
        summary, rows, latency, mi_updates, pi_updates = evaluate_dataset(
            adaptor=adaptor,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            device=device,
            max_clicks=args.max_clicks,
            use_online_learning=True,
            mi_update_below_iou=args.mi_update_below_iou,
            enable_pi=args.enable_pi,
            pi_iou_low=args.pi_iou_low,
            pi_iou_high=args.pi_iou_high,
            reset_per_image=args.reset_per_image,
        )
        oaims_summaries.append(summary)
        latencies.append(latency)
        mi_update_counts.append(mi_updates)
        pi_update_counts.append(pi_updates)
        for row in rows:
            row = deepcopy(row)
            row["run"] = run_idx + 1
            oaims_rows.append(row)

    write_csv(run_dir / "oaims_per_image_iou.csv", oaims_rows)

    print("\n" + "=" * 70)
    print("Au IoU Evaluation: Base vs Guarded OAIMS")
    print("=" * 70)
    print(f"Image dir : {args.image_dir}")
    print(f"Mask dir  : {args.mask_dir}")
    print(f"Model     : {args.model_path}")
    print(f"MI guard  : update only if IoU < {args.mi_update_below_iou:.2f}")
    print(f"PI update : {'enabled' if args.enable_pi else 'disabled'}")
    print("-" * 70)
    print("Clicks | Base IoU        | OAIMS IoU")
    print("-" * 70)

    summary_rows = []
    for click_id in TARGET_CLICKS:
        base_mean, base_std = base_summary[click_id]
        run_means = [summary[click_id][0] for summary in oaims_summaries]
        oaims_mean = float(np.mean(run_means))
        oaims_std = float(np.std(run_means))
        diff = oaims_mean - base_mean
        print(
            f"@{click_id:2d}    | {base_mean * 100:6.2f}% +/- {base_std * 100:5.2f}% "
            f"| {oaims_mean * 100:6.2f}% +/- {oaims_std * 100:5.2f}% "
            f"({diff * 100:+.2f}%)"
        )
        summary_rows.append(
            {
                "clicks": click_id,
                "base_mean_iou": base_mean,
                "base_std_iou": base_std,
                "oaims_mean_iou": oaims_mean,
                "oaims_std_across_runs": oaims_std,
                "diff_iou": diff,
            }
        )

    print("-" * 70)
    print(f"OAIMS avg latency: {np.mean(latencies):.2f} ms")
    print(f"OAIMS MI updates : {np.mean(mi_update_counts):.2f} per run")
    print(f"OAIMS PI updates : {np.mean(pi_update_counts):.2f} per run")
    print(f"Saved to         : {run_dir}")
    print("=" * 70)

    write_csv(run_dir / "summary_iou.csv", summary_rows)


if __name__ == "__main__":
    main()
