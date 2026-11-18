import argparse
import csv
import glob
import importlib
import os
import re
import time
from datetime import datetime

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import cv2
import numpy as np
import torch
from scipy.ndimage import label


DEFAULT_IMAGE_DIR = "pi_seg/data/datasets/Skeletal_muscle/domain_5_Au/test/input"
DEFAULT_MASK_DIR = "pi_seg/data/datasets/Skeletal_muscle/domain_5_Au/test/target"
DEFAULT_MODEL_PATH = "save/skeletal_muscle_run_Au/model_best_iou.pth"
DEFAULT_DF_CCG_ADAPTOR = "pi_seg.inference.oaims_adaptor_original_df_ccg"
DEFAULT_OURS_ADAPTOR = "pi_seg.inference.oaims_adaptor_KD_EWC_IAIoU"


def parse_clicks(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def natural_sort_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def calculate_iou(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection) / float(union) if union > 0 else 1.0


class RobotClicker:
    def __init__(self, gt_mask):
        self.gt_mask = gt_mask.astype(np.uint8)
        self.clicked_mask = np.zeros_like(self.gt_mask, dtype=np.uint8)

    def get_next_click(self, pred_mask):
        error_mask = (pred_mask.astype(np.uint8) != self.gt_mask).astype(np.uint8)
        error_mask[self.clicked_mask == 1] = 0
        labeled_array, num_features = label(error_mask)
        if num_features == 0:
            return None

        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0
        component_id = int(np.argmax(component_sizes))
        if component_sizes[component_id] < 5:
            return None

        component_mask = (labeled_array == component_id).astype(np.uint8)
        dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
        dist[self.clicked_mask == 1] = 0
        if np.max(dist) < 1:
            return None

        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        self.clicked_mask[y, x] = 1
        is_pos = 1 if self.gt_mask[y, x] == 1 else 0
        return int(y), int(x), int(is_pos)


def append_click_to_tensor(points_tensor, click):
    y, x, is_pos = click
    new_points = points_tensor.clone()
    num_points = new_points.size(1) // 2
    offset = 0 if is_pos == 1 else num_points

    for i in range(offset, offset + num_points):
        if new_points[0, i, 0] == -1.0:
            new_points[0, i, 0] = y
            new_points[0, i, 1] = x
            new_points[0, i, 2] = 1.0
            break

    return new_points


def import_adaptor(module_name):
    module = importlib.import_module(module_name)
    return module.OAIMSAdaptor


def load_model(model_path, device):
    from model import initialize_model

    print(f"Loading model weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    clean_state = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    model = initialize_model()
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.load_state_dict(clean_state, strict=False)
    return model.to(device).eval()


def read_image_and_mask(image_path, mask_dir, device):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_candidates = glob.glob(os.path.join(mask_dir, base_name + ".*"))
    if not mask_candidates:
        return None, None, None

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_candidates[0], cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return None, None, None

    gt_mask = (mask > 127).astype(np.uint8)
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0
    return image, gt_mask, image_tensor


def forward_mask(model, image_tensor, points_tensor):
    with torch.no_grad():
        output = model(image_tensor, points_tensor, training=False)
        logits = output["instances"] if isinstance(output, dict) else output
        return (torch.sigmoid(logits) > 0.5).detach().cpu().numpy()[0, 0].astype(np.uint8)


def build_base_click_sequence(model, image_tensor, gt_mask, max_clicks, device):
    robot = RobotClicker(gt_mask)
    clicks = []
    points_tensor = torch.full((1, 20, 3), -1.0, device=device)
    pred_mask = forward_mask(model, image_tensor, points_tensor)

    for _ in range(max_clicks):
        click = robot.get_next_click(pred_mask)
        if click is None:
            break
        clicks.append(click)
        points_tensor = append_click_to_tensor(points_tensor, click)
        pred_mask = forward_mask(model, image_tensor, points_tensor)

    return clicks


def run_one_method(
    method_name,
    adaptor_cls,
    args,
    image_paths,
    target_clicks,
    device,
    run_index,
    use_online_learning,
    fixed_clicks_by_image,
    log_write,
):
    model = load_model(args.model_path, device)
    adaptor = adaptor_cls(model)
    adaptor.reset_weights()

    records = []
    latencies = []

    for image_index, image_path in enumerate(image_paths, start=1):
        if args.reset_each_image:
            adaptor.reset_weights()

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image, gt_mask, image_tensor = read_image_and_mask(image_path, args.mask_dir, device)
        if image is None:
            log_write(f"[WARN] Skip image without matched mask: {image_path}")
            continue

        robot = RobotClicker(gt_mask)
        all_clicks = []
        points_tensor = torch.full((1, 20, 3), -1.0, device=device)
        pred_mask = forward_mask(adaptor.model, image_tensor, points_tensor)
        current_iou = calculate_iou(pred_mask, gt_mask)
        pi_done = False
        fixed_clicks = fixed_clicks_by_image.get(base_name) if fixed_clicks_by_image is not None else None
        mi_updates = 0

        for click_id in range(1, args.max_clicks + 1):
            if fixed_clicks is not None:
                click = fixed_clicks[click_id - 1] if click_id <= len(fixed_clicks) else None
            else:
                click = robot.get_next_click(pred_mask)

            if click is not None:
                all_clicks.append(click)
                points_tensor = append_click_to_tensor(points_tensor, click)

                if device == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()

                should_adapt_mi = (
                    use_online_learning
                    and (args.adapt_max_clicks <= 0 or len(all_clicks) <= args.adapt_max_clicks)
                )
                if should_adapt_mi:
                    adapt_clicks = all_clicks
                    if args.adapt_recent_clicks > 0:
                        adapt_clicks = all_clicks[-args.adapt_recent_clicks:]
                    adaptor.adapt_MI(image_tensor, points_tensor, adapt_clicks)
                    mi_updates += 1
                pred_mask = forward_mask(adaptor.model, image_tensor, points_tensor)

                if device == "cuda":
                    torch.cuda.synchronize()
                latencies.append(time.time() - start_time)

                current_iou = calculate_iou(pred_mask, gt_mask)

                should_update_pi = (
                    use_online_learning
                    and args.pi_policy in {"threshold-every", "threshold-once"}
                    and current_iou >= args.pi_threshold
                    and (args.pi_policy == "threshold-every" or not pi_done)
                )
                if should_update_pi:
                    adaptor.adapt_PI(image_tensor, points_tensor, gt_mask, max_clicks=args.pi_max_clicks)
                    pi_done = True

            if click_id in target_clicks:
                records.append({
                    "Method": method_name,
                    "Run": run_index,
                    "Image": base_name,
                    "Clicks": click_id,
                    "IoU": current_iou,
                    "NumActualClicks": len(all_clicks),
                    "MIUpdates": mi_updates,
                })

        if (
            use_online_learning
            and args.pi_policy == "after-image"
            and all_clicks
            and current_iou >= args.pi_threshold
        ):
            adaptor.adapt_PI(image_tensor, points_tensor, gt_mask, max_clicks=args.pi_max_clicks)

        if image_index % args.log_every == 0 or image_index == len(image_paths):
            log_write(f"  [{method_name} Run {run_index}] {image_index}/{len(image_paths)} images done")

    avg_latency_ms = float(np.mean(latencies) * 1000.0) if latencies else 0.0
    return records, avg_latency_ms


def summarize(records, target_clicks):
    summary = {}
    for method in sorted({row["Method"] for row in records}):
        method_rows = [row for row in records if row["Method"] == method]
        summary[method] = {}
        for clicks in target_clicks:
            values = [row["IoU"] for row in method_rows if row["Clicks"] == clicks]
            summary[method][clicks] = {
                "mean": float(np.mean(values)) if values else float("nan"),
                "std": float(np.std(values)) if values else float("nan"),
                "n": len(values),
            }
    return summary


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_clicks(records, click_groups):
    rows = []
    for group_name, clicks in click_groups.items():
        for method in sorted({row["Method"] for row in records}):
            values = [
                row["IoU"]
                for row in records
                if row["Method"] == method and row["Clicks"] in clicks
            ]
            rows.append({
                "Group": group_name,
                "Method": method,
                "Clicks": ",".join(str(c) for c in clicks),
                "IoU_mean": float(np.mean(values)) if values else float("nan"),
                "IoU_std": float(np.std(values)) if values else float("nan"),
                "N": len(values),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Loss ablation: DF+CCG vs KD+Anchor+IA-CCG with IoU metrics.")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--mask-dir", default=DEFAULT_MASK_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--df-ccg-adaptor", default=DEFAULT_DF_CCG_ADAPTOR)
    parser.add_argument("--ours-adaptor", default=DEFAULT_OURS_ADAPTOR)
    parser.add_argument("--output-dir", default="test_result")
    parser.add_argument("--target-clicks", default="1,3,5,10,15")
    parser.add_argument(
        "--early-clicks",
        default="1,3,5",
        help="Click checkpoints used for the early-interaction aggregate metric.",
    )
    parser.add_argument("--max-clicks", type=int, default=15)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--pi-threshold", type=float, default=0.65)
    parser.add_argument("--pi-max-clicks", type=int, default=10)
    parser.add_argument(
        "--pi-policy",
        choices=["after-image", "threshold-once", "threshold-every", "off"],
        default="after-image",
        help=(
            "PI/LTM update policy. after-image updates once after all click metrics for an image are recorded; "
            "threshold-every matches the older oaims_test_iou2.py behavior."
        ),
    )
    parser.add_argument("--pi-once", action="store_true", help="Deprecated alias for --pi-policy threshold-once.")
    parser.add_argument("--disable-pi", action="store_true", help="Deprecated alias for --pi-policy off.")
    parser.add_argument("--no-base", action="store_true", help="Do not run the static base model baseline.")
    parser.add_argument(
        "--reset-each-image",
        action="store_true",
        help="Reset model weights before every image. Use this for within-image loss ablation.",
    )
    parser.add_argument(
        "--click-policy",
        choices=["adaptive", "base-fixed"],
        default="adaptive",
        help=(
            "adaptive lets each method receive clicks from its own current errors; "
            "base-fixed precomputes one static Base click sequence and reuses it for every method."
        ),
    )
    parser.add_argument(
        "--adapt-max-clicks",
        type=int,
        default=0,
        help="If > 0, run online MI/STA updates only for the first N actual clicks.",
    )
    parser.add_argument(
        "--adapt-recent-clicks",
        type=int,
        default=0,
        help="If > 0, each online MI/STA update uses only the most recent N clicks.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.disable_pi:
        args.pi_policy = "off"
    elif args.pi_once:
        args.pi_policy = "threshold-once"

    target_clicks = parse_clicks(args.target_clicks)
    early_clicks = parse_clicks(args.early_clicks)
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.*")), key=natural_sort_key)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.image_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(args.image_dir))) or "dataset"
    exp_dir = os.path.join(args.output_dir, f"loss_ablation_iou_{dataset_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, "test_log.txt")

    def log_write(text):
        print(text)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    log_write("=" * 80)
    log_write("Loss ablation with IoU")
    log_write(f"Device: {device}")
    log_write(f"Images: {args.image_dir}")
    log_write(f"Masks: {args.mask_dir}")
    log_write(f"Model: {args.model_path}")
    log_write(f"DF+CCG adaptor: {args.df_ccg_adaptor}")
    log_write(f"KD+Anchor+IA-CCG adaptor: {args.ours_adaptor}")
    log_write(f"Target clicks: {target_clicks}; max clicks: {args.max_clicks}; runs: {args.num_runs}")
    log_write(f"Early-click aggregate: {early_clicks}")
    log_write(f"PI policy: {args.pi_policy}; threshold: {args.pi_threshold}")
    log_write(f"Reset each image: {args.reset_each_image}")
    log_write(f"Click policy: {args.click_policy}")
    log_write(f"Adapt max clicks: {args.adapt_max_clicks}; adapt recent clicks: {args.adapt_recent_clicks}")
    log_write("=" * 80)

    fixed_clicks_by_image = None
    if args.click_policy == "base-fixed":
        fixed_clicks_by_image = {}
        log_write("\nPrecomputing fixed click sequences from the static Base model...")
        base_model_for_clicks = load_model(args.model_path, device)
        for image_index, image_path in enumerate(image_paths, start=1):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            _, gt_mask, image_tensor = read_image_and_mask(image_path, args.mask_dir, device)
            if image_tensor is None:
                continue
            fixed_clicks_by_image[base_name] = build_base_click_sequence(
                base_model_for_clicks,
                image_tensor,
                gt_mask,
                args.max_clicks,
                device,
            )
            if image_index % args.log_every == 0 or image_index == len(image_paths):
                log_write(f"  [Fixed clicks] {image_index}/{len(image_paths)} images done")
        del base_model_for_clicks
        if device == "cuda":
            torch.cuda.empty_cache()

    df_ccg_cls = import_adaptor(args.df_ccg_adaptor)
    ours_cls = import_adaptor(args.ours_adaptor)
    methods = []
    if not args.no_base:
        methods.append(("Base-no-online", ours_cls, False, 1))
    methods.extend([
        ("DF+CCG", df_ccg_cls, True, args.num_runs),
        ("KD+Anchor+IA-CCG", ours_cls, True, args.num_runs),
    ])

    all_records = []
    latency_rows = []
    for method_name, adaptor_cls, use_online_learning, method_runs in methods:
        for run_index in range(1, method_runs + 1):
            log_write(f"\nRunning {method_name} - run {run_index}/{method_runs}")
            records, latency_ms = run_one_method(
                method_name,
                adaptor_cls,
                args,
                image_paths,
                target_clicks,
                device,
                run_index,
                use_online_learning,
                fixed_clicks_by_image,
                log_write,
            )
            all_records.extend(records)
            latency_rows.append({"Method": method_name, "Run": run_index, "Latency_ms": latency_ms})
            log_write(f"  Average online response latency: {latency_ms:.2f} ms")

    summary = summarize(all_records, target_clicks)
    summary_rows = []
    log_write("\n" + "=" * 80)
    log_write("Final IoU summary")
    if "Base-no-online" in summary:
        log_write("Clicks | Base-no-online | DF+CCG mean+-std | KD+Anchor+IA-CCG mean+-std | Ours-DF")
    else:
        log_write("Clicks | DF+CCG mean+-std | KD+Anchor+IA-CCG mean+-std | Ours-DF")
    log_write("-" * 80)
    for clicks in target_clicks:
        df_stats = summary["DF+CCG"][clicks]
        ours_stats = summary["KD+Anchor+IA-CCG"][clicks]
        delta = ours_stats["mean"] - df_stats["mean"]
        if "Base-no-online" in summary:
            base_stats = summary["Base-no-online"][clicks]
            log_write(
                f"@{clicks:02d}    | "
                f"{base_stats['mean'] * 100:.2f}+-{base_stats['std'] * 100:.2f} | "
                f"{df_stats['mean'] * 100:.2f}+-{df_stats['std'] * 100:.2f} | "
                f"{ours_stats['mean'] * 100:.2f}+-{ours_stats['std'] * 100:.2f} | "
                f"{delta * 100:+.2f}"
            )
        else:
            log_write(
                f"@{clicks:02d}    | "
                f"{df_stats['mean'] * 100:.2f}+-{df_stats['std'] * 100:.2f} | "
                f"{ours_stats['mean'] * 100:.2f}+-{ours_stats['std'] * 100:.2f} | "
                f"{delta * 100:+.2f}"
            )

        for method_name in sorted(summary):
            stats = summary[method_name][clicks]
            summary_rows.append({
                "Method": method_name,
                "Clicks": clicks,
                "IoU_mean": stats["mean"],
                "IoU_std": stats["std"],
                "N": stats["n"],
            })

    aggregate_rows = aggregate_clicks(all_records, {
        "EarlyMean": early_clicks,
        "AllMean": target_clicks,
    })

    log_write("\nAggregate IoU summary")
    log_write("Group | Method | Clicks | mean+-std")
    log_write("-" * 80)
    for row in aggregate_rows:
        log_write(
            f"{row['Group']} | {row['Method']} | @{row['Clicks']} | "
            f"{row['IoU_mean'] * 100:.2f}+-{row['IoU_std'] * 100:.2f}"
        )

    write_csv(
        os.path.join(exp_dir, "per_image_iou.csv"),
        all_records,
        ["Method", "Run", "Image", "Clicks", "IoU", "NumActualClicks", "MIUpdates"],
    )
    write_csv(
        os.path.join(exp_dir, "summary_iou.csv"),
        summary_rows,
        ["Method", "Clicks", "IoU_mean", "IoU_std", "N"],
    )
    write_csv(
        os.path.join(exp_dir, "aggregate_iou.csv"),
        aggregate_rows,
        ["Group", "Method", "Clicks", "IoU_mean", "IoU_std", "N"],
    )
    write_csv(
        os.path.join(exp_dir, "latency.csv"),
        latency_rows,
        ["Method", "Run", "Latency_ms"],
    )

    log_write("=" * 80)
    log_write(f"Done. Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
