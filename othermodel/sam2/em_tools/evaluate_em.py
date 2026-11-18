import argparse
import csv
import glob
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import matplotlib.pyplot as plt

OTHERMODEL_ROOT = Path(__file__).resolve().parents[3]
if str(OTHERMODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(OTHERMODEL_ROOT))

from patent_figure_utils import generate_patent_figures, save_patent_npz


def natural_sort_key(value):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"([0-9]+)", str(value))
    ]


def collect_image_paths(image_dir):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(str(Path(image_dir) / ext)))
    return sorted(set(paths), key=natural_sort_key)


def match_mask_path(image_path, mask_dir):
    candidates = sorted(
        glob.glob(str(Path(mask_dir) / f"{Path(image_path).stem}.*")),
        key=natural_sort_key,
    )
    return candidates[0] if candidates else None


def to_uint8_gray(image):
    if image.dtype == np.uint8:
        return image
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def load_binary_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    return (mask > 127).astype(np.uint8)


def load_sam2_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    if image.ndim == 3:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = to_uint8_gray(gray)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray, rgb


def calculate_iou(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum()) / float(union)


def calculate_dice(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum()) / float(denom)


def mask_surface(mask):
    mask = mask.astype(np.uint8)
    if not mask.any():
        return np.zeros_like(mask, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return (mask.astype(bool) ^ eroded.astype(bool)).astype(np.uint8)


def directed_surface_distances(surface_a, surface_b):
    if not surface_a.any() or not surface_b.any():
        return np.array([50.0], dtype=np.float32)
    dist = cv2.distanceTransform((1 - surface_b).astype(np.uint8), cv2.DIST_L2, 5)
    return dist[surface_a.astype(bool)].astype(np.float32)


def calculate_hd95(pred_mask, gt_mask):
    pred = pred_mask.astype(np.uint8)
    gt = gt_mask.astype(np.uint8)
    if not pred.any() or not gt.any():
        return 50.0
    surface_pred = mask_surface(pred)
    surface_gt = mask_surface(gt)
    distances = np.concatenate([
        directed_surface_distances(surface_pred, surface_gt),
        directed_surface_distances(surface_gt, surface_pred),
    ])
    return float(np.percentile(distances, 95))


def current_metrics(pred_mask, gt_mask):
    return {
        "iou": calculate_iou(pred_mask, gt_mask),
        "dice": calculate_dice(pred_mask, gt_mask),
        "hd95": calculate_hd95(pred_mask, gt_mask),
    }


class RobotClicker:
    def __init__(self, gt_mask, min_error_area=5):
        self.gt_mask = gt_mask.astype(np.uint8)
        self.clicked_mask = np.zeros_like(self.gt_mask, dtype=np.uint8)
        self.min_error_area = min_error_area

    def get_next_click(self, pred_mask):
        error_mask = (pred_mask.astype(np.uint8) != self.gt_mask).astype(np.uint8)
        error_mask[self.clicked_mask == 1] = 0
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(error_mask, connectivity=8)
        if num_labels <= 1:
            return None

        areas = stats[:, cv2.CC_STAT_AREA].copy()
        areas[0] = 0
        max_id = int(np.argmax(areas))
        if areas[max_id] < self.min_error_area:
            return None

        largest_component = (labels == max_id).astype(np.uint8)
        distance = cv2.distanceTransform(largest_component, cv2.DIST_L2, 5)
        distance[self.clicked_mask == 1] = 0
        if float(distance.max()) < 1.0:
            return None

        y, x = np.unravel_index(np.argmax(distance), distance.shape)
        self.clicked_mask[y, x] = 1
        is_positive = 1 if self.gt_mask[y, x] == 1 else 0
        return int(y), int(x), int(is_positive)


class SAM2InteractiveWrapper:
    def __init__(self, sam2_root, model_cfg, checkpoint, device, merge_strategy, multimask_output, use_mask_input):
        self.sam2_root = Path(sam2_root).resolve()
        self.model_cfg = model_cfg
        self.checkpoint = str(Path(checkpoint).resolve())
        self.device = device
        self.merge_strategy = merge_strategy
        self.multimask_output = multimask_output
        self.use_mask_input = use_mask_input
        self.predictor = None
        self.low_res_logits = None

    def load(self):
        sys.path.insert(0, str(self.sam2_root))
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(self.model_cfg, self.checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(model)
        self.predictor.model.eval()
        return torch

    def set_image(self, rgb_image):
        self.low_res_logits = None
        self.predictor.set_image(rgb_image)

    def predict(self, clicks):
        if not clicks:
            h, w = self.predictor._orig_hw[-1]
            return np.zeros((h, w), dtype=np.uint8)
        if self.merge_strategy == "positive_union":
            return self._predict_positive_union(clicks)
        return self._predict_prompt_set(clicks)

    def _select_by_score(self, masks, scores, low_res_masks):
        best_idx = int(np.argmax(scores))
        if self.use_mask_input:
            self.low_res_logits = low_res_masks[best_idx:best_idx + 1]
        return masks[best_idx].astype(np.uint8)

    def _predict_prompt_set(self, clicks):
        coords = np.array([[x, y] for y, x, _ in clicks], dtype=np.float32)
        labels = np.array([is_pos for _, _, is_pos in clicks], dtype=np.int32)
        mask_input = self.low_res_logits if self.use_mask_input else None
        masks, scores, low_res_masks = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            mask_input=mask_input,
            multimask_output=self.multimask_output,
        )
        return self._select_by_score(masks, scores, low_res_masks)

    def _predict_positive_union(self, clicks):
        positive_clicks = [click for click in clicks if click[2] == 1]
        negative_clicks = [click for click in clicks if click[2] == 0]
        h, w = self.predictor._orig_hw[-1]
        if not positive_clicks:
            return np.zeros((h, w), dtype=np.uint8)

        union_mask = np.zeros((h, w), dtype=np.uint8)
        for pos_click in positive_clicks:
            local_clicks = [pos_click] + negative_clicks
            coords = np.array([[x, y] for y, x, _ in local_clicks], dtype=np.float32)
            labels = np.array([is_pos for _, _, is_pos in local_clicks], dtype=np.int32)
            masks, scores, _ = self.predictor.predict(
                point_coords=coords,
                point_labels=labels,
                mask_input=None,
                multimask_output=self.multimask_output,
            )
            union_mask = np.logical_or(union_mask, masks[int(np.argmax(scores))]).astype(np.uint8)
        return union_mask


def save_visual(image_gray, gt_mask, pred_mask, clicks, output_path, title):
    canvas = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    overlay = canvas.copy().astype(np.float32)

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)
    overlay[tp] = overlay[tp] * 0.5 + np.array([0, 255, 0]) * 0.5
    overlay[fp] = overlay[fp] * 0.5 + np.array([255, 0, 0]) * 0.5
    overlay[fn] = overlay[fn] * 0.5 + np.array([0, 0, 255]) * 0.5

    for y, x, is_pos in clicks:
        color = (0, 255, 0) if is_pos == 1 else (255, 0, 0)
        cv2.circle(canvas, (x, y), 4, color, -1)
        cv2.circle(canvas, (x, y), 8, color, 1)
        cv2.circle(overlay, (x, y), 4, color, -1)
        cv2.circle(overlay, (x, y), 8, color, 1)

    gt_rgb = np.stack([gt_mask * 255] * 3, axis=-1).astype(np.uint8)
    panel = np.concatenate([canvas, gt_rgb, np.clip(overlay, 0, 255).astype(np.uint8)], axis=1)
    cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def save_paper_visual(image_gray, gt_mask, pred_mask, clicks, output_path, title, iou_score=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    for y, x, is_pos in clicks:
        color = (0, 255, 0) if is_pos == 1 else (255, 0, 0)
        cv2.circle(image_rgb, (x, y), 3, color, -1)
        cv2.circle(image_rgb, (x, y), 6, color, 1)

    axes[0].imshow(image_rgb)
    axes[0].set_title(f"Image + {len(clicks)} Clicks", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth", fontsize=13)
    axes[1].axis("off")

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    overlay_rgb = image_rgb.copy()
    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)
    overlay_rgb[tp] = overlay_rgb[tp] * 0.5 + np.array([0, 255, 0]) * 0.5
    overlay_rgb[fp] = overlay_rgb[fp] * 0.5 + np.array([255, 0, 0]) * 0.5
    overlay_rgb[fn] = overlay_rgb[fn] * 0.5 + np.array([0, 0, 255]) * 0.5

    axes[2].imshow(overlay_rgb)
    if iou_score is None:
        axes[2].set_title(title, fontsize=12)
    else:
        axes[2].set_title(f"{title}\nIoU: {iou_score * 100:.1f}%", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=220, bbox_inches="tight")
    plt.close(fig)


def evaluate_image(wrapper, image_path, mask_path, args):
    image_gray, image_rgb = load_sam2_rgb(image_path)
    gt_mask = load_binary_mask(mask_path)
    wrapper.set_image(image_rgb)

    robot = RobotClicker(gt_mask, min_error_area=args.min_error_area)
    clicks = []
    pred_mask = wrapper.predict(clicks)
    metrics = current_metrics(pred_mask, gt_mask)
    fixed_clicks = {}
    noc = {target: None for target in args.target_ious}
    reached = {target: metrics["iou"] >= target for target in args.target_ious}
    fixed_states = {}
    noc_states = {target: None for target in args.target_ious}
    for target in args.target_ious:
        if reached[target]:
            noc[target] = 0
            noc_states[target] = {
                "pred": pred_mask.copy(),
                "clicks": list(clicks),
                "metrics": dict(metrics),
            }
    latencies = []

    for click_index in range(1, args.max_clicks + 1):
        click = robot.get_next_click(pred_mask)
        if click is not None:
            clicks.append(click)
            start = time.time()
            pred_mask = wrapper.predict(clicks)
            latencies.append(time.time() - start)
            metrics = current_metrics(pred_mask, gt_mask)

        if click_index in args.target_clicks:
            fixed_clicks[click_index] = dict(metrics)
            fixed_states[click_index] = {
                "pred": pred_mask.copy(),
                "clicks": list(clicks),
                "metrics": dict(metrics),
            }

        for target in args.target_ious:
            if noc[target] is None and metrics["iou"] >= target:
                noc[target] = click_index
                reached[target] = True
                noc_states[target] = {
                    "pred": pred_mask.copy(),
                    "clicks": list(clicks),
                    "metrics": dict(metrics),
                }

    for click_count in args.target_clicks:
        fixed_clicks.setdefault(click_count, dict(metrics))
        fixed_states.setdefault(
            click_count,
            {"pred": pred_mask.copy(), "clicks": list(clicks), "metrics": dict(metrics)},
        )
    for target in args.target_ious:
        if noc[target] is None:
            noc[target] = args.max_clicks
            reached[target] = False
        if noc_states[target] is None:
            noc_states[target] = {
                "pred": pred_mask.copy(),
                "clicks": list(clicks),
                "metrics": dict(metrics),
            }

    return {
        "image": Path(image_path).name,
        "shape": f"{gt_mask.shape[0]}x{gt_mask.shape[1]}",
        "fixed_clicks": fixed_clicks,
        "fixed_states": fixed_states,
        "noc": noc,
        "reached": reached,
        "noc_states": noc_states,
        "final_pred": pred_mask,
        "gt": gt_mask,
        "image_gray": image_gray,
        "clicks": clicks,
        "latency_ms": float(np.mean(latencies) * 1000.0) if latencies else 0.0,
        "num_clicks_generated": len(clicks),
    }


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def infer_model_cfg(checkpoint_name):
    name = Path(checkpoint_name).name.lower()
    if "tiny" in name:
        return "configs/sam2.1/sam2.1_hiera_t.yaml"
    if "small" in name:
        return "configs/sam2.1/sam2.1_hiera_s.yaml"
    if "large" in name or "_l" in name:
        return "configs/sam2.1/sam2.1_hiera_l.yaml"
    return "configs/sam2.1/sam2.1_hiera_b+.yaml"


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate SAM2 on whole-image EM semantic interactive segmentation.")
    parser.add_argument("--image-dir", required=True, help="Path to test/input images.")
    parser.add_argument("--mask-dir", required=True, help="Path to test/target masks.")
    parser.add_argument("--sam2-root", default=None, help="Path to local SAM2 repo. Defaults to this script's parent repo.")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM2 checkpoint .pt.")
    parser.add_argument("--model-cfg", default=None, help="SAM2 config path relative to sam2-root. Defaults to base-plus.")
    parser.add_argument("--output-root", default=None, help="Root folder for CSVs and visuals.")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--run-name", default=None, help="Stable output subfolder name. Defaults to timestamped name.")
    parser.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    parser.add_argument("--merge-strategy", choices=["prompt_set", "positive_union"], default="prompt_set")
    parser.add_argument("--target-clicks", default="1,3,5,10,15")
    parser.add_argument("--target-ious", default="0.85,0.88,0.90,0.93")
    parser.add_argument("--max-clicks", type=int, default=15)
    parser.add_argument("--min-error-area", type=int, default=5)
    parser.add_argument("--visualize-max-images", type=int, default=3)
    parser.add_argument("--single-mask", action="store_true", help="Disable SAM2 multimask output.")
    parser.add_argument("--use-mask-input", action="store_true", help="Feed selected low-res logits back into SAM2.")
    parser.add_argument("--save-patent", action="store_true", help="Save patent npz and black-white comparison figures.")
    parser.add_argument("--patent-step", type=int, default=10, help="Click step used for patent figure generation.")
    parser.add_argument("--patent-title", default="TEM interactive segmentation result")
    parser.add_argument("--patent-method-label", default=None)
    parser.add_argument("--patent-reference-dir", default=None, help="Directory containing Base npz files, e.g. .../patent_npz/Base.")
    parser.add_argument("--patent-reference-label", default="Base model")
    return parser


def main():
    default_sam2_root = Path(__file__).resolve().parents[1]
    parser = build_arg_parser()
    args = parser.parse_args()
    args.target_clicks = parse_int_list(args.target_clicks)
    args.target_ious = parse_float_list(args.target_ious)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam2_root = Path(args.sam2_root).resolve() if args.sam2_root else default_sam2_root
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

    model_cfg = args.model_cfg or infer_model_cfg(checkpoint.name)
    dataset_name = args.dataset_name or Path(args.image_dir).parent.parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else sam2_root / "test_result" / "SAM2_EM"
    output_dir = output_root / (args.run_name or f"{dataset_name}_{args.merge_strategy}_{timestamp}")
    visual_dir = output_dir / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    img_paths = collect_image_paths(args.image_dir)
    if not img_paths:
        raise RuntimeError(f"No images found in: {args.image_dir}")

    wrapper = SAM2InteractiveWrapper(
        sam2_root=sam2_root,
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        device=device,
        merge_strategy=args.merge_strategy,
        multimask_output=not args.single_mask,
        use_mask_input=args.use_mask_input,
    )
    wrapper.load()

    per_image_rows = []
    print("=" * 80)
    print("SAM2 EM whole-image semantic interactive evaluation")
    print(f"Dataset: {dataset_name}")
    print(f"Images: {len(img_paths)}")
    print(f"SAM2 root: {sam2_root}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Config: {model_cfg}")
    print(f"Device: {device}")
    print(f"Merge strategy: {args.merge_strategy}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    with torch.inference_mode():
        for idx, image_path in enumerate(img_paths, start=1):
            mask_path = match_mask_path(image_path, args.mask_dir)
            if mask_path is None:
                print(f"[SKIP] No mask found for {Path(image_path).name}")
                continue

            result = evaluate_image(wrapper, image_path, mask_path, args)
            for click_count, metrics in result["fixed_clicks"].items():
                row = {
                    "dataset": dataset_name,
                    "image": result["image"],
                    "image_shape": result["shape"],
                    "clicks": click_count,
                    "iou": metrics["iou"],
                    "dice": metrics["dice"],
                    "hd95": metrics["hd95"],
                    "latency_ms": result["latency_ms"],
                    "num_clicks_generated": result["num_clicks_generated"],
                }
                for target_iou in args.target_ious:
                    row[f"noc_{target_iou:.2f}"] = result["noc"][target_iou]
                    row[f"reached_{target_iou:.2f}"] = result["reached"][target_iou]
                per_image_rows.append(row)

            if idx <= args.visualize_max_images:
                stem = Path(image_path).stem
                for click_count in args.target_clicks:
                    state = result["fixed_states"][click_count]
                    metrics = state["metrics"]
                    if args.save_patent and click_count == args.patent_step:
                        save_patent_npz(
                            result["image_gray"].astype(np.float32) / 255.0,
                            result["gt"],
                            state["pred"],
                            state["clicks"],
                            stem,
                            click_count,
                            f"SAM2_{args.merge_strategy}",
                            metrics["iou"],
                            output_dir,
                        )
                    title = f"SAM2 {args.merge_strategy} | {dataset_name} | {click_count} clicks"
                    save_paper_visual(
                        result["image_gray"],
                        result["gt"],
                        state["pred"],
                        state["clicks"],
                        visual_dir / "FixedClicks" / f"{stem}_step{click_count}.png",
                        title,
                        iou_score=metrics["iou"],
                    )
                for target_iou in args.target_ious:
                    state = result["noc_states"][target_iou]
                    metrics = state["metrics"]
                    target_dir = visual_dir / "NoC" / f"Target_{int(target_iou * 100)}"
                    title = (
                        f"SAM2 {args.merge_strategy} | {dataset_name} | Target IoU {target_iou * 100:.0f}%"
                        f" | Used {result['noc'][target_iou]} clicks"
                    )
                    save_paper_visual(
                        result["image_gray"],
                        result["gt"],
                        state["pred"],
                        state["clicks"],
                        target_dir / f"{stem}_used_{result['noc'][target_iou]}clicks.png",
                        title,
                        iou_score=metrics["iou"],
                    )
                print(f"Saved visuals for {dataset_name}/{stem} to {visual_dir}")

            if idx % 10 == 0 or idx == len(img_paths):
                print(f"Processed {idx}/{len(img_paths)} images")

    per_image_fields = [
        "dataset", "image", "image_shape", "clicks", "iou", "dice", "hd95",
        "latency_ms", "num_clicks_generated",
    ]
    for target_iou in args.target_ious:
        per_image_fields.extend([f"noc_{target_iou:.2f}", f"reached_{target_iou:.2f}"])
    write_csv(output_dir / "sam2_per_image.csv", per_image_rows, per_image_fields)

    fixed_summary = []
    for click_count in args.target_clicks:
        subset = [row for row in per_image_rows if row["clicks"] == click_count]
        iou_mean, iou_std = mean_std([row["iou"] for row in subset])
        dice_mean, dice_std = mean_std([row["dice"] for row in subset])
        hd95_mean, hd95_std = mean_std([row["hd95"] for row in subset])
        lat_mean, lat_std = mean_std([row["latency_ms"] for row in subset])
        fixed_summary.append({
            "method": f"SAM2_{args.merge_strategy}",
            "dataset": dataset_name,
            "clicks": click_count,
            "iou_mean": iou_mean,
            "iou_std": iou_std,
            "dice_mean": dice_mean,
            "dice_std": dice_std,
            "hd95_mean": hd95_mean,
            "hd95_std": hd95_std,
            "latency_ms_mean": lat_mean,
            "latency_ms_std": lat_std,
            "num_samples": len(subset),
        })
    write_csv(
        output_dir / "sam2_summary_fixed_clicks.csv",
        fixed_summary,
        [
            "method", "dataset", "clicks", "iou_mean", "iou_std", "dice_mean", "dice_std",
            "hd95_mean", "hd95_std", "latency_ms_mean", "latency_ms_std", "num_samples",
        ],
    )

    noc_summary = []
    unique_images = sorted({row["image"] for row in per_image_rows})
    image_rows = {image: next(row for row in per_image_rows if row["image"] == image) for image in unique_images}
    for target_iou in args.target_ious:
        values = [row[f"noc_{target_iou:.2f}"] for row in image_rows.values()]
        reached_values = [1.0 if row[f"reached_{target_iou:.2f}"] else 0.0 for row in image_rows.values()]
        noc_mean, noc_std = mean_std(values)
        reached_mean, _ = mean_std(reached_values)
        noc_summary.append({
            "method": f"SAM2_{args.merge_strategy}",
            "dataset": dataset_name,
            "target_iou": target_iou,
            "noc_mean": noc_mean,
            "noc_std": noc_std,
            "success_rate": reached_mean,
            "num_samples": len(values),
        })
    write_csv(
        output_dir / "sam2_summary_noc.csv",
        noc_summary,
        ["method", "dataset", "target_iou", "noc_mean", "noc_std", "success_rate", "num_samples"],
    )

    if args.save_patent:
        generate_patent_figures(
            output_root=output_dir,
            method_mode=f"SAM2_{args.merge_strategy}",
            step=args.patent_step,
            fig_title=args.patent_title,
            reference_dir=args.patent_reference_dir,
            reference_label=args.patent_reference_label,
            method_label=args.patent_method_label or f"SAM2 {args.merge_strategy}",
        )

    print("\n[DONE] SAM2 evaluation finished.")
    print(f"Per-image rows: {output_dir / 'sam2_per_image.csv'}")
    print(f"Fixed-click summary: {output_dir / 'sam2_summary_fixed_clicks.csv'}")
    print(f"NoC summary: {output_dir / 'sam2_summary_noc.csv'}")
    print(f"Visuals: {visual_dir}")


if __name__ == "__main__":
    main()
