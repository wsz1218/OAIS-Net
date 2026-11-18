import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import sys

OTHERMODEL_ROOT = Path(__file__).resolve().parents[1]
if str(OTHERMODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(OTHERMODEL_ROOT))

from patent_figure_utils import generate_patent_figures, save_patent_npz

from em_utils import (
    RobotClicker,
    calculate_dice,
    calculate_iou,
    collect_image_paths,
    current_metrics,
    encode_click_maps,
    load_binary_mask,
    load_gray_image,
    match_mask_path,
    mean_std,
    pad_to_multiple,
    parse_float_list,
    parse_int_list,
    save_paper_visual,
    write_csv,
)
from model import ClickUNet


DEFAULT_DOMAINS = [
    "domain_2_cells",
    "domain_3_muscle",
    "domain_4_fuzzy_blobs",
    "domain_6_Lucchi",
]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate Click U-Net on EM domains with OAIS-style automatic clicks."
    )
    parser.add_argument("--dataset-root", required=True, help="Root containing domain_xxx/test/input,target folders.")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--checkpoint-root", default="experiments/click_unet_em_domains")
    parser.add_argument("--checkpoint-pattern", default="{domain}/checkpoints/last_checkpoint.pth")
    parser.add_argument("--output-root", default="test_result/ClickUNet_EM")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--click-sigma", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-clicks", default="1,3,5,10,15")
    parser.add_argument("--target-ious", default="0.85,0.88,0.90,0.93")
    parser.add_argument("--max-clicks", type=int, default=15)
    parser.add_argument("--min-error-area", type=int, default=5)
    parser.add_argument("--visualize-max-images", type=int, default=3)
    parser.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    parser.add_argument("--save-patent", action="store_true", help="Save patent npz and black-white comparison figures.")
    parser.add_argument("--patent-step", type=int, default=10)
    parser.add_argument("--patent-title", default="TEM interactive segmentation result")
    parser.add_argument("--patent-method-label", default="Interactive U-Net")
    parser.add_argument("--patent-reference-root", default=None, help="Root containing per-domain Base patent_npz/Base folders.")
    parser.add_argument("--patent-reference-label", default="Base model")
    return parser


def resolve_checkpoint(args, domain):
    checkpoint = Path(args.checkpoint_root) / args.checkpoint_pattern.format(domain=domain)
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint for {domain} not found: {checkpoint}")
    return checkpoint


def load_model(checkpoint_path, device, default_base_channels):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    base_channels = int(saved_args.get("base_channels", default_base_channels))
    model = ClickUNet(in_channels=3, base_channels=base_channels).to(device)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def predict_mask(model, image_gray, clicks, sigma, threshold, device):
    h, w = image_gray.shape[:2]
    pos_map, neg_map = encode_click_maps((h, w), clicks, sigma=sigma)
    image_f = image_gray.astype(np.float32) / 255.0
    x = np.stack([image_f, pos_map, neg_map], axis=0)[None].astype(np.float32)
    tensor = torch.from_numpy(x).to(device)
    tensor, original_hw = pad_to_multiple(tensor, multiple=16)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0, :original_hw[0], :original_hw[1]]
    pred = (probs.detach().cpu().numpy() >= threshold).astype(np.uint8)
    return pred


def evaluate_image(model, image_path, mask_path, args, device):
    image_gray = load_gray_image(image_path)
    gt_mask = load_binary_mask(mask_path)
    robot = RobotClicker(gt_mask, min_error_area=args.min_error_area)

    clicks = []
    pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)
    metrics = current_metrics(pred_mask, gt_mask)
    fixed_clicks = {}
    noc = {target: (0 if metrics["iou"] >= target else None) for target in args.target_ious}
    latencies = []
    states = {}
    noc_states = {target: None for target in args.target_ious}

    for click_index in range(1, args.max_clicks + 1):
        click = robot.get_next_click(pred_mask)
        if click is not None:
            clicks.append(click)
            start = time.time()
            pred_mask = predict_mask(
                model=model,
                image_gray=image_gray,
                clicks=clicks,
                sigma=args.click_sigma,
                threshold=args.threshold,
                device=device,
            )
            latencies.append(time.time() - start)
            metrics = current_metrics(pred_mask, gt_mask)

        if click_index in args.target_clicks:
            fixed_clicks[click_index] = dict(metrics)
            states[click_index] = {
                "pred": pred_mask.copy(),
                "clicks": list(clicks),
                "metrics": dict(metrics),
            }

        for target in args.target_ious:
            if noc[target] is None and metrics["iou"] >= target:
                noc[target] = click_index
                noc_states[target] = {
                    "pred": pred_mask.copy(),
                    "clicks": list(clicks),
                    "metrics": dict(metrics),
                }

    for click_count in args.target_clicks:
        fixed_clicks.setdefault(click_count, dict(metrics))
        states.setdefault(
            click_count,
            {"pred": pred_mask.copy(), "clicks": list(clicks), "metrics": dict(metrics)},
        )
    for target in args.target_ious:
        if noc[target] is None:
            noc[target] = args.max_clicks
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
        "noc": noc,
        "states": states,
        "noc_states": noc_states,
        "gt": gt_mask,
        "image_gray": image_gray,
        "latency_ms": float(np.mean(latencies) * 1000.0) if latencies else 0.0,
        "num_clicks_generated": len(clicks),
    }


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def evaluate_domain(domain, domain_dir, checkpoint, output_root, args, device):
    image_dir = domain_dir / "test" / "input"
    mask_dir = domain_dir / "test" / "target"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"{domain} must contain test/input and test/target.")

    output_dir = output_root / domain
    visual_dir = output_dir / "visuals"
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint, device, args.base_channels)
    image_paths = collect_image_paths(image_dir)
    if not image_paths:
        raise RuntimeError(f"No test images found in {image_dir}")

    print("\n" + "=" * 80)
    print(f"Evaluating Click U-Net domain: {domain}")
    print(f"Images: {len(image_paths)}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {output_dir.resolve()}")
    print("=" * 80)

    per_image_rows = []
    for idx, image_path in enumerate(image_paths, start=1):
        mask_path = match_mask_path(image_path, mask_dir)
        if mask_path is None:
            print(f"[SKIP] No mask found for {Path(image_path).name}")
            continue

        result = evaluate_image(model, image_path, mask_path, args, device)
        for click_count, metrics in result["fixed_clicks"].items():
            row = {
                "method": "ClickUNet",
                "dataset": domain,
                "image": result["image"],
                "image_shape": result["shape"],
                "clicks": click_count,
                "iou": metrics["iou"],
                "dice": metrics["dice"],
                "latency_ms": result["latency_ms"],
                "num_clicks_generated": result["num_clicks_generated"],
            }
            for target_iou in args.target_ious:
                row[f"noc_{target_iou:.2f}"] = result["noc"][target_iou]
            per_image_rows.append(row)

        if idx <= args.visualize_max_images:
            stem = Path(image_path).stem
            for click_count in args.target_clicks:
                state = result["states"][click_count]
                metrics = state["metrics"]
                if args.save_patent and click_count == args.patent_step:
                    save_patent_npz(
                        result["image_gray"].astype(np.float32) / 255.0,
                        result["gt"],
                        state["pred"],
                        state["clicks"],
                        stem,
                        click_count,
                        "ClickUNet",
                        metrics["iou"],
                        output_dir,
                    )
                title = f"ClickUNet | {domain} | {click_count} clicks"
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
                    f"ClickUNet | {domain} | Target IoU {target_iou * 100:.0f}%"
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

        if idx % 10 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)} images")

    per_image_fields = [
        "method", "dataset", "image", "image_shape", "clicks", "iou", "dice",
        "latency_ms", "num_clicks_generated",
    ] + [f"noc_{target_iou:.2f}" for target_iou in args.target_ious]
    write_csv(output_dir / "click_unet_per_image.csv", per_image_rows, per_image_fields)

    fixed_summary = []
    for click_count in args.target_clicks:
        subset = [row for row in per_image_rows if row["clicks"] == click_count]
        iou_mean, iou_std = mean_std([row["iou"] for row in subset])
        dice_mean, dice_std = mean_std([row["dice"] for row in subset])
        fixed_summary.append({
            "method": "ClickUNet",
            "dataset": domain,
            "clicks": click_count,
            "iou_mean": iou_mean,
            "iou_std": iou_std,
            "dice_mean": dice_mean,
            "dice_std": dice_std,
            "num_samples": len(subset),
        })
    write_csv(
        output_dir / "click_unet_summary_fixed_clicks.csv",
        fixed_summary,
        ["method", "dataset", "clicks", "iou_mean", "iou_std", "dice_mean", "dice_std", "num_samples"],
    )

    noc_summary = []
    unique_images = sorted({row["image"] for row in per_image_rows})
    image_rows = {image: next(row for row in per_image_rows if row["image"] == image) for image in unique_images}
    for target_iou in args.target_ious:
        values = [row[f"noc_{target_iou:.2f}"] for row in image_rows.values()]
        noc_mean, noc_std = mean_std(values)
        noc_summary.append({
            "method": "ClickUNet",
            "dataset": domain,
            "target_iou": target_iou,
            "noc_mean": noc_mean,
            "noc_std": noc_std,
            "num_samples": len(values),
        })
    write_csv(
        output_dir / "click_unet_summary_noc.csv",
        noc_summary,
        ["method", "dataset", "target_iou", "noc_mean", "noc_std", "num_samples"],
    )

    if args.save_patent:
        reference_dir = None
        if args.patent_reference_root:
            reference_dir = Path(args.patent_reference_root) / domain / "patent_npz" / "Base"
        generate_patent_figures(
            output_root=output_dir,
            method_mode="ClickUNet",
            step=args.patent_step,
            fig_title=args.patent_title,
            reference_dir=reference_dir,
            reference_label=args.patent_reference_label,
            method_label=args.patent_method_label,
        )

    return output_dir


def main():
    args = build_arg_parser().parse_args()
    args.target_clicks = parse_int_list(args.target_clicks)
    args.target_ious = parse_float_list(args.target_ious)
    dataset_root = Path(args.dataset_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    output_root = Path(args.output_root).resolve()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    output_dirs = []
    for domain in args.domains:
        domain_dir = dataset_root / domain
        checkpoint = resolve_checkpoint(args, domain)
        output_dirs.append(evaluate_domain(domain, domain_dir, checkpoint, output_root, args, device))

    fixed_rows = []
    noc_rows = []
    for output_dir in output_dirs:
        fixed_rows.extend(read_csv_rows(output_dir / "click_unet_summary_fixed_clicks.csv"))
        noc_rows.extend(read_csv_rows(output_dir / "click_unet_summary_noc.csv"))

    write_csv(output_root / "click_unet_all_domains_fixed_clicks.csv", fixed_rows)
    write_csv(output_root / "click_unet_all_domains_noc.csv", noc_rows)
    print("\n[DONE] Click U-Net all-domain evaluation finished.")
    print(f"Combined fixed-click summary: {output_root / 'click_unet_all_domains_fixed_clicks.csv'}")
    print(f"Combined NoC summary: {output_root / 'click_unet_all_domains_noc.csv'}")


if __name__ == "__main__":
    main()
