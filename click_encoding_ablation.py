import argparse
import csv
import os
import random
import shutil
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import segmentation_models_pytorch as smp

from config import (
    device,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_MAX_POINTS,
    PRE_TRAINED,
    PRE_TRAINED_PATH,
)
from dataset import train_dataloader, val_dataloader
from pi_seg.model import initializer
from pi_seg.model.modifiers import LRMult
from pi_seg.model.ops import DistMaps, ScaleLayer
from pi_seg.model.modeling.probabilistic_unet.segmentator import ProbabilisticUnet
from pi_seg.utils.train_utils import get_next_points_removeall, fn_performance, fn_threshold, fn_tonumpy


VARIANT_HELP = {
    "ours": "Exact project encoding: DistMaps(use_disks=True) + learnable maps_transform.",
    "distance": "Continuous tanh-normalized distance maps + learnable maps_transform.",
    "gaussian": "Positive/negative Gaussian click heatmaps + learnable maps_transform.",
    "disk": "Alias of ours kept only for backward compatibility; do not use as an independent baseline.",
    "sparse": "Sparse one-hot click maps, optionally dilated + learnable maps_transform.",
    "distance_no_transform": "Continuous distance maps concatenated directly with image, no maps_transform.",
    "gaussian_no_transform": "Gaussian heatmaps concatenated directly with image, no maps_transform.",
    "disk_no_transform": "DistMaps(use_disks=True) concatenated directly with image, no maps_transform.",
    "sparse_no_transform": "Sparse click maps concatenated directly with image, no maps_transform.",
}


def parse_list(value, cast):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_base_model(model):
    return model.module if hasattr(model, "module") else model


def load_unet_for_coord_channels(coord_channels):
    unet = smp.Unet("resnet50", in_channels=1, encoder_weights=None, classes=1)

    if PRE_TRAINED:
        print(f"Loading CEM/EM pre-trained weights from: {PRE_TRAINED_PATH}")
        state = torch.load(PRE_TRAINED_PATH, map_location="cpu")
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        unet_state_dict = deepcopy(state_dict)
        for key in list(unet_state_dict.keys()):
            unet_state_dict["encoder." + key] = unet_state_dict[key]
            del unet_state_dict[key]
        unet.load_state_dict(unet_state_dict, strict=False)

    encoder_conv1 = nn.Conv2d(1 + coord_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    decoder3_conv1_0 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
    decoder3_conv1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    decoder3_conv2_0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
    decoder3_conv2_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    unet.encoder.conv1 = encoder_conv1
    unet.decoder.blocks[3].conv1[0] = decoder3_conv1_0
    unet.decoder.blocks[3].conv1[1] = decoder3_conv1_1
    unet.decoder.blocks[3].conv2[0] = decoder3_conv2_0
    unet.decoder.blocks[3].conv2[1] = decoder3_conv2_1
    unet.decoder.blocks = unet.decoder.blocks[:4]
    unet.segmentation_head = nn.Sequential()
    return unet


class DistMapClickEncoder(nn.Module):
    def __init__(self, norm_radius=20, use_disks=False):
        super().__init__()
        self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=use_disks)

    def forward(self, image, points):
        return self.dist_maps(image, points)


class GaussianClickEncoder(nn.Module):
    def __init__(self, sigma=10.0):
        super().__init__()
        self.sigma = float(sigma)

    def forward(self, image, points):
        return build_click_heatmaps(image, points, mode="gaussian", sigma=self.sigma)


class SparseClickEncoder(nn.Module):
    def __init__(self, dilation_radius=0):
        super().__init__()
        self.dilation_radius = int(dilation_radius)

    def forward(self, image, points):
        maps = build_click_heatmaps(image, points, mode="sparse")
        if self.dilation_radius > 0:
            kernel = 2 * self.dilation_radius + 1
            maps = F.max_pool2d(maps, kernel_size=kernel, stride=1, padding=self.dilation_radius)
        return maps


def build_click_heatmaps(image, points, mode, sigma=10.0):
    batch_size, _, height, width = image.shape
    num_points = points.shape[1] // 2
    groups = [points[:, :num_points, :2], points[:, num_points:, :2]]
    out_maps = []

    y_grid = torch.arange(height, dtype=torch.float32, device=points.device).view(1, 1, height, 1)
    x_grid = torch.arange(width, dtype=torch.float32, device=points.device).view(1, 1, 1, width)

    for group_points in groups:
        valid = torch.max(group_points, dim=2, keepdim=True)[0] >= 0
        safe_points = torch.clamp(group_points, min=0)
        y = safe_points[:, :, 0].view(batch_size, num_points, 1, 1)
        x = safe_points[:, :, 1].view(batch_size, num_points, 1, 1)

        if mode == "gaussian":
            dist_sq = (y_grid - y) ** 2 + (x_grid - x) ** 2
            maps = torch.exp(-dist_sq / (2.0 * sigma ** 2))
            maps = maps * valid.float().view(batch_size, num_points, 1, 1)
            maps = maps.max(dim=1, keepdim=True)[0]
        elif mode == "sparse":
            maps = torch.zeros((batch_size, 1, height, width), dtype=torch.float32, device=points.device)
            for bidx in range(batch_size):
                for pidx in range(num_points):
                    if not valid[bidx, pidx, 0]:
                        continue
                    yy = int(safe_points[bidx, pidx, 0].item())
                    xx = int(safe_points[bidx, pidx, 1].item())
                    if 0 <= yy < height and 0 <= xx < width:
                        maps[bidx, 0, yy, xx] = 1.0
        else:
            raise ValueError(f"Unknown heatmap mode: {mode}")

        out_maps.append(maps)

    return torch.cat(out_maps, dim=1)


def make_maps_transform(in_channels=2, out_channels=64):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        ScaleLayer(init_value=0.05, lr_mult=1),
    )


class ClickEncodingAblationModel(nn.Module):
    def __init__(self, variant, norm_radius=20, gaussian_sigma=10.0, sparse_radius=0):
        super().__init__()
        self.variant = variant
        self.with_prev_mask = False

        base_variant = variant.replace("_no_transform", "")
        self.use_maps_transform = not variant.endswith("_no_transform")

        if base_variant == "ours":
            self.click_encoder = DistMapClickEncoder(norm_radius=norm_radius, use_disks=True)
        elif base_variant == "distance":
            self.click_encoder = DistMapClickEncoder(norm_radius=norm_radius, use_disks=False)
        elif base_variant == "gaussian":
            self.click_encoder = GaussianClickEncoder(sigma=gaussian_sigma)
        elif base_variant == "disk":
            self.click_encoder = DistMapClickEncoder(norm_radius=norm_radius, use_disks=True)
        elif base_variant == "sparse":
            self.click_encoder = SparseClickEncoder(dilation_radius=sparse_radius)
        else:
            raise ValueError(f"Unknown click encoding variant: {variant}")

        self.raw_coord_channels = 2
        self.coord_channels = 64 if self.use_maps_transform else self.raw_coord_channels
        self.maps_transform = make_maps_transform(self.raw_coord_channels, self.coord_channels) if self.use_maps_transform else nn.Identity()

        self.feature_extractor = ProbabilisticUnet(input_channels=1 + self.coord_channels, num_classes=1)
        self.feature_extractor.apply(LRMult(0.1))
        self.apply(initializer.XavierGluon(rnd_type="gaussian", magnitude=2.0))
        self.feature_extractor.unet = deepcopy(load_unet_for_coord_channels(self.coord_channels))

    def forward(self, image, points, mask=None, training=True):
        raw_coord_features = self.click_encoder(image, points)
        coord_features = self.maps_transform(raw_coord_features)
        outputs = self.feature_extractor(
            patch=image,
            segm=mask,
            coord_features=coord_features,
            training=training,
        )
        outputs["click_map"] = raw_coord_features
        return outputs


def initialize_ablation_model(variant, norm_radius=20, gaussian_sigma=10.0, sparse_radius=0):
    model = ClickEncodingAblationModel(
        variant=variant,
        norm_radius=norm_radius,
        gaussian_sigma=gaussian_sigma,
        sparse_radius=sparse_radius,
    )
    model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for variant {variant}.")
        model = nn.DataParallel(model)
    return model


def empty_points(batch_size, point_slots, height, width, dev):
    return torch.full((batch_size, point_slots * 2, 3), -1.0, dtype=torch.float32, device=dev)


def train_one_epoch(model, train_loader, optimizer, point_slots):
    model.train()
    losses = []

    for batch_data in tqdm(train_loader, desc="Training", leave=False):
        batch_data = {key: value.to(device) for key, value in batch_data.items()}
        image, gt_mask, points = batch_data["images"], batch_data["instances"], batch_data["points"]

        prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
        num_points = random.randint(0, point_slots)

        with torch.no_grad():
            model.eval()
            for click_idx in range(num_points):
                outputs = model(image=image, mask=gt_mask, points=points, training=False)
                prev_output = torch.sigmoid(outputs["instances"])
                points = get_next_points_removeall(prev_output, gt_mask, points, click_idx + 1)

        model.train()
        outputs = model(image=image, mask=gt_mask, points=points, training=True)
        loss = F.binary_cross_entropy_with_logits(outputs["instances"], gt_mask.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else 0.0


def validate_original_protocol(model, val_loader, max_clicks):
    accuracy_lst, iou_lst, recall_lst, precision_lst, dice_lst = [], [], [], [], []

    with torch.no_grad():
        model.eval()
        for batch_data in tqdm(val_loader, desc="Validation", leave=False):
            batch_data = {key: value.to(device) for key, value in batch_data.items()}
            image, gt_mask, points = batch_data["images"], batch_data["instances"], batch_data["points"]

            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
            for click_idx in range(max_clicks):
                outputs = model(image=image, mask=gt_mask, points=points, training=False)
                prev_output = torch.sigmoid(outputs["instances"])
                points = get_next_points_removeall(prev_output, gt_mask, points, click_idx + 1)

            for res, tar in zip(prev_output, gt_mask):
                pred_np = fn_threshold(fn_tonumpy(res).copy())
                gt_np = fn_threshold(fn_tonumpy(tar).copy())
                performance = fn_performance(pred_np, gt_np)
                accuracy_lst.append(performance["accuracy"])
                iou_lst.append(performance["iou_f"])
                recall_lst.append(performance["recall"])
                precision_lst.append(performance["precision"])
                dice_lst.append(performance["DSC"])

    return {
        "orig_accuracy": float(np.mean(accuracy_lst)) if accuracy_lst else 0.0,
        "orig_iou": float(np.mean(iou_lst)) if iou_lst else 0.0,
        "orig_recall": float(np.mean(recall_lst)) if recall_lst else 0.0,
        "orig_precision": float(np.mean(precision_lst)) if precision_lst else 0.0,
        "orig_dsc": float(np.mean(dice_lst)) if dice_lst else 0.0,
    }


def validate_fixed_clicks(model, val_loader, point_slots, target_clicks, iou_thresholds):
    model.eval()
    target_clicks = sorted(target_clicks)
    max_clicks = max(target_clicks)
    per_click = {click: {"iou": [], "dice": [], "precision": [], "recall": []} for click in target_clicks}
    all_iou_curves = []

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validation", leave=False):
            batch_data = {key: value.to(device) for key, value in batch_data.items()}
            image, gt_mask = batch_data["images"], batch_data["instances"]
            batch_size, _, height, width = image.shape

            points = empty_points(batch_size, point_slots, height, width, image.device)
            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
            batch_iou_curve = [[] for _ in range(batch_size)]

            for click_idx in range(1, max_clicks + 1):
                points = get_next_points_removeall(prev_output, gt_mask, points, click_idx)
                outputs = model(image=image, mask=gt_mask, points=points, training=False)
                prev_output = torch.sigmoid(outputs["instances"])

                if click_idx not in per_click:
                    continue

                for item_idx, (res, tar) in enumerate(zip(prev_output, gt_mask)):
                    pred_np = fn_threshold(fn_tonumpy(res).copy())
                    gt_np = fn_threshold(fn_tonumpy(tar).copy())
                    metrics = fn_performance(pred_np, gt_np)
                    per_click[click_idx]["iou"].append(metrics["iou_f"])
                    per_click[click_idx]["dice"].append(metrics["DSC"])
                    per_click[click_idx]["precision"].append(metrics["precision"])
                    per_click[click_idx]["recall"].append(metrics["recall"])
                    batch_iou_curve[item_idx].append(metrics["iou_f"])

            all_iou_curves.extend(batch_iou_curve)

    summary = {}
    for click in target_clicks:
        for metric_name, values in per_click[click].items():
            summary[f"{metric_name}@{click}"] = float(np.mean(values)) if values else 0.0

    for thr in iou_thresholds:
        noc_values = []
        for curve in all_iou_curves:
            reached = [idx + 1 for idx, value in enumerate(curve) if value >= thr]
            noc_values.append(reached[0] if reached else max_clicks)
        summary[f"noc@{thr:.2f}"] = float(np.mean(noc_values)) if noc_values else float(max_clicks)

    return summary


def save_checkpoint(path, model, optimizer, epoch, best_metric, history, args):
    base_model = get_base_model(model)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_metric,
            "history": history,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint_if_available(path, model, optimizer):
    if not path.exists():
        return 1, -1.0, []
    checkpoint = torch.load(path, map_location=device)
    get_base_model(model).load_state_dict(checkpoint["state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_metric = float(checkpoint.get("best_metric", -1.0))
    history = checkpoint.get("history", [])
    print(f"Resumed from {path} at epoch {start_epoch}")
    return start_epoch, best_metric, history


def append_history_csv(path, row, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_variant(variant, args):
    print("\n" + "=" * 80)
    print(f"Click encoding variant: {variant}")
    print(VARIANT_HELP.get(variant, ""))
    print("=" * 80)

    variant_dir = Path(args.save_root) / variant
    if variant_dir.exists() and not args.resume:
        if args.overwrite:
            shutil.rmtree(variant_dir)
        elif (variant_dir / "history.csv").exists():
            raise FileExistsError(
                f"{variant_dir} already contains history.csv. "
                "Use --resume to continue it or --overwrite to start a clean run."
            )
    variant_dir.mkdir(parents=True, exist_ok=True)

    model = initialize_ablation_model(
        variant=variant,
        norm_radius=args.norm_radius,
        gaussian_sigma=args.gaussian_sigma,
        sparse_radius=args.sparse_radius,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    latest_path = variant_dir / "model_latest.pth"
    best_suffix = "orig_iou" if args.selection_metric == "orig_iou" else f"iou_at_{args.selection_click}click"
    best_path = variant_dir / f"model_best_{best_suffix}.pth"

    if args.resume:
        start_epoch, best_metric, history = load_checkpoint_if_available(latest_path, model, optimizer)
    else:
        start_epoch, best_metric, history = 1, -1.0, []

    metric_fields = []
    orig_metric_fields = ["orig_accuracy", "orig_iou", "orig_recall", "orig_precision", "orig_dsc"]
    for click in args.target_clicks:
        metric_fields.extend([f"iou@{click}", f"dice@{click}", f"precision@{click}", f"recall@{click}"])
    metric_fields.extend([f"noc@{thr:.2f}" for thr in args.iou_thresholds])
    fieldnames = ["variant", "epoch", "train_loss", "best_metric"] + orig_metric_fields + metric_fields

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n[{variant}] Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, train_dataloader, optimizer, args.point_slots)
        row = {
            "variant": variant,
            "epoch": epoch,
            "train_loss": train_loss,
            "best_metric": best_metric,
        }

        if epoch % args.validation_step == 0 or epoch == args.epochs:
            metrics = {}
            if args.original_validation:
                metrics.update(
                    validate_original_protocol(
                        model=model,
                        val_loader=val_dataloader,
                        max_clicks=args.max_clicks,
                    )
                )
            if args.fixed_click_validation:
                metrics.update(
                    validate_fixed_clicks(
                        model=model,
                        val_loader=val_dataloader,
                        point_slots=args.point_slots,
                        target_clicks=args.target_clicks,
                        iou_thresholds=args.iou_thresholds,
                    )
                )
            row.update(metrics)
            selected_key = "orig_iou" if args.selection_metric == "orig_iou" else f"iou@{args.selection_click}"
            selected_metric = metrics.get(selected_key, 0.0)
            if selected_metric > best_metric:
                best_metric = selected_metric
                row["best_metric"] = best_metric
                save_checkpoint(best_path, model, optimizer, epoch, best_metric, history, args)
                print(f"[{variant}] New best {selected_key}: {best_metric:.4f}")

            metric_parts = []
            if "orig_iou" in metrics:
                metric_parts.append(f"orig_iou: {metrics['orig_iou']:.4f}")
            metric_parts.extend(
                [f"iou@{click}: {metrics.get(f'iou@{click}', 0.0):.4f}" for click in args.target_clicks]
            )
            metric_text = " | ".join(metric_parts)
            print(f"[{variant}] Train loss: {train_loss:.4f} | {metric_text}")
        else:
            print(f"[{variant}] Train loss: {train_loss:.4f}")

        history.append(row)
        save_checkpoint(latest_path, model, optimizer, epoch, best_metric, history, args)
        append_history_csv(variant_dir / "history.csv", row, fieldnames)
        append_history_csv(Path(args.save_root) / "all_variants_history.csv", row, fieldnames)

    return history[-1] if history else {}


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Ablation study for interactive click encoding modules."
    )
    parser.add_argument(
        "--variants",
        default="ours,sparse_no_transform,gaussian_no_transform,distance_no_transform,disk_no_transform",
        help="Comma-separated variants. Available: " + ", ".join(VARIANT_HELP.keys()),
    )
    parser.add_argument("--save-root", default="SAVE_DIR/click_encoding_ablation")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--validation-step", type=int, default=10)
    parser.add_argument("--target-clicks", default="1,3,5,10,15")
    parser.add_argument("--max-clicks", type=int, default=15)
    parser.add_argument("--selection-click", type=int, default=10)
    parser.add_argument("--selection-metric", choices=["orig_iou", "fixed_click_iou"], default="orig_iou")
    parser.add_argument("--iou-thresholds", default="0.85,0.88,0.90,0.93")
    # Keep the original training click-slot setting. The test-time maximum is
    # controlled independently by --max-clicks and --target-clicks.
    parser.add_argument("--point-slots", type=int, default=NUM_MAX_POINTS)
    parser.add_argument("--norm-radius", type=float, default=20.0)
    parser.add_argument("--gaussian-sigma", type=float, default=10.0)
    parser.add_argument("--sparse-radius", type=int, default=2)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--original-validation", action="store_true", default=True)
    parser.add_argument("--no-original-validation", dest="original_validation", action="store_false")
    parser.add_argument("--fixed-click-validation", action="store_true", default=True)
    parser.add_argument("--no-fixed-click-validation", dest="fixed_click_validation", action="store_false")
    return parser


def main():
    args = build_arg_parser().parse_args()
    args.variants = parse_list(args.variants, str)
    args.target_clicks = parse_list(args.target_clicks, int)
    args.iou_thresholds = parse_list(args.iou_thresholds, float)

    unknown = [variant for variant in args.variants if variant not in VARIANT_HELP]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {list(VARIANT_HELP)}")
    if "disk" in args.variants:
        print("[WARN] 'disk' is an alias of 'ours' in this codebase. Use 'disk_no_transform' or omit 'disk' for a clean ablation.")
    if args.selection_click not in args.target_clicks:
        raise ValueError("--selection-click must be included in --target-clicks")
    if max(args.target_clicks) > args.max_clicks:
        raise ValueError("--target-clicks cannot exceed --max-clicks")
    if max(args.target_clicks) > args.point_slots:
        raise ValueError("--point-slots must be >= max(--target-clicks)")
    if args.selection_metric == "orig_iou" and not args.original_validation:
        raise ValueError("--selection-metric orig_iou requires original validation.")
    if args.selection_metric == "fixed_click_iou" and not args.fixed_click_validation:
        raise ValueError("--selection-metric fixed_click_iou requires fixed-click validation.")

    set_seed(args.seed)
    print(f"Device: {device}")
    print(f"Variants: {', '.join(args.variants)}")
    print(f"Save root: {args.save_root}")

    final_rows = []
    for variant in args.variants:
        final_rows.append(run_variant(variant, args))

    summary_path = Path(args.save_root) / "final_summary.csv"
    if final_rows:
        fieldnames = sorted({key for row in final_rows for key in row.keys()})
        with open(summary_path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_rows)
    print(f"\nAblation finished. Summary: {summary_path}")


if __name__ == "__main__":
    main()
