import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import csv
import glob
import importlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, label

from model import initialize_model


ABLATION_MODES = {
    "Base": {"sta": False, "ltm": "none", "reset_sta_after_image": False},
    "Base+STA": {"sta": True, "ltm": "none", "reset_sta_after_image": True},
    "Base+LTM": {"sta": False, "ltm": "pure", "ltm_trigger": "end-of-image", "reset_sta_after_image": False},
    "OAIS-Net": {"sta": True, "ltm": "original", "ltm_trigger": "per-click", "reset_sta_after_image": False},
}


def natural_sort_key(value):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"([0-9]+)", str(value))]


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", value).strip("_")


def calculate_iou(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection / union) if union > 0 else 1.0


def calculate_dice(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    denom = pred.sum() + gt.sum()
    return float((2.0 * np.logical_and(pred, gt).sum()) / denom) if denom > 0 else 1.0


def surface_distances(mask_a, mask_b):
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if not a.any() or not b.any():
        return np.array([50.0], dtype=np.float32)

    structure = np.ones((3, 3), dtype=bool)
    surface_a = np.logical_xor(a, binary_erosion(a, structure=structure, border_value=0))
    surface_b = np.logical_xor(b, binary_erosion(b, structure=structure, border_value=0))

    if not surface_a.any() or not surface_b.any():
        return np.array([50.0], dtype=np.float32)

    dist_to_b = distance_transform_edt(~surface_b)
    dist_to_a = distance_transform_edt(~surface_a)
    return np.concatenate([dist_to_b[surface_a], dist_to_a[surface_b]]).astype(np.float32)


def calculate_hd95(pred_mask, gt_mask):
    return float(np.percentile(surface_distances(pred_mask, gt_mask), 95))


def load_binary_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    return (mask > 127).astype(np.uint8)


def load_image_tensor(path, device):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    tensor = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) / 255.0
    return image, tensor


def match_mask_path(image_path, mask_dir):
    base_name = Path(image_path).stem
    candidates = sorted(glob.glob(str(Path(mask_dir) / f"{base_name}.*")), key=natural_sort_key)
    return candidates[0] if candidates else None


def collect_image_paths(image_dir):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(str(Path(image_dir) / ext)))
    return sorted(set(paths), key=natural_sort_key)


class RobotClicker:
    def __init__(self, gt_mask, min_error_area=5):
        self.gt_mask = gt_mask.astype(np.uint8)
        self.clicked_mask = np.zeros_like(gt_mask, dtype=np.uint8)
        self.min_error_area = min_error_area

    def get_next_click(self, pred_mask):
        error_mask = (pred_mask.astype(np.uint8) != self.gt_mask).astype(np.uint8)
        error_mask[self.clicked_mask == 1] = 0

        labeled_array, num_features = label(error_mask)
        if num_features == 0:
            return None

        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0
        max_id = int(np.argmax(component_sizes))
        if component_sizes[max_id] < self.min_error_area:
            return None

        largest_component = (labeled_array == max_id).astype(np.uint8)
        distance = cv2.distanceTransform(largest_component, cv2.DIST_L2, 5)
        distance[self.clicked_mask == 1] = 0
        if float(distance.max()) < 1.0:
            return None

        y, x = np.unravel_index(np.argmax(distance), distance.shape)
        self.clicked_mask[y, x] = 1
        is_positive = 1 if self.gt_mask[y, x] == 1 else 0
        return int(y), int(x), int(is_positive)


def empty_points_tensor(device, point_slots_per_class):
    return torch.full((1, point_slots_per_class * 2, 3), -1.0, device=device)


def append_click_to_tensor(points_tensor, click):
    y, x, is_positive = click
    new_points = points_tensor.clone()
    point_slots_per_class = new_points.size(1) // 2
    offset = 0 if is_positive == 1 else point_slots_per_class
    for index in range(offset, offset + point_slots_per_class):
        if new_points[0, index, 0] == -1.0:
            new_points[0, index, 0] = y
            new_points[0, index, 1] = x
            new_points[0, index, 2] = 1.0
            return new_points
    return new_points


def forward_mask(model, image_tensor, points_tensor):
    with torch.no_grad():
        output = model(image_tensor, points_tensor, training=False)
        logits = output["instances"] if isinstance(output, dict) else output
        pred = (torch.sigmoid(logits) > 0.5).detach().cpu().numpy()[0, 0].astype(np.uint8)
    return pred


def load_adaptor_class(module_name):
    module = importlib.import_module(module_name)
    return module.OAIMSAdaptor


def load_ready_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = initialize_model()
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    clean_state = {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading checkpoint: {len(unexpected)}")
    return model.to(device).eval()


def reset_short_term_weights(adaptor):
    if not hasattr(adaptor, "initial_state_dict"):
        return

    sta_ids = {id(param) for param in getattr(adaptor, "sta_params", [])}
    with torch.no_grad():
        for name, param in adaptor.model.named_parameters():
            lower_name = name.lower()
            is_sta_param = id(param) in sta_ids if sta_ids else False
            is_default_short_term = "decoder" in lower_name or "head" in lower_name
            if (is_sta_param or (not sta_ids and is_default_short_term)) and name in adaptor.initial_state_dict:
                param.copy_(adaptor.initial_state_dict[name])

    if hasattr(adaptor, "optimizer"):
        adaptor.optimizer.state.clear()


def snapshot_sta_weights(adaptor):
    sta_ids = {id(param) for param in getattr(adaptor, "sta_params", [])}
    if not sta_ids:
        sta_ids = {id(param) for param in getattr(adaptor, "decoder_params", [])}
    return {
        name: param.detach().clone()
        for name, param in adaptor.model.named_parameters()
        if id(param) in sta_ids
    }


def snapshot_initial_sta_weights(adaptor):
    sta_ids = {id(param) for param in getattr(adaptor, "sta_params", [])}
    if not sta_ids:
        sta_ids = {id(param) for param in getattr(adaptor, "decoder_params", [])}
    return {
        name: adaptor.initial_state_dict[name].detach().clone()
        for name, param in adaptor.model.named_parameters()
        if id(param) in sta_ids and name in adaptor.initial_state_dict
    }


def snapshot_trainable_weights(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }


def restore_weights(model, state):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state:
                param.copy_(state[name])


def dice_bce_loss(pred_logits, target_mask):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, target_mask)
    pred_probs = torch.sigmoid(pred_logits)
    intersection = (pred_probs * target_mask).sum(dim=(2, 3))
    union = pred_probs.sum(dim=(2, 3)) + target_mask.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
    return bce + dice.mean()


def anchor_loss(adaptor):
    if not hasattr(adaptor, "initial_state_dict"):
        return torch.tensor(0.0, device=next(adaptor.model.parameters()).device)

    reg_loss = torch.tensor(0.0, device=next(adaptor.model.parameters()).device)
    for name, param in adaptor.model.named_parameters():
        if param.requires_grad and name in adaptor.initial_state_dict:
            reg_loss = reg_loss + torch.sum((param - adaptor.initial_state_dict[name]) ** 2)
    return reg_loss


def adapt_pure_ltm(adaptor, image_tensor, final_mask, anchor_weight=0.5, lr=1e-5, steps=2):
    if isinstance(final_mask, np.ndarray) and final_mask.ndim == 3:
        final_mask = final_mask[:, :, 0]

    target = torch.tensor(final_mask, device=image_tensor.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    target = (target > 0.5).float()
    empty_points = torch.full((1, 20, 3), -1.0, device=image_tensor.device)

    adaptor.model.eval()
    if hasattr(adaptor, "_set_mode"):
        adaptor._set_mode("pi_memory")
    else:
        for name, param in adaptor.model.named_parameters():
            lower_name = name.lower()
            param.requires_grad = "decoder" not in lower_name and "head" not in lower_name

    if hasattr(adaptor, "optimizer") and adaptor.optimizer.param_groups:
        adaptor.optimizer.param_groups[0]["lr"] = lr

    last_loss = None
    with torch.enable_grad():
        for _ in range(steps):
            adaptor.optimizer.zero_grad(set_to_none=True)
            output = adaptor.model(image_tensor, empty_points, training=False)
            logits = output["instances"] if isinstance(output, dict) else output
            loss = dice_bce_loss(logits, target) + anchor_weight * anchor_loss(adaptor)
            loss.backward()
            if hasattr(adaptor, "encoder_params"):
                torch.nn.utils.clip_grad_norm_(adaptor.encoder_params, max_norm=1.0)
            adaptor.optimizer.step()
            last_loss = loss

    return float(last_loss.detach().cpu()) if last_loss is not None else 0.0


def run_ltm_update(adaptor, image_tensor, points_tensor, gt_mask, mode_cfg, args):
    if mode_cfg["ltm"] == "none":
        return
    if mode_cfg["ltm"] == "original":
        adaptor.adapt_PI(image_tensor, points_tensor, gt_mask, max_clicks=args.ltm_max_clicks)
        return
    if mode_cfg["ltm"] == "pure":
        adapt_pure_ltm(
            adaptor,
            image_tensor,
            gt_mask,
            anchor_weight=args.pure_ltm_anchor_weight,
            lr=args.pure_ltm_lr,
            steps=args.pure_ltm_steps,
        )
        return
    raise ValueError(f"Unknown LTM mode: {mode_cfg['ltm']}")


def metrics_pass_guard(before, after, args, guard_type):
    if after["iou"] + args.ltm_iou_tolerance < before["iou"]:
        return False
    if after["dice"] + args.ltm_dice_tolerance < before["dice"]:
        return False

    if guard_type == "quality":
        return after["hd95"] <= before["hd95"] + args.ltm_hd95_tolerance

    if guard_type == "boundary":
        iou_gain = after["iou"] - before["iou"]
        dice_gain = after["dice"] - before["dice"]
        hd95_gain = before["hd95"] - after["hd95"]
        if after["hd95"] > before["hd95"] + args.ltm_hd95_tolerance:
            return False
        return (
            iou_gain >= args.ltm_min_accept_iou_gain
            or dice_gain >= args.ltm_min_accept_dice_gain
            or hd95_gain >= args.ltm_min_accept_hd95_gain
        )

    return True


def guarded_ltm_update(adaptor, image_tensor, points_tensor, gt_mask, mode_cfg, args, guard_type):
    if not args.ltm_rollback:
        run_ltm_update(adaptor, image_tensor, points_tensor, gt_mask, mode_cfg, args)
        return

    before_pred = forward_mask(adaptor.model, image_tensor, points_tensor)
    before_metrics = current_metrics(before_pred, gt_mask)
    before_state = snapshot_trainable_weights(adaptor.model)

    run_ltm_update(adaptor, image_tensor, points_tensor, gt_mask, mode_cfg, args)

    after_pred = forward_mask(adaptor.model, image_tensor, points_tensor)
    after_metrics = current_metrics(after_pred, gt_mask)
    if not metrics_pass_guard(before_metrics, after_metrics, args, guard_type):
        restore_weights(adaptor.model, before_state)


def current_metrics(pred_mask, gt_mask):
    return {
        "iou": calculate_iou(pred_mask, gt_mask),
        "dice": calculate_dice(pred_mask, gt_mask),
        "hd95": calculate_hd95(pred_mask, gt_mask),
    }


def maybe_synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def run_one_image(
    adaptor,
    image_path,
    mask_path,
    mode_cfg,
    args,
    device,
):
    image_np, image_tensor = load_image_tensor(image_path, device)
    gt_mask = load_binary_mask(mask_path)
    ltm_trigger = mode_cfg.get("ltm_trigger", args.ltm_trigger)

    robot = RobotClicker(gt_mask, min_error_area=args.min_error_area)
    points_tensor = empty_points_tensor(device, args.point_slots_per_class)
    all_clicks = []
    click_latencies = []
    ltm_latencies = []

    pred_mask = forward_mask(adaptor.model, image_tensor, points_tensor)
    metrics = current_metrics(pred_mask, gt_mask)

    fixed_click_records = {}
    noc_records = {}
    if args.with_noc:
        noc_records = {
            target_iou: (0 if metrics["iou"] >= target_iou else None)
            for target_iou in args.target_ious
        }

    for click_index in range(1, args.max_clicks + 1):
        click = robot.get_next_click(pred_mask)
        if click is not None:
            all_clicks.append(click)
            points_tensor = append_click_to_tensor(points_tensor, click)

            maybe_synchronize(device)
            start = time.time()
            if mode_cfg["sta"]:
                before_sta_state = None
                before_sta_metrics = None
                if args.sta_rollback:
                    current_sta_state = snapshot_sta_weights(adaptor)
                    current_pred = forward_mask(adaptor.model, image_tensor, points_tensor)
                    current_sta_metrics = current_metrics(current_pred, gt_mask)

                    if mode_cfg["ltm"] == "none":
                        initial_sta_state = snapshot_initial_sta_weights(adaptor)
                        restore_weights(adaptor.model, initial_sta_state)
                        base_pred = forward_mask(adaptor.model, image_tensor, points_tensor)
                        base_metrics = current_metrics(base_pred, gt_mask)

                        if current_sta_metrics["iou"] >= base_metrics["iou"]:
                            before_sta_state = current_sta_state
                            before_sta_metrics = current_sta_metrics
                            restore_weights(adaptor.model, current_sta_state)
                        else:
                            before_sta_state = initial_sta_state
                            before_sta_metrics = base_metrics
                    else:
                        before_sta_state = current_sta_state
                        before_sta_metrics = current_sta_metrics

                if hasattr(adaptor, "ablation_sta_only_mode"):
                    adaptor.ablation_sta_only_mode = mode_cfg["ltm"] == "none"
                adaptor.adapt_MI(image_tensor, points_tensor, all_clicks)
                if hasattr(adaptor, "ablation_sta_only_mode"):
                    adaptor.ablation_sta_only_mode = False
            pred_mask = forward_mask(adaptor.model, image_tensor, points_tensor)
            if (
                mode_cfg["sta"]
                and args.sta_rollback
                and before_sta_state is not None
                and before_sta_metrics is not None
            ):
                updated_metrics = current_metrics(pred_mask, gt_mask)
                if mode_cfg["ltm"] == "none":
                    should_restore_sta = updated_metrics["iou"] + args.sta_rollback_tolerance < before_sta_metrics["iou"]
                else:
                    should_restore_sta = not metrics_pass_guard(before_sta_metrics, updated_metrics, args, "quality")
                if should_restore_sta:
                    restore_weights(adaptor.model, before_sta_state)
                    pred_mask = forward_mask(adaptor.model, image_tensor, points_tensor)
            maybe_synchronize(device)
            click_latencies.append(time.time() - start)

            metrics = current_metrics(pred_mask, gt_mask)

        if click_index in args.target_clicks:
            fixed_click_records[click_index] = dict(metrics)

        if args.with_noc:
            for target_iou in args.target_ious:
                if noc_records[target_iou] is None and metrics["iou"] >= target_iou:
                    noc_records[target_iou] = click_index

        if ltm_trigger == "per-click" and mode_cfg["ltm"] != "none" and all_clicks and metrics["iou"] >= args.ltm_min_iou:
            maybe_synchronize(device)
            start = time.time()
            guarded_ltm_update(
                adaptor,
                image_tensor,
                points_tensor,
                gt_mask,
                mode_cfg,
                args,
                guard_type="quality",
            )
            maybe_synchronize(device)
            ltm_latencies.append(time.time() - start)

    for click_count in args.target_clicks:
        fixed_click_records.setdefault(click_count, dict(metrics))

    if args.with_noc:
        for target_iou in args.target_ious:
            if noc_records[target_iou] is None:
                noc_records[target_iou] = args.max_clicks

    if ltm_trigger == "end-of-image" and mode_cfg["ltm"] != "none" and all_clicks and metrics["iou"] >= args.ltm_min_iou:
        maybe_synchronize(device)
        start = time.time()
        guarded_ltm_update(
            adaptor,
            image_tensor,
            points_tensor,
            gt_mask,
            mode_cfg,
            args,
            guard_type="boundary",
        )
        maybe_synchronize(device)
        ltm_latencies.append(time.time() - start)

    if args.reset_short_term_between_images or mode_cfg.get("reset_sta_after_image", False):
        reset_short_term_weights(adaptor)

    return {
        "image": Path(image_path).name,
        "image_shape": f"{image_np.shape[0]}x{image_np.shape[1]}",
        "num_clicks_generated": len(all_clicks),
        "fixed_clicks": fixed_click_records,
        "noc": noc_records,
        "final_iou": metrics["iou"],
        "final_dice": metrics["dice"],
        "final_hd95": metrics["hd95"],
        "click_latency_ms": float(np.mean(click_latencies) * 1000.0) if click_latencies else 0.0,
        "ltm_latency_ms": float(np.mean(ltm_latencies) * 1000.0) if ltm_latencies else 0.0,
    }


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std(ddof=0))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_mean_std(mean_value, std_value, scale=100.0, digits=2):
    return f"{mean_value * scale:.{digits}f}±{std_value * scale:.{digits}f}"


def format_delta(value, scale=100.0, digits=2):
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * scale:.{digits}f}"


def build_paper_tables(fixed_rows):
    by_key = {
        (row["clicks"], row["mode"]): row
        for row in fixed_rows
    }
    modes = list(ABLATION_MODES.keys())
    paper_rows = []
    improvement_rows = []

    for click_count in sorted({row["clicks"] for row in fixed_rows}):
        base = by_key.get((click_count, "Base"))
        if base is None:
            continue

        iou_values = [by_key[(click_count, mode)]["iou_mean"] for mode in modes if (click_count, mode) in by_key]
        dice_values = [by_key[(click_count, mode)]["dice_mean"] for mode in modes if (click_count, mode) in by_key]
        hd95_values = [by_key[(click_count, mode)]["hd95_mean"] for mode in modes if (click_count, mode) in by_key]
        best_iou = max(iou_values) if iou_values else None
        best_dice = max(dice_values) if dice_values else None
        best_hd95 = min(hd95_values) if hd95_values else None

        row = {"clicks": click_count}
        improvement_row = {"clicks": click_count}

        for mode in modes:
            item = by_key.get((click_count, mode))
            if item is None:
                continue

            iou_text = format_mean_std(item["iou_mean"], item["iou_std"])
            dice_text = format_mean_std(item["dice_mean"], item["dice_std"])
            hd95_text = f"{item['hd95_mean']:.2f}±{item['hd95_std']:.2f}"

            row[f"{mode}_IoU(%)"] = f"*{iou_text}" if item["iou_mean"] == best_iou else iou_text
            row[f"{mode}_Dice(%)"] = f"*{dice_text}" if item["dice_mean"] == best_dice else dice_text
            row[f"{mode}_HD95"] = f"*{hd95_text}" if item["hd95_mean"] == best_hd95 else hd95_text

            improvement_row[f"{mode}_delta_IoU(pp)"] = format_delta(item["iou_mean"] - base["iou_mean"])
            improvement_row[f"{mode}_delta_Dice(pp)"] = format_delta(item["dice_mean"] - base["dice_mean"])
            improvement_row[f"{mode}_delta_HD95(px)"] = f"{item['hd95_mean'] - base['hd95_mean']:+.2f}"

        paper_rows.append(row)
        improvement_rows.append(improvement_row)

    paper_fields = ["clicks"]
    improvement_fields = ["clicks"]
    for mode in modes:
        paper_fields.extend([f"{mode}_IoU(%)", f"{mode}_Dice(%)", f"{mode}_HD95"])
        improvement_fields.extend([
            f"{mode}_delta_IoU(pp)",
            f"{mode}_delta_Dice(pp)",
            f"{mode}_delta_HD95(px)",
        ])

    return paper_rows, paper_fields, improvement_rows, improvement_fields


def summarize(per_image_rows, args):
    fixed_rows = []
    noc_rows = []

    for mode_name in ABLATION_MODES:
        mode_rows = [row for row in per_image_rows if row["mode"] == mode_name]

        for click_count in args.target_clicks:
            subset = [row for row in mode_rows if row["clicks"] == click_count]
            iou_mean, iou_std = mean_std([row["iou"] for row in subset])
            dice_mean, dice_std = mean_std([row["dice"] for row in subset])
            hd95_mean, hd95_std = mean_std([row["hd95"] for row in subset])
            lat_mean, lat_std = mean_std([row["click_latency_ms"] for row in subset])
            ltm_lat_mean, ltm_lat_std = mean_std([row["ltm_latency_ms"] for row in subset])
            fixed_rows.append({
                "mode": mode_name,
                "clicks": click_count,
                "iou_mean": iou_mean,
                "iou_std": iou_std,
                "dice_mean": dice_mean,
                "dice_std": dice_std,
                
                "hd95_mean": hd95_mean,
                "hd95_std": hd95_std,
                "click_latency_ms_mean": lat_mean,
                "click_latency_ms_std": lat_std,
                "ltm_latency_ms_mean": ltm_lat_mean,
                "ltm_latency_ms_std": ltm_lat_std,
                "num_samples": len(subset),
            })

        if not args.with_noc:
            continue

        image_level = {}
        for row in mode_rows:
            image_level.setdefault((row["run"], row["image"]), row)

        for target_iou in args.target_ious:
            values = [row[f"noc_{target_iou:.2f}"] for row in image_level.values()]
            noc_mean, noc_std = mean_std(values)
            success_values = [1.0 if row[f"noc_{target_iou:.2f}"] < args.max_clicks else 0.0 for row in image_level.values()]
            success_mean, _ = mean_std(success_values)
            noc_rows.append({
                "mode": mode_name,
                "target_iou": target_iou,
                "noc_mean": noc_mean,
                "noc_std": noc_std,
                "success_rate": success_mean,
                "num_samples": len(values),
            })

    return fixed_rows, noc_rows


def run_mode(mode_name, mode_cfg, img_paths, mask_dir, AdaptorClass, args, device, output_dir):
    print(f"\n[MODE] {mode_name} | STA={mode_cfg['sta']} | LTM={mode_cfg['ltm']}")
    model = load_ready_model(args.model_path, device)
    adaptor = AdaptorClass(model)

    mode_rows = []
    for run_idx in range(1, args.num_runs + 1):
        adaptor.reset_weights()
        print(f"  Run {run_idx}/{args.num_runs}")

        for img_idx, image_path in enumerate(img_paths, start=1):
            mask_path = match_mask_path(image_path, mask_dir)
            if mask_path is None:
                print(f"  [SKIP] No mask found for {Path(image_path).name}")
                continue

            result = run_one_image(adaptor, image_path, mask_path, mode_cfg, args, device)

            for click_count, metrics in result["fixed_clicks"].items():
                row = {
                    "run": run_idx,
                    "mode": mode_name,
                    "image": result["image"],
                    "image_shape": result["image_shape"],
                    "clicks": click_count,
                    "iou": metrics["iou"],
                    "dice": metrics["dice"],
                    "hd95": metrics["hd95"],
                    "final_iou": result["final_iou"],
                    "final_dice": result["final_dice"],
                    "final_hd95": result["final_hd95"],
                    "num_clicks_generated": result["num_clicks_generated"],
                    "click_latency_ms": result["click_latency_ms"],
                    "ltm_latency_ms": result["ltm_latency_ms"],
                }
                if args.with_noc:
                    for target_iou in args.target_ious:
                        row[f"noc_{target_iou:.2f}"] = result["noc"][target_iou]
                mode_rows.append(row)

            if img_idx % args.log_every == 0 or img_idx == len(img_paths):
                print(f"    {img_idx}/{len(img_paths)} images processed")

    return mode_rows


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Ablation evaluation for Base, Base+STA, Base+LTM, and full OAIS-Net."
    )
    parser.add_argument("--image-dir", required=True, help="Path to test/input images.")
    parser.add_argument("--mask-dir", required=True, help="Path to test/target masks.")
    parser.add_argument("--model-path", required=True, help="Path to trained checkpoint.")
    parser.add_argument("--dataset-name", default=None, help="Name used in output folder.")
    parser.add_argument(
        "--adaptor-module",
        default="pi_seg.inference.oaims_adaptor_ablation",
        help="Python module containing OAIMSAdaptor.",
    )
    parser.add_argument("--output-root", default="test_result", help="Root folder for ablation outputs.")
    parser.add_argument("--target-clicks", default="1,3,5,10,15", help="Fixed click counts for IoU/Dice/HD95.")
    parser.add_argument("--target-ious", default="0.85,0.88,0.90,0.93", help="Target IoUs for NoC when --with-noc is set.")
    parser.add_argument("--max-clicks", type=int, default=15)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--point-slots-per-class", type=int, default=10)
    parser.add_argument("--min-error-area", type=int, default=5)
    parser.add_argument("--sta-rollback", action="store_true", default=True)
    parser.add_argument("--no-sta-rollback", action="store_false", dest="sta_rollback")
    parser.add_argument("--sta-rollback-tolerance", type=float, default=0.0)
    parser.add_argument("--ltm-rollback", action="store_true", default=True)
    parser.add_argument("--no-ltm-rollback", action="store_false", dest="ltm_rollback")
    parser.add_argument("--ltm-iou-tolerance", type=float, default=0.002)
    parser.add_argument("--ltm-dice-tolerance", type=float, default=0.002)
    parser.add_argument("--ltm-hd95-tolerance", type=float, default=1.0)
    parser.add_argument("--ltm-min-accept-iou-gain", type=float, default=0.001)
    parser.add_argument("--ltm-min-accept-dice-gain", type=float, default=0.001)
    parser.add_argument("--ltm-min-accept-hd95-gain", type=float, default=0.5)
    parser.add_argument("--ltm-min-iou", type=float, default=0.65)
    parser.add_argument("--ltm-max-clicks", type=int, default=10)
    parser.add_argument("--pure-ltm-lr", type=float, default=3e-6)
    parser.add_argument("--pure-ltm-steps", type=int, default=1)
    parser.add_argument("--pure-ltm-anchor-weight", type=float, default=1.0)
    parser.add_argument(
        "--ltm-trigger",
        choices=["per-click", "end-of-image"],
        default="per-click",
        help="Fallback LTM trigger. Individual ablation modes can override this.",
    )
    parser.add_argument(
        "--reset-short-term-between-images",
        action="store_true",
        help="Reset decoder/head after each image. Leave off to match the existing oaims_test_iou2.py behavior.",
    )
    parser.add_argument(
        "--with-noc",
        action="store_true",
        help="Also compute NoC summaries. Default is off because module ablation usually only needs fixed-click metrics.",
    )
    parser.add_argument("--device", default=None, choices=[None, "cuda", "cpu", "mps"])
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.target_clicks = parse_int_list(args.target_clicks)
    args.target_ious = parse_float_list(args.target_ious)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset_name or Path(args.image_dir).parent.parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"Ablation_{safe_name(dataset_name)}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = collect_image_paths(args.image_dir)
    if not img_paths:
        raise RuntimeError(f"No images found in: {args.image_dir}")

    AdaptorClass = load_adaptor_class(args.adaptor_module)

    config = vars(args).copy()
    config["target_clicks"] = args.target_clicks
    config["target_ious"] = args.target_ious
    config["device"] = device
    config["num_images"] = len(img_paths)
    with open(output_dir / "ablation_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("OAIS-Net STA/LTM ablation")
    print(f"Dataset: {dataset_name}")
    print(f"Images: {len(img_paths)}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    all_rows = []
    for mode_name, mode_cfg in ABLATION_MODES.items():
        rows = run_mode(mode_name, mode_cfg, img_paths, args.mask_dir, AdaptorClass, args, device, output_dir)
        all_rows.extend(rows)

    noc_fields = [f"noc_{target_iou:.2f}" for target_iou in args.target_ious] if args.with_noc else []
    per_image_fields = [
        "run", "mode", "image", "image_shape", "clicks",
        "iou", "dice", "hd95", "final_iou", "final_dice", "final_hd95",
        "num_clicks_generated", "click_latency_ms", "ltm_latency_ms",
    ] + noc_fields
    write_csv(output_dir / "ablation_per_image.csv", all_rows, per_image_fields)

    fixed_rows, noc_rows = summarize(all_rows, args)
    write_csv(
        output_dir / "ablation_summary_fixed_clicks.csv",
        fixed_rows,
        [
            "mode", "clicks", "iou_mean", "iou_std", "dice_mean", "dice_std",
            "hd95_mean", "hd95_std", "click_latency_ms_mean", "click_latency_ms_std",
            "ltm_latency_ms_mean", "ltm_latency_ms_std", "num_samples",
        ],
    )

    paper_rows, paper_fields, improvement_rows, improvement_fields = build_paper_tables(fixed_rows)
    write_csv(output_dir / "ablation_paper_table.csv", paper_rows, paper_fields)
    write_csv(output_dir / "ablation_improvement_vs_base.csv", improvement_rows, improvement_fields)

    if args.with_noc:
        write_csv(
            output_dir / "ablation_summary_noc.csv",
            noc_rows,
            ["mode", "target_iou", "noc_mean", "noc_std", "success_rate", "num_samples"],
        )

    print("\n[DONE] Ablation finished.")
    print(f"Detailed rows: {output_dir / 'ablation_per_image.csv'}")
    print(f"Fixed-click summary: {output_dir / 'ablation_summary_fixed_clicks.csv'}")
    print(f"Paper-friendly table: {output_dir / 'ablation_paper_table.csv'}")
    print(f"Improvement table: {output_dir / 'ablation_improvement_vs_base.csv'}")
    if args.with_noc:
        print(f"NoC summary: {output_dir / 'ablation_summary_noc.csv'}")


if __name__ == "__main__":
    main()
