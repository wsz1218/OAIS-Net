"""Click simulation, image-level metrics, and sequential evaluation."""
import cv2
import numpy as np
import torch
from scipy.ndimage import label

from seg.dataio import load_pair, image_tensor


def empty_points(batch, max_clicks, device):
    return torch.full((batch, 2 * max_clicks, 3), -1.0, device=device)


def next_click(pred, target, clicked):
    error = (pred != target).astype(np.uint8)
    for y, x, _ in clicked:
        error[y, x] = 0
    regions, count = label(error)
    if not count:
        return None
    sizes = np.bincount(regions.ravel())
    sizes[0] = 0
    region = (regions == sizes.argmax()).astype(np.uint8)
    # Zero padding makes distance-to-boundary defined even for an all-error image.
    distance = cv2.distanceTransform(np.pad(region, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(distance.argmax(), distance.shape)
    return int(y), int(x), int(target[y, x] > 0)


def add_click(points, click, sample=0):
    y, x, positive = click
    half = points.shape[1] // 2
    start = 0 if positive else half
    for slot in range(start, start + half):
        if points[sample, slot, 0] < 0:
            points[sample, slot] = points.new_tensor([y, x, 1.0])
            return
    raise ValueError("Click tensor is full; increase max-clicks/point capacity.")


def metrics(pred, target):
    pred, target = pred.astype(bool), target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union, total = np.logical_or(pred, target).sum(), pred.sum() + target.sum()
    return {"iou": float(intersection / union) if union else 1.0,
            "dice": float(2 * intersection / total) if total else 1.0}


@torch.no_grad()
def prediction(model, image, points):
    model.eval()
    return torch.sigmoid(model(image, points, training=False)["instances"]) > 0.5


def evaluate_sequence(model, pairs, device, max_clicks=15, mode="base", adaptor=None,
                      target_iou=None, max_size=1024, save_callback=None):
    """target_iou=None runs fixed clicks; a threshold runs a separate capped-NoC sequence."""
    if mode != "base" and adaptor is None:
        raise ValueError("Online modes require an adaptor.")
    rows = []
    for index, pair in enumerate(pairs, 1):
        image_np, target = load_pair(pair, max_size)
        image = image_tensor(image_np, device)
        h, w = target.shape
        points = empty_points(1, max_clicks, device)
        clicks, curve = [], []
        if adaptor is not None:
            adaptor.clear_momentum()
        final = prediction(model, image, points)
        for step in range(max_clicks + 1):
            mask = final[0, 0, :h, :w].cpu().numpy()
            score = metrics(mask, target)
            curve.append({"clicks": step, **score})
            if step == max_clicks or (target_iou is not None and score["iou"] >= target_iou):
                break
            click = next_click(mask, target, clicks)
            if click is None:
                break
            clicks.append(click)
            add_click(points, click)
            if mode in ("sta", "oais"):
                adaptor.adapt_MI(image, points, clicks)
            final = prediction(model, image, points)
        attained = None if target_iou is None else score["iou"] >= target_iou
        row = {"image": pair[0].name, "sequence_index": index, "zero_iou": curve[0]["iou"],
               "actual_clicks": len(clicks), "final_iou": score["iou"], "final_dice": score["dice"],
               "target_attained": attained,
               "noc": None if target_iou is None else len(clicks) if attained else max_clicks,
               "curve": curve, "ltm_updates": 0}
        if save_callback:
            save_callback(pair[0].stem, mask)
        if mode in ("ltm", "oais") and clicks:
            # Use the full padded model output, not target, for the next image's adaptation.
            adaptor.adapt_ltm(image, points, final)
            row["ltm_updates"] = 1
        rows.append(row)
        print(f"[{index}/{len(pairs)}] {pair[0].name}: clicks={len(clicks)}, IoU={score['iou']:.4f}", flush=True)
    return rows
