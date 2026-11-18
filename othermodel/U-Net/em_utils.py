import csv
import glob
import math
import random
import re
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_sort_key(value):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"([0-9]+)", str(value))
    ]


def collect_image_paths(image_dir):
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(str(Path(image_dir) / f"*{ext}")))
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


def load_gray_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    return to_uint8_gray(image)


def load_binary_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {path}")
    return (mask > 127).astype(np.uint8)


def random_crop_pair(image, mask, crop_size, foreground_prob=0.7):
    h, w = image.shape[:2]
    if crop_size <= 0 or (h == crop_size and w == crop_size):
        return image, mask
    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        h, w = image.shape[:2]
    foreground = np.argwhere(mask > 0)
    if len(foreground) > 0 and random.random() < foreground_prob:
        cy, cx = foreground[random.randrange(len(foreground))]
        top = int(np.clip(cy - random.randint(0, crop_size - 1), 0, h - crop_size))
        left = int(np.clip(cx - random.randint(0, crop_size - 1), 0, w - crop_size))
    else:
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
    return image[top:top + crop_size, left:left + crop_size], mask[top:top + crop_size, left:left + crop_size]


def augment_pair(image, mask):
    if random.random() < 0.5:
        image = np.ascontiguousarray(np.fliplr(image))
        mask = np.ascontiguousarray(np.fliplr(mask))
    if random.random() < 0.5:
        image = np.ascontiguousarray(np.flipud(image))
        mask = np.ascontiguousarray(np.flipud(mask))
    angle = random.randint(-180, 180)
    while abs(angle) < 5:
        angle = random.randint(-90, 90)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask


def sample_training_clicks(mask, max_points=15):
    positive_pixels = np.argwhere(mask > 0)
    negative_pixels = np.argwhere(mask == 0)
    clicks = []

    if len(positive_pixels) > 0:
        num_pos = random.randint(1, max(1, max_points // 2))
        for _ in range(num_pos):
            y, x = positive_pixels[random.randrange(len(positive_pixels))]
            clicks.append((int(y), int(x), 1))

    remaining = max(0, max_points - len(clicks))
    if remaining > 0 and len(negative_pixels) > 0:
        num_neg = random.randint(0, remaining)
        for _ in range(num_neg):
            y, x = negative_pixels[random.randrange(len(negative_pixels))]
            clicks.append((int(y), int(x), 0))

    random.shuffle(clicks)
    return clicks


def encode_click_maps(shape, clicks, sigma=10.0):
    h, w = shape
    pos = np.zeros((h, w), dtype=np.float32)
    neg = np.zeros((h, w), dtype=np.float32)
    if not clicks:
        return pos, neg

    yy, xx = np.mgrid[0:h, 0:w]
    denom = 2.0 * sigma * sigma
    for y, x, is_positive in clicks:
        gaussian = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / denom).astype(np.float32)
        if is_positive:
            pos = np.maximum(pos, gaussian)
        else:
            neg = np.maximum(neg, gaussian)
    return pos, neg


class EMClickDataset(Dataset):
    def __init__(self, image_dir, mask_dir, crop_size=512, max_points=15, sigma=10.0, augment=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.crop_size = crop_size
        self.max_points = max_points
        self.sigma = sigma
        self.augment = augment
        self.samples = []
        for image_path in collect_image_paths(self.image_dir):
            mask_path = match_mask_path(image_path, self.mask_dir)
            if mask_path is not None:
                self.samples.append((Path(image_path), Path(mask_path)))
        if not self.samples:
            raise RuntimeError(f"No image/mask pairs found in {image_dir} and {mask_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = load_gray_image(image_path)
        mask = load_binary_mask(mask_path)
        image, mask = random_crop_pair(image, mask, self.crop_size)
        if self.augment:
            image, mask = augment_pair(image, mask)

        clicks = sample_training_clicks(mask, max_points=self.max_points)
        pos_map, neg_map = encode_click_maps(mask.shape, clicks, sigma=self.sigma)
        image_f = image.astype(np.float32) / 255.0
        x = np.stack([image_f, pos_map, neg_map], axis=0).astype(np.float32)
        y = mask[None].astype(np.float32)
        return {
            "image": torch.from_numpy(x),
            "mask": torch.from_numpy(y),
            "name": image_path.name,
        }


class EMImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, crop_size=512, augment=True):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.crop_size = crop_size
        self.augment = augment
        self.samples = []
        for image_path in collect_image_paths(self.image_dir):
            mask_path = match_mask_path(image_path, self.mask_dir)
            if mask_path is not None:
                self.samples.append((Path(image_path), Path(mask_path)))
        if not self.samples:
            raise RuntimeError(f"No image/mask pairs found in {image_dir} and {mask_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = load_gray_image(image_path)
        mask = load_binary_mask(mask_path)
        image, mask = random_crop_pair(image, mask, self.crop_size)
        if self.augment:
            image, mask = augment_pair(image, mask)

        image_f = image.astype(np.float32) / 255.0
        return {
            "image": torch.from_numpy(image_f[None].astype(np.float32)),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
            "name": image_path.name,
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


def pad_to_multiple(tensor, multiple=16):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (h, w)
    tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, (h, w)


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


def current_metrics(pred_mask, gt_mask):
    return {
        "iou": calculate_iou(pred_mask, gt_mask),
        "dice": calculate_dice(pred_mask, gt_mask),
    }


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or [])
        writer.writeheader()
        writer.writerows(rows)


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

    for y, x, is_positive in clicks:
        color = (0, 255, 0) if is_positive else (255, 0, 0)
        cv2.circle(canvas, (x, y), 4, color, -1)
        cv2.circle(canvas, (x, y), 8, color, 1)
        cv2.circle(overlay, (x, y), 4, color, -1)
        cv2.circle(overlay, (x, y), 8, color, 1)

    gt_rgb = np.stack([gt_mask * 255] * 3, axis=-1).astype(np.uint8)
    panel = np.concatenate([canvas, gt_rgb, np.clip(overlay, 0, 255).astype(np.uint8)], axis=1)
    cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def save_paper_visual(image_gray, gt_mask, pred_mask, clicks, output_path, title, iou_score=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    for y, x, is_positive in clicks:
        color = (0, 255, 0) if is_positive == 1 else (255, 0, 0)
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


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]
