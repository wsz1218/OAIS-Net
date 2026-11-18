"""Paired binary segmentation images; split source images before patch extraction."""
from pathlib import Path
import re
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(path))]


def discover_pairs(root):
    root = Path(root)
    def index(folder):
        if not folder.is_dir():
            raise FileNotFoundError(f"Required directory missing: {folder}")
        found = {}
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                if path.stem in found:
                    raise ValueError(f"Duplicate filename stem in {folder}: {path.stem}")
                found[path.stem] = path
        return found
    images, masks = index(root / "input"), index(root / "target")
    if not images:
        raise ValueError(f"No supported images in {root / 'input'}")
    if images.keys() != masks.keys():
        raise ValueError(f"Unpaired files: missing masks={sorted(images.keys()-masks.keys())}, "
                         f"missing images={sorted(masks.keys()-images.keys())}")
    return [(images[k], masks[k]) for k in sorted(images, key=natural_key)]


def read_gray(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f"Expected a single 2D image: {path}")
    return image


def load_pair(pair, max_size=1024):
    image, mask = read_gray(pair[0]), read_gray(pair[1])
    if image.shape != mask.shape:
        raise ValueError(f"Image/mask shapes differ: {pair[0]} {image.shape}, {mask.shape}")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    elif np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image dtype: {image.dtype}")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Floating images must contain finite values in [0, 1].")
    mask = (mask > 0).astype(np.uint8)
    if max_size and max(image.shape) > max_size:
        scale = max_size / max(image.shape)
        size = tuple(max(1, round(v * scale)) for v in image.shape[::-1])
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return image, mask


def image_tensor(image, device):
    h, w = image.shape
    padded = np.pad(image, ((0, (-h) % 32), (0, (-w) % 32)), mode="edge")
    return torch.from_numpy(padded.copy()).float()[None, None].to(device)


def mask_tensor(mask, shape, device):
    h, w = shape
    padded = np.pad(mask, ((0, h-mask.shape[0]), (0, w-mask.shape[1])), mode="edge")
    return torch.from_numpy(padded.copy()).float()[None, None].to(device)


class PatchDataset(Dataset):
    def __init__(self, pairs, crop_size=256, max_size=1024):
        self.pairs, self.crop_size, self.max_size = pairs, crop_size, max_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image, mask = load_pair(self.pairs[index], self.max_size)
        size = self.crop_size
        pad = ((0, max(0, size-image.shape[0])), (0, max(0, size-image.shape[1])))
        image, mask = np.pad(image, pad, mode="edge"), np.pad(mask, pad, mode="edge")
        y, x = random.randint(0, image.shape[0]-size), random.randint(0, image.shape[1]-size)
        image, mask = image[y:y+size, x:x+size], mask[y:y+size, x:x+size]
        turns = random.randrange(4)
        image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        if random.random() < 0.5:
            image, mask = np.fliplr(image), np.fliplr(mask)
        return torch.from_numpy(image.copy()).float()[None], torch.from_numpy(mask.copy()).float()[None]
