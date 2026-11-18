from __future__ import annotations

import argparse
import csv
import re
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
H5_EXTS = {".h5", ".hdf5", ".hdf"}


def natural_key(path: Path) -> list[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.stem)]


def slice_index(path: Path) -> int | None:
    digits = re.findall(r"\d+", path.stem)
    return int(digits[-1]) if digits else None


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array.astype(np.float32) - min_value) / (max_value - min_value)
    return np.clip(np.round(scaled * 255), 0, 255).astype(np.uint8)


def read_2d_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return normalize_to_uint8(image)


def resize_image(image: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    resized = cv2.resize(image, (size, size), interpolation=interpolation)
    if is_mask:
        resized = (resized > 0).astype(np.uint8) * 255
    return resized


def label_entries(label_zip: Path, split: str) -> list[tuple[Path, str]]:
    dirname = "mito-train-v2" if split == "train" else "mito-val-v2"
    with zipfile.ZipFile(label_zip) as zf:
        members = [
            name
            for name in zf.namelist()
            if name.startswith(f"{dirname}/") and Path(name).name.startswith("seg") and name.lower().endswith(".tif")
        ]
    members = sorted(members, key=lambda name: natural_key(Path(name)))
    if not members:
        raise FileNotFoundError(f"No {dirname}/seg*.tif labels found in {label_zip}")
    return [(label_zip, member) for member in members]


def label_entry_index(entry: tuple[Path, str]) -> int | None:
    return slice_index(Path(entry[1]))


def read_label_entry(entry: tuple[Path, str]) -> np.ndarray:
    zip_path, member = entry
    with zipfile.ZipFile(zip_path) as zf:
        data = np.frombuffer(zf.read(member), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to decode {member} from {zip_path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return normalize_to_uint8(image)


def h5_datasets(path: Path) -> list[tuple[str, tuple[int, ...]]]:
    datasets: list[tuple[str, tuple[int, ...]]] = []
    with h5py.File(path, "r") as f:
        def visit(name: str, obj: object) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets.append((name, tuple(obj.shape)))
        f.visititems(visit)
    return datasets


def choose_h5_key(path: Path, explicit_key: str | None) -> str:
    datasets = h5_datasets(path)
    if explicit_key:
        keys = {name for name, _ in datasets}
        if explicit_key not in keys:
            raise KeyError(f"H5 key {explicit_key!r} not found in {path}. Available: {datasets[:20]}")
        return explicit_key

    candidates = [
        name
        for name, shape in datasets
        if len(shape) >= 3 and any(token in name.lower() for token in ("raw", "image", "em", "data"))
    ]
    if not candidates:
        candidates = [name for name, shape in datasets if len(shape) >= 3]
    if not candidates:
        raise RuntimeError(f"No 3D image dataset found in {path}. Available: {datasets[:20]}")
    return candidates[0]


class RawVolume:
    def __init__(self, raw_path: Path, h5_key: str | None = None):
        self.raw_path = raw_path
        self.h5_key = h5_key
        self._image_files: dict[int, Path] | None = None
        self._h5_file: h5py.File | None = None
        self._dataset = None

        if raw_path.is_dir():
            self._image_files = {}
            for path in raw_path.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    idx = slice_index(path)
                    if idx is not None:
                        self._image_files[idx] = path
            if not self._image_files:
                raise FileNotFoundError(f"No image files with numeric names found under {raw_path}")
        elif raw_path.suffix.lower() in H5_EXTS:
            self._h5_file = h5py.File(raw_path, "r")
            key = choose_h5_key(raw_path, h5_key)
            self._dataset = self._h5_file[key]
        else:
            raise ValueError(f"Unsupported raw image source: {raw_path}")

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()

    def read_slice(self, idx: int) -> np.ndarray:
        if self._image_files is not None:
            path = self._image_files.get(idx)
            if path is None:
                raise FileNotFoundError(f"No raw image slice {idx} found under {self.raw_path}")
            return read_2d_image(path)

        assert self._dataset is not None
        volume = self._dataset
        if idx >= volume.shape[0] and idx - 1 >= 0:
            idx = idx - 1
        image = np.asarray(volume[idx])
        if image.ndim == 3:
            image = image[..., 0]
        return normalize_to_uint8(image)


def discover_raw_source(raw_root: Path, domain: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit

    domain_upper = domain.upper()
    candidates: list[Path] = []
    for path in raw_root.rglob("*"):
        if not path.is_file() and not path.is_dir():
            continue
        name = path.name.lower()
        text = str(path).lower()
        if domain_upper == "H":
            domain_hit = any(token in text for token in ("em30-h", "mitoem-h", "mito-h", "_h", "-h"))
        else:
            domain_hit = any(token in text for token in ("em30-r", "mitoem-r", "mito-r", "_r", "-r"))
        if not domain_hit:
            continue
        if path.is_file() and path.suffix.lower() in H5_EXTS:
            candidates.append(path)
        elif path.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in path.rglob("*")):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find raw MitoEM-{domain_upper} images under {raw_root}. "
            "The downloaded mito-train/mito-val zip files contain labels only. "
            f"Please put a raw image .h5/.hdf5 file or an image-sequence folder for MitoEM-{domain_upper} under {raw_root}, "
            f"or pass --raw-{domain.lower()} explicitly."
        )
    return sorted(candidates, key=lambda p: (p.is_dir(), len(str(p)), str(p)))[0]


def write_split(
    labels: list[tuple[Path, str]],
    raw_volume: RawVolume,
    output_root: Path,
    split: str,
    prefix: str,
    size: int,
    max_count: int | None,
    skip_empty_masks: bool,
) -> list[dict[str, object]]:
    input_dir = output_root / split / "input"
    target_dir = output_root / split / "target"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    exported = 0
    for label_entry in labels:
        if max_count is not None and exported >= max_count:
            break

        idx = label_entry_index(label_entry)
        if idx is None:
            continue
        mask = read_label_entry(label_entry)
        mask = (mask > 0).astype(np.uint8) * 255
        if skip_empty_masks and not np.any(mask > 0):
            continue

        image = raw_volume.read_slice(idx)
        if image.shape != mask.shape:
            image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)

        image_out = resize_image(image, size=size, is_mask=False)
        mask_out = resize_image(mask, size=size, is_mask=True)

        sample_name = f"{prefix}_{idx:04d}"
        out_image_path = input_dir / f"{sample_name}.tif"
        out_mask_path = target_dir / f"{sample_name}.png"
        cv2.imwrite(str(out_image_path), image_out)
        cv2.imwrite(str(out_mask_path), mask_out)

        rows.append(
            {
                "split": split,
                "sample": sample_name,
                "source_raw": str(raw_volume.raw_path),
                "source_label": f"{label_entry[0]}::{label_entry[1]}",
                "slice_index": idx,
                "original_h": mask.shape[0],
                "original_w": mask.shape[1],
                "foreground_ratio": round(float((mask_out > 0).mean()), 6),
            }
        )
        exported += 1
    return rows


def write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with (output_root / "mitoem_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MitoEM-H -> MitoEM-R in DeepPI-EM dataset format.")
    parser.add_argument("--raw-root", type=Path, default=Path("pi_seg/data/datasets/mitoem_raw"))
    parser.add_argument("--output-root", type=Path, default=Path("pi_seg/data/datasets/mitoem_h_to_r"))
    parser.add_argument("--raw-h", type=Path, default=None, help="MitoEM-H raw image H5 file or image folder.")
    parser.add_argument("--raw-r", type=Path, default=None, help="MitoEM-R raw image H5 file or image folder.")
    parser.add_argument("--h5-key-h", default=None, help="H5 dataset key for MitoEM-H raw images.")
    parser.add_argument("--h5-key-r", default=None, help="H5 dataset key for MitoEM-R raw images.")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--max-train", type=int, default=400)
    parser.add_argument("--max-test", type=int, default=100)
    parser.add_argument("--test-label-split", choices=("train", "val"), default="val")
    parser.add_argument("--skip-empty-masks", action="store_true")
    args = parser.parse_args()

    h_zip = args.raw_root / "EM30-H-mito-train-val-v2.zip"
    r_zip = args.raw_root / "EM30-R-mito-train-val-v2.zip"
    if not h_zip.exists():
        raise FileNotFoundError(f"Missing label zip: {h_zip}")
    if not r_zip.exists():
        raise FileNotFoundError(f"Missing label zip: {r_zip}")

    h_raw_source = discover_raw_source(args.raw_root, "H", args.raw_h)
    r_raw_source = discover_raw_source(args.raw_root, "R", args.raw_r)
    print(f"MitoEM-H raw source: {h_raw_source}")
    print(f"MitoEM-R raw source: {r_raw_source}")

    h_raw = RawVolume(h_raw_source, h5_key=args.h5_key_h)
    r_raw = RawVolume(r_raw_source, h5_key=args.h5_key_r)
    try:
        rows: list[dict[str, object]] = []
        rows.extend(
            write_split(
                labels=label_entries(h_zip, "train"),
                raw_volume=h_raw,
                output_root=args.output_root,
                split="train",
                prefix="mitoem_h",
                size=args.size,
                max_count=args.max_train,
                skip_empty_masks=args.skip_empty_masks,
            )
        )
        rows.extend(
            write_split(
                labels=label_entries(r_zip, args.test_label_split),
                raw_volume=r_raw,
                output_root=args.output_root,
                split="test",
                prefix="mitoem_r",
                size=args.size,
                max_count=args.max_test,
                skip_empty_masks=args.skip_empty_masks,
            )
        )
        write_manifest(args.output_root, rows)
    finally:
        h_raw.close()
        r_raw.close()

    train_count = len(list((args.output_root / "train" / "input").glob("*.tif")))
    test_count = len(list((args.output_root / "test" / "input").glob("*.tif")))
    print(f"Exported train/input: {train_count}")
    print(f"Exported test/input : {test_count}")
    print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()
