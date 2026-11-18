from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import h5py
import numpy as np


DOMAIN_FILES = {
    "330kx_80e": (
        "Au_5nm_330kx_80e_Std_UTC_FFCorr_Team05_Images.h5",
        "Au_5nm_330kx_80e_Std_UTC_FFCorr_Team05_Labels.h5",
    ),
    "330kx_884e": (
        "Au_5nm_330kx_884e_Std_UTC_FFCorr_Team05_Images.h5",
        "Au_5nm_330kx_884e_Std_UTC_FFCorr_Team05_Labels.h5",
    ),
    "330kx_421e": (
        "Au_5nm_330kx_421e_UTC_Team05_Std_Sess1109FFCorr_Images.h5",
        "Au_5nm_330kx_421e_UTC_Team05_Std_Sess1109FFCorr_Labels.h5",
    ),
    "330kx_423e": (
        "Au_5nm_330kx_423e_UTC_Team05_Std_Sess0821FFCorr_Images.h5",
        "Au_5nm_330kx_423e_UTC_Team05_Std_Sess0821FFCorr_Labels.h5",
    ),
    "330kx_423e_sess1006": (
        "Au_5nm_330kx_423e_UTC_Team05_Std_Sess1006FFCorr_Images.h5",
        "Au_5nm_330kx_423e_UTC_Team05_Std_Sess1006FFCorr_Labels.h5",
    ),
    "330kx_425e": (
        "Au_5nm_330kx_425e_UTC_Team05_Std_Sess1025FFCorr_Images.h5",
        "Au_5nm_330kx_425e_UTC_Team05_Std_Sess1025FFCorr_Labels.h5",
    ),
    "260kx_450e": (
        "Au_5nm_260kx_450e_Std_UTC_FFCorr_Team05_Images.h5",
        "Au_5nm_260kx_450e_Std_UTC_FFCorr_Team05_Labels.h5",
    ),
    "205kx_420e": (
        "Au_5nm_205kx_420e_Std_UTC_FFCorr_Team05_Images.h5",
        "Au_5nm_205kx_420e_Std_UTC_FFCorr_Team05_Labels.h5",
    ),
    "160kx_425e": (
        "Au_5nm_160kx_425e_Std_UTC_FFCorr_Team05_Images.h5",
        "Au_5nm_160kx_425e_Std_UTC_FFCorr_Team05_Labels.h5",
    ),
}


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array
    lo = float(np.min(array))
    hi = float(np.max(array))
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array.astype(np.float32) - lo) / (hi - lo)
    return np.clip(np.round(scaled * 255), 0, 255).astype(np.uint8)


def parse_domain(domain: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)kx_(\d+)e", domain)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def read_h5_dataset(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if key not in f:
            available = list(f.keys())
            raise KeyError(f"Key {key!r} not found in {path}. Available top-level keys: {available}")
        return np.asarray(f[key])


def to_slices(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim == 3:
        return array
    if array.ndim == 4 and array.shape[-1] == 1:
        return array[..., 0]
    raise ValueError(f"Unsupported H5 array shape: {array.shape}")


def resize_pair(image: np.ndarray, label: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    image = normalize_to_uint8(image)
    label = (label > 0).astype(np.uint8) * 255
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    label = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
    label = (label > 127).astype(np.uint8) * 255
    return image, label


def export_domain(
    raw_root: Path,
    output_root: Path,
    domain: str,
    split: str,
    image_key: str,
    label_key: str,
    size: int,
    max_slices: int | None,
    source_start: int,
    start_index: int,
    prefix: str,
) -> list[dict[str, object]]:
    image_name, label_name = DOMAIN_FILES[domain]
    image_path = raw_root / image_name
    label_path = raw_root / label_name
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image H5 for {domain}: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label H5 for {domain}: {label_path}")

    images = to_slices(read_h5_dataset(image_path, image_key))
    labels = to_slices(read_h5_dataset(label_path, label_key))
    if len(images) != len(labels):
        raise ValueError(f"Slice count mismatch for {domain}: images={len(images)}, labels={len(labels)}")

    out_input = output_root / split / "input"
    out_target = output_root / split / "target"
    out_input.mkdir(parents=True, exist_ok=True)
    out_target.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    if source_start < 0:
        raise ValueError(f"source_start must be >= 0, got {source_start}")
    available = max(0, len(images) - source_start)
    count = available if max_slices is None else min(available, max_slices)
    if count <= 0:
        raise ValueError(
            f"No slices available for {domain} split={split}: "
            f"total={len(images)}, source_start={source_start}, max_slices={max_slices}"
        )
    magnification, dose = parse_domain(domain)

    for local_idx in range(count):
        source_idx = source_start + local_idx
        image, label = resize_pair(images[source_idx], labels[source_idx], size)
        sample_idx = start_index + local_idx
        sample_name = f"{prefix}_{sample_idx:04d}"
        out_image = out_input / f"{sample_name}.tif"
        out_label = out_target / f"{sample_name}.png"
        cv2.imwrite(str(out_image), image)
        cv2.imwrite(str(out_label), label)

        rows.append(
            {
                "split": split,
                "sample": sample_name,
                "domain": domain,
                "magnification_kx": magnification,
                "electron_dose_e": dose,
                "source_slice": source_idx,
                "source_image_h5": image_name,
                "source_label_h5": label_name,
                "foreground_ratio": round(float((label > 0).mean()), 6),
                "image_mean": round(float(image.mean()), 6),
                "image_std": round(float(image.std()), 6),
            }
        )

    return rows


def write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with (output_root / "au_generalization_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Dryad Au nanoparticle H5 files to DeepPI-EM format.")
    parser.add_argument("--raw-root", type=Path, default=Path("pi_seg/data/datasets/Au_generalization_raw"))
    parser.add_argument("--output-root", type=Path, default=Path("pi_seg/data/datasets/Au_generalization_330kx_80e_source"))
    parser.add_argument("--source-domain", default="330kx_80e", choices=sorted(DOMAIN_FILES))
    parser.add_argument(
        "--target-domains",
        nargs="+",
        default=["330kx_80e", "330kx_884e", "260kx_450e", "205kx_420e", "160kx_425e"],
        choices=sorted(DOMAIN_FILES),
    )
    parser.add_argument("--image-key", default="images")
    parser.add_argument("--label-key", default="labels")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--max-train", type=int, default=80)
    parser.add_argument("--max-test-per-domain", type=int, default=30)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument(
        "--source-test-start",
        type=int,
        default=None,
        help="Start slice for the same-domain source test split. Defaults to train_start + max_train.",
    )
    parser.add_argument("--target-test-start", type=int, default=0)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    rows.extend(
        export_domain(
            raw_root=args.raw_root,
            output_root=args.output_root,
            domain=args.source_domain,
            split="train",
            image_key=args.image_key,
            label_key=args.label_key,
            size=args.size,
            max_slices=args.max_train,
            source_start=args.train_start,
            start_index=0,
            prefix=f"au_{args.source_domain}",
        )
    )

    sample_offset = 0
    for domain in args.target_domains:
        split = f"test_{domain}"
        source_start = (
            args.source_test_start
            if domain == args.source_domain and args.source_test_start is not None
            else args.train_start + args.max_train
            if domain == args.source_domain
            else args.target_test_start
        )
        rows.extend(
            export_domain(
                raw_root=args.raw_root,
                output_root=args.output_root,
                domain=domain,
                split=split,
                image_key=args.image_key,
                label_key=args.label_key,
                size=args.size,
                max_slices=args.max_test_per_domain,
                source_start=source_start,
                start_index=sample_offset,
                prefix=f"au_{domain}",
            )
        )
        sample_offset += args.max_test_per_domain

    write_manifest(args.output_root, rows)
    print(f"Exported {len(rows)} samples into {args.output_root}")
    print(f"Manifest: {args.output_root / 'au_generalization_manifest.csv'}")


if __name__ == "__main__":
    main()
