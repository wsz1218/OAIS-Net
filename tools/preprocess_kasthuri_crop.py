from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def natural_key(path: Path) -> list[object]:
    parts: list[object] = []
    token = ""
    is_digit = False
    for char in path.stem:
        char_is_digit = char.isdigit()
        if token and char_is_digit != is_digit:
            parts.append(int(token) if is_digit else token.lower())
            token = ""
        token += char
        is_digit = char_is_digit
    if token:
        parts.append(int(token) if is_digit else token.lower())
    return parts


def find_valid_bbox(image: np.ndarray, threshold: int, padding: int) -> tuple[int, int, int, int]:
    valid = image > threshold
    ys, xs = np.where(valid)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("No valid non-black pixels were found.")

    y0 = max(int(ys.min()) - padding, 0)
    x0 = max(int(xs.min()) - padding, 0)
    y1 = min(int(ys.max()) + 1 + padding, image.shape[0])
    x1 = min(int(xs.max()) + 1 + padding, image.shape[1])
    return y0, x0, y1, x1


def preprocess_split(
    source_root: Path,
    output_root: Path,
    source_split: str,
    output_split: str,
    threshold: int,
    padding: int,
    mask_threshold: int,
) -> list[dict[str, object]]:
    image_dir = source_root / f"{source_split}_In"
    mask_dir = source_root / f"{source_split}_Out"
    out_image_dir = output_root / output_split / "input"
    out_mask_dir = output_root / output_split / "target"
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.png"), key=natural_key)
    rows: list[dict[str, object]] = []

    for image_path in image_paths:
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        if image.shape != mask.shape:
            raise ValueError(f"Image/mask shape mismatch for {image_path.name}: {image.shape} vs {mask.shape}")

        y0, x0, y1, x1 = find_valid_bbox(image, threshold=threshold, padding=padding)
        cropped_image = image[y0:y1, x0:x1]
        cropped_mask = (mask[y0:y1, x0:x1] > mask_threshold).astype(np.uint8) * 255

        sample_id = image_path.stem.replace("mask", "")
        out_image_path = out_image_dir / f"x_{sample_id}.tif"
        out_mask_path = out_mask_dir / f"x_{sample_id}.png"

        cv2.imwrite(str(out_image_path), cropped_image)
        cv2.imwrite(str(out_mask_path), cropped_mask)

        original_area = image.shape[0] * image.shape[1]
        crop_area = cropped_image.shape[0] * cropped_image.shape[1]
        rows.append(
            {
                "split": output_split,
                "source_file": image_path.name,
                "output_image": str(out_image_path.relative_to(output_root)),
                "output_mask": str(out_mask_path.relative_to(output_root)),
                "original_h": image.shape[0],
                "original_w": image.shape[1],
                "crop_y0": y0,
                "crop_x0": x0,
                "crop_y1": y1,
                "crop_x1": x1,
                "crop_h": cropped_image.shape[0],
                "crop_w": cropped_image.shape[1],
                "crop_area_ratio": round(crop_area / original_area, 6),
                "foreground_ratio": round(float((cropped_mask > 0).mean()), 6),
            }
        )

    return rows


def write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    manifest_path = output_root / "crop_manifest.csv"
    if not rows:
        return

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop invalid black padding from Kasthuri++ images and masks."
    )
    parser.add_argument(
        "--source-root",
        default="pi_seg/data/datasets/kasthuri_pp/Kasthuri++",
        type=Path,
        help="Directory containing Train_In, Train_Out, Test_In, and Test_Out.",
    )
    parser.add_argument(
        "--output-root",
        default="pi_seg/data/datasets/kasthuri_pp_cropped",
        type=Path,
        help="Output dataset root. Creates train/test input/target folders.",
    )
    parser.add_argument("--threshold", default=5, type=int, help="Pixel threshold for valid EM region.")
    parser.add_argument("--padding", default=8, type=int, help="Padding around the detected valid bbox.")
    parser.add_argument("--mask-threshold", default=127, type=int, help="Threshold used to binarize masks.")
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only preprocess Test_In/Test_Out into test/input and test/target.",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    if not args.test_only:
        rows.extend(
            preprocess_split(
                args.source_root,
                args.output_root,
                source_split="Train",
                output_split="train",
                threshold=args.threshold,
                padding=args.padding,
                mask_threshold=args.mask_threshold,
            )
        )

    rows.extend(
        preprocess_split(
            args.source_root,
            args.output_root,
            source_split="Test",
            output_split="test",
            threshold=args.threshold,
            padding=args.padding,
            mask_threshold=args.mask_threshold,
        )
    )
    write_manifest(args.output_root, rows)

    print(f"Processed {len(rows)} samples into {args.output_root}")
    print(f"Manifest: {args.output_root / 'crop_manifest.csv'}")


if __name__ == "__main__":
    main()
