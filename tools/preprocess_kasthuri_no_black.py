from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def natural_key(path: Path) -> list[object]:
    import re

    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.stem)]


def largest_rectangle_in_binary(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return y0, x0, y1, x1 for the largest all-True rectangle."""
    heights = np.zeros(mask.shape[1], dtype=np.int32)
    best_area = 0
    best = (0, 0, mask.shape[0], mask.shape[1])

    for y in range(mask.shape[0]):
        heights = np.where(mask[y], heights + 1, 0)
        stack: list[int] = []

        for x in range(mask.shape[1] + 1):
            current_height = heights[x] if x < mask.shape[1] else 0
            while stack and current_height < heights[stack[-1]]:
                h = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                width = x - left
                area = h * width
                if area > best_area:
                    best_area = area
                    y1 = y + 1
                    y0 = y1 - h
                    best = (y0, left, y1, x)
            stack.append(x)

    return best


def valid_imaging_mask(image: np.ndarray, black_threshold: int, close_kernel: int) -> np.ndarray:
    valid = image > black_threshold
    kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
    valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    return valid


def crop_no_black(
    image: np.ndarray,
    mask: np.ndarray,
    black_threshold: int,
    close_kernel: int,
    safety_margin: int,
    min_side: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], float]:
    valid = valid_imaging_mask(image, black_threshold=black_threshold, close_kernel=close_kernel)
    y0, x0, y1, x1 = largest_rectangle_in_binary(valid)

    y0 += safety_margin
    x0 += safety_margin
    y1 -= safety_margin
    x1 -= safety_margin

    if y1 - y0 < min_side or x1 - x0 < min_side:
        raise ValueError(f"Crop too small after black removal: {(y0, x0, y1, x1)}")

    cropped_image = image[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1]
    black_ratio = float((cropped_image <= black_threshold).mean())
    return cropped_image, cropped_mask, (y0, x0, y1, x1), black_ratio


def process_split(
    source_root: Path,
    output_root: Path,
    source_split: str,
    output_split: str,
    black_threshold: int,
    close_kernel: int,
    safety_margin: int,
    min_side: int,
    output_size: int,
) -> list[dict[str, object]]:
    in_dir = source_root / f"{source_split}_In"
    out_dir = source_root / f"{source_split}_Out"
    output_input = output_root / output_split / "input"
    output_target = output_root / output_split / "target"
    output_input.mkdir(parents=True, exist_ok=True)
    output_target.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for image_path in sorted(in_dir.glob("*.png"), key=natural_key):
        mask_path = out_dir / image_path.name
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        if image.shape != mask.shape:
            raise ValueError(f"Shape mismatch for {image_path.name}: {image.shape} vs {mask.shape}")

        mask = (mask > 127).astype(np.uint8) * 255
        cropped_image, cropped_mask, bbox, black_ratio = crop_no_black(
            image=image,
            mask=mask,
            black_threshold=black_threshold,
            close_kernel=close_kernel,
            safety_margin=safety_margin,
            min_side=min_side,
        )

        resized_image = cv2.resize(cropped_image, (output_size, output_size), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(cropped_mask, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
        resized_mask = (resized_mask > 127).astype(np.uint8) * 255

        sample_id = image_path.stem.replace("mask", "")
        out_image_path = output_input / f"x_{sample_id}.tif"
        out_mask_path = output_target / f"x_{sample_id}.png"
        cv2.imwrite(str(out_image_path), resized_image)
        cv2.imwrite(str(out_mask_path), resized_mask)

        y0, x0, y1, x1 = bbox
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
                "crop_h": y1 - y0,
                "crop_w": x1 - x0,
                "black_ratio_after_crop": round(black_ratio, 6),
                "foreground_ratio": round(float((resized_mask > 0).mean()), 6),
            }
        )

    return rows


def write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    with (output_root / "no_black_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Kasthuri++ split with black padding removed.")
    parser.add_argument("--source-root", type=Path, default=Path("pi_seg/data/datasets/kasthuri_pp/Kasthuri++"))
    parser.add_argument("--output-root", type=Path, default=Path("pi_seg/data/datasets/kasthuri_pp_no_black"))
    parser.add_argument("--black-threshold", type=int, default=5)
    parser.add_argument("--close-kernel", type=int, default=9)
    parser.add_argument("--safety-margin", type=int, default=6)
    parser.add_argument("--min-side", type=int, default=256)
    parser.add_argument("--output-size", type=int, default=1024)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--test-split-name", default="kasthuri_test")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    if not args.test_only:
        rows.extend(
            process_split(
                args.source_root,
                args.output_root,
                "Train",
                "train",
                args.black_threshold,
                args.close_kernel,
                args.safety_margin,
                args.min_side,
                args.output_size,
            )
        )
    rows.extend(
        process_split(
            args.source_root,
            args.output_root,
            "Test",
            args.test_split_name,
            args.black_threshold,
            args.close_kernel,
            args.safety_margin,
            args.min_side,
            args.output_size,
        )
    )
    write_manifest(args.output_root, rows)
    print(f"Processed {len(rows)} samples into {args.output_root}")
    print(f"Manifest: {args.output_root / 'no_black_manifest.csv'}")


if __name__ == "__main__":
    main()
