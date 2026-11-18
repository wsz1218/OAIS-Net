from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def load_json_from_zip(zip_file: zipfile.ZipFile, member: str) -> dict:
    return json.loads(zip_file.read(member).decode("utf-8"))


def read_image_from_zip(zip_file: zipfile.ZipFile, member: str) -> np.ndarray:
    image = Image.open(BytesIO(zip_file.read(member))).convert("L")
    return np.array(image)


def annotation_to_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    segmentation = annotation.get("segmentation")
    if not segmentation:
        return np.zeros((height, width), dtype=np.uint8)

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation

    return mask_utils.decode(rle).astype(np.uint8)


def resize_pair(image: np.ndarray, mask: np.ndarray, output_size: int) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 255
    return image, mask


def convert_split(
    zip_path: Path,
    split: str,
    output_root: Path,
    output_size: int,
    max_images: int | None,
) -> list[dict[str, object]]:
    output_input = output_root / "input"
    output_target = output_root / "target"
    output_input.mkdir(parents=True, exist_ok=True)
    output_target.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as zf:
        coco = load_json_from_zip(zf, f"labels/{split}.json")
        images = sorted(coco["images"], key=lambda item: item["id"])
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            annotations_by_image[ann["image_id"]].append(ann)

        if max_images is not None:
            images = images[:max_images]

        for out_idx, image_info in enumerate(images):
            file_name = image_info["file_name"]
            image_member = file_name if file_name.startswith(f"{split}/") else f"{split}/{file_name}"
            image = read_image_from_zip(zf, image_member)
            height, width = int(image_info["height"]), int(image_info["width"])

            mask = np.zeros((height, width), dtype=np.uint8)
            for ann in annotations_by_image.get(image_info["id"], []):
                mask |= annotation_to_mask(ann, height, width)

            image, mask = resize_pair(image, mask, output_size)
            sample_name = f"x_{out_idx:04d}"
            out_image_path = output_input / f"{sample_name}.tif"
            out_mask_path = output_target / f"{sample_name}.png"
            cv2.imwrite(str(out_image_path), image)
            cv2.imwrite(str(out_mask_path), mask)

            rows.append(
                {
                    "sample": sample_name,
                    "split": split,
                    "source_file": image_member,
                    "image_id": image_info["id"],
                    "annotations": len(annotations_by_image.get(image_info["id"], [])),
                    "original_h": height,
                    "original_w": width,
                    "foreground_ratio": round(float((mask > 0).mean()), 6),
                }
            )

    return rows


def write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with (output_root / "hela_coco_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HeLa COCO annotations into DeepPI-EM input/target format.")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("pi_seg/data/datasets/Skeletal_muscle_test/domain_2_cells/hela_cells_coco.zip"),
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="COCO split to export. Use val for external generalization testing.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("pi_seg/data/datasets/Skeletal_muscle_test/domain_2_cells/test_fanhua"),
    )
    parser.add_argument("--output-size", type=int, default=1024)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    rows = convert_split(
        zip_path=args.zip_path,
        split=args.split,
        output_root=args.output_root,
        output_size=args.output_size,
        max_images=args.max_images,
    )
    write_manifest(args.output_root, rows)
    print(f"Converted {len(rows)} {args.split} images to {args.output_root}")
    print(f"Manifest: {args.output_root / 'hela_coco_manifest.csv'}")


if __name__ == "__main__":
    main()
