"""Prepare CTC HeLa sequence 01 without mixing GT and silver labels."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT / "pi_seg/data/datasets/DIC-C2DH-HeLa_training/DIC-C2DH-HeLa"
)
LABEL_SOURCES = {
    "GT": "Manual segmentation annotations from 01_GT/SEG; annotated frames only.",
    "ST": "Silver-standard segmentation annotations from 01_ST/SEG; not manual GT.",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_array(path):
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Expected a single-frame image: {}".format(path))
        return np.array(image)


def collect_pairs(source):
    pairs = []
    for label_source in LABEL_SOURCES:
        mask_dir = source / ("01_" + label_source) / "SEG"
        mask_paths = sorted(mask_dir.glob("man_seg*.tif"))
        if not mask_paths:
            raise FileNotFoundError("No segmentation annotations in {}".format(mask_dir))
        for mask_path in mask_paths:
            match = re.fullmatch(r"man_seg(\d{3})\.tif", mask_path.name)
            if match is None:
                raise ValueError("Unexpected annotation filename: {}".format(mask_path))
            frame = match.group(1)
            image_path = source / "01" / ("t" + frame + ".tif")
            image, labels = read_array(image_path), read_array(mask_path)
            if image.ndim != 2 or image.dtype != np.uint8:
                raise ValueError("Expected an 8-bit grayscale image: {}".format(image_path))
            if labels.shape != image.shape or labels.dtype.kind not in "ui":
                raise ValueError("Invalid label shape/type: {}".format(mask_path))
            pairs.append({
                "label_source": label_source,
                "frame": int(frame),
                "sample": "x_" + frame,
                "source_image": image_path.relative_to(source).as_posix(),
                "source_mask": mask_path.relative_to(source).as_posix(),
                "source_image_sha256": sha256(image_path),
                "source_mask_sha256": sha256(mask_path),
                "source_height": image.shape[0],
                "source_width": image.shape[1],
                "num_instances": int(np.count_nonzero(np.unique(labels))),
            })
    return pairs


def prepare(source, output, size):
    if size <= 0:
        raise ValueError("Output size must be positive.")
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing output: {}".format(output))

    # Validate every source pair before creating any exported files.
    pairs = collect_pairs(source)
    output.mkdir(parents=True)
    for label_source, description in LABEL_SOURCES.items():
        split_root = output / label_source
        (split_root / "input").mkdir(parents=True)
        (split_root / "target").mkdir()
        rows = []
        for pair in pairs:
            if pair["label_source"] != label_source:
                continue
            image = read_array(source / pair["source_image"])
            labels = read_array(source / pair["source_mask"])
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.resize(
                (labels > 0).astype(np.uint8) * 255,
                (size, size), interpolation=cv2.INTER_NEAREST,
            )
            image_relative = "input/{}.tif".format(pair["sample"])
            mask_relative = "target/{}.png".format(pair["sample"])
            image_output = split_root / image_relative
            mask_output = split_root / mask_relative
            Image.fromarray(image).save(image_output, compression="tiff_deflate")
            Image.fromarray(mask).save(mask_output)
            if not np.array_equal(read_array(image_output), image):
                raise RuntimeError("Saved image differs: {}".format(image_output))
            if not np.array_equal(read_array(mask_output), mask):
                raise RuntimeError("Saved mask differs: {}".format(mask_output))
            rows.append(dict(
                pair,
                output_image=image_relative,
                output_mask=mask_relative,
                output_height=size,
                output_width=size,
                foreground_pixels=int(np.count_nonzero(mask)),
            ))

        with (split_root / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("{}: {} paired images/masks -> {}".format(label_source, len(rows), split_root))

    for pair in pairs:
        for kind in ("image", "mask"):
            if sha256(source / pair["source_" + kind]) != pair["source_" + kind + "_sha256"]:
                raise RuntimeError("Source file changed during conversion: {}".format(pair))

    metadata = {
        "dataset": "Cell Tracking Challenge DIC-C2DH-HeLa",
        "modality": "Differential interference contrast optical microscopy, not EM",
        "sequence": "01",
        "source_root": str(source),
        "original_frame_count": len(list((source / "01").glob("t*.tif"))),
        "label_sources": {
            key: {"description": description,
                  "pairs": sum(p["label_source"] == key for p in pairs)}
            for key, description in LABEL_SOURCES.items()
        },
        "output_size": [size, size],
        "image_resize": "OpenCV INTER_LANCZOS4",
        "mask_resize": "OpenCV INTER_NEAREST",
        "mask_encoding": "0 = background; 255 = all positive instance IDs merged",
        "naming": "x_NNN preserves the original tNNN frame index",
        "excluded_annotations": ["01_GT/TRA (tracking markers)", "01_ERR_SEG (erroneous masks)"],
        "train_test_split": None,
        "warning": "GT and ST share frames. Do not treat these exports as independent train/test splits.",
        "source_hashes_verified_unchanged": True,
    }
    (output / "dataset_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    output = args.output_root or args.source_root / "01_prepared"
    prepare(args.source_root, output, args.size)


if __name__ == "__main__":
    main()
