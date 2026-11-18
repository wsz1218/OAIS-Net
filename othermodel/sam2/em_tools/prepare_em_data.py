import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_sort_key(value):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"([0-9]+)", str(value))
    ]


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(value)).strip("_")


def collect_train_pairs_from_scan_root(scan_root):
    scan_root = Path(scan_root)
    pairs = []
    for input_dir in scan_root.rglob("input"):
        if input_dir.parent.name.lower() != "train":
            continue
        target_dir = input_dir.parent / "target"
        if target_dir.is_dir():
            pairs.append((input_dir, target_dir))
    return sorted(pairs, key=lambda item: str(item[0]).lower())


def collect_train_pairs_from_sources(sources):
    pairs = []
    for source in sources:
        source = Path(source)
        if (source / "input").is_dir() and (source / "target").is_dir():
            pairs.append((source / "input", source / "target"))
        elif (source / "train" / "input").is_dir() and (source / "train" / "target").is_dir():
            pairs.append((source / "train" / "input", source / "train" / "target"))
        else:
            raise ValueError(
                f"Cannot find input/target folders under {source}. "
                "Pass a train folder, or a domain folder containing train/input and train/target."
            )
    return pairs


def collect_images(input_dir):
    return sorted(
        [path for path in Path(input_dir).iterdir() if path.suffix.lower() in IMAGE_EXTS],
        key=lambda path: natural_sort_key(path.name),
    )


def match_mask(image_path, target_dir):
    candidates = sorted(
        Path(target_dir).glob(f"{image_path.stem}.*"),
        key=lambda path: natural_sort_key(path.name),
    )
    return candidates[0] if candidates else None


def to_uint8_gray(image):
    if image.dtype == np.uint8:
        return image
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def write_palette_mask(mask, path):
    mask_p = Image.fromarray(mask.astype(np.uint8), mode="P")
    palette = [
        0, 0, 0,
        255, 255, 255,
    ]
    palette += [0] * (768 - len(palette))
    mask_p.putpalette(palette)
    mask_p.save(path)


def convert_sequence(input_dir, target_dir, out_img_root, out_ann_root, sequence_name, keep_empty):
    image_paths = collect_images(input_dir)
    seq_img_dir = out_img_root / sequence_name
    seq_ann_dir = out_ann_root / sequence_name
    seq_img_dir.mkdir(parents=True, exist_ok=True)
    seq_ann_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    frame_idx = 0
    for image_path in image_paths:
        mask_path = match_mask(image_path, target_dir)
        if mask_path is None:
            rows.append({
                "sequence": sequence_name,
                "source_image": str(image_path),
                "source_mask": "",
                "status": "missing_mask",
            })
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask_gray is None:
            rows.append({
                "sequence": sequence_name,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "status": "read_failed",
            })
            continue

        if image.ndim == 3:
            image_gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
        image_gray = to_uint8_gray(image_gray)
        mask = (mask_gray > 127).astype(np.uint8)
        if not keep_empty and int(mask.sum()) == 0:
            rows.append({
                "sequence": sequence_name,
                "source_image": str(image_path),
                "source_mask": str(mask_path),
                "status": "empty_mask_skipped",
            })
            continue

        frame_name = f"{frame_idx:05d}"
        image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(seq_img_dir / f"{frame_name}.jpg"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        write_palette_mask(mask, seq_ann_dir / f"{frame_name}.png")

        rows.append({
            "sequence": sequence_name,
            "frame": frame_name,
            "source_image": str(image_path),
            "source_mask": str(mask_path),
            "status": "converted",
            "foreground_pixels": int(mask.sum()),
        })
        frame_idx += 1

    return (sequence_name if frame_idx > 0 else None), rows


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert DeepPI/OAIS EM train/input + train/target folders to the "
            "SAM2 PNGRawDataset format. Masks are converted to one semantic "
            "foreground object with label id 1."
        )
    )
    parser.add_argument(
        "--scan-root",
        default=None,
        help="Root recursively scanned for train/input + train/target folders.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Specific train or domain folder. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root. Defaults to <sam2-root>/em_data/oais_semantic_train.",
    )
    parser.add_argument("--name-prefix", default="", help="Optional sequence name prefix.")
    parser.add_argument("--keep-empty", action="store_true", help="Keep empty foreground masks.")
    return parser


def main():
    sam2_root = Path(__file__).resolve().parents[1]
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else sam2_root / "em_data" / "oais_semantic_train"
    out_img_root = output_dir / "JPEGImages"
    out_ann_root = output_dir / "Annotations"
    out_img_root.mkdir(parents=True, exist_ok=True)
    out_ann_root.mkdir(parents=True, exist_ok=True)

    if args.source:
        pairs = collect_train_pairs_from_sources(args.source)
    else:
        scan_root = Path(args.scan_root) if args.scan_root else sam2_root.parents[2] / "pi_seg" / "data" / "datasets"
        if not scan_root.exists():
            raise FileNotFoundError(
                f"Dataset scan root not found: {scan_root}. Pass --scan-root on the server."
            )
        pairs = collect_train_pairs_from_scan_root(scan_root)

    if not pairs:
        raise RuntimeError("No train/input + train/target pairs found.")

    sequence_names = []
    manifest_rows = []
    used_names = set()
    for input_dir, target_dir in pairs:
        domain_dir = input_dir.parent.parent if input_dir.parent.name.lower() == "train" else input_dir.parent
        sequence_name = safe_name(f"{args.name_prefix}{domain_dir.name}")
        if sequence_name in used_names:
            sequence_name = safe_name(f"{args.name_prefix}{domain_dir.parent.name}_{domain_dir.name}")
        base_name = sequence_name
        suffix = 2
        while sequence_name in used_names:
            sequence_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(sequence_name)

        converted_name, rows = convert_sequence(
            input_dir=input_dir,
            target_dir=target_dir,
            out_img_root=out_img_root,
            out_ann_root=out_ann_root,
            sequence_name=sequence_name,
            keep_empty=args.keep_empty,
        )
        manifest_rows.extend(rows)
        if converted_name is not None:
            sequence_names.append(converted_name)

    train_list = output_dir / "train.txt"
    train_list.write_text("\n".join(sequence_names) + "\n", encoding="utf-8")

    manifest_path = output_dir / "manifest.csv"
    fieldnames = sorted({key for row in manifest_rows for key in row.keys()})
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("=" * 80)
    print("SAM2 EM semantic data conversion finished.")
    print(f"Input train pairs: {len(pairs)}")
    print(f"Sequences: {len(sequence_names)}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Train list: {train_list.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
