import argparse
from pathlib import Path

import cv2


def convert_png_folder(input_dir, output_dir, size=(1024, 1024), overwrite=False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_path.glob("*.png"))
    if not png_files:
        print(f"No PNG files found in: {input_path}")
        return

    converted = 0
    skipped = 0

    for png_file in png_files:
        out_file = output_path / f"{png_file.stem}.tif"
        if out_file.exists() and not overwrite:
            skipped += 1
            print(f"[SKIP] Exists: {out_file}")
            continue

        image = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
        if image is None:
            skipped += 1
            print(f"[SKIP] Cannot read: {png_file}")
            continue

        resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        ok = cv2.imwrite(str(out_file), resized)
        if not ok:
            skipped += 1
            print(f"[SKIP] Cannot write: {out_file}")
            continue

        converted += 1
        print(f"[OK] {png_file.name} -> {out_file.name}")

    print(f"Finished. Converted: {converted}, skipped: {skipped}, output: {output_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert PNG images in a folder to 1024x1024 TIF images."
    )
    parser.add_argument("input_dir", help="Folder containing PNG files.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output folder. Defaults to '<input_dir>_tif'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing TIF files.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.with_name(f"{input_dir.name}_tif")
    convert_png_folder(input_dir, output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
