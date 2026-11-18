import argparse
from pathlib import Path

from patent_figure_utils import generate_patent_figures


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate patent-style black-white comparison figures from saved npz files.")
    parser.add_argument("--output-root", required=True, help="Folder that contains patent_npz/<method>.")
    parser.add_argument("--method-mode", required=True, help="Method npz folder name, e.g. ClickUNet or SAM2_prompt_set.")
    parser.add_argument("--reference-dir", required=True, help="Folder containing Base npz files.")
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--title", default="TEM interactive segmentation result")
    parser.add_argument("--reference-label", default="Base model")
    parser.add_argument("--method-label", default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()
    generate_patent_figures(
        output_root=Path(args.output_root),
        method_mode=args.method_mode,
        step=args.step,
        fig_title=args.title,
        reference_dir=Path(args.reference_dir),
        reference_label=args.reference_label,
        method_label=args.method_label or args.method_mode,
    )


if __name__ == "__main__":
    main()
