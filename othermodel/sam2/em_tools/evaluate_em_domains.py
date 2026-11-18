import argparse
import csv
import subprocess
import sys
from pathlib import Path


DEFAULT_DOMAINS = [
    "domain_2_cells",
    "domain_3_muscle",
    "domain_4_fuzzy_blobs",
    "domain_6_Lucchi",
]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run SAM2 EM evaluation on multiple DeepPI/OAIS domains.")
    parser.add_argument("--dataset-root", required=True, help="Root containing domain_xxx/test/input,target folders.")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--sam2-root", default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Single checkpoint used for all domains. Mutually compatible with the old one-model evaluation mode.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help=(
            "Root containing one fine-tuned checkpoint per domain. "
            "Default layout is <checkpoint-root>/<domain>/checkpoints/checkpoint.pt."
        ),
    )
    parser.add_argument(
        "--checkpoint-pattern",
        default="{domain}/checkpoints/checkpoint.pt",
        help="Pattern relative to --checkpoint-root. The token {domain} is replaced by each domain name.",
    )
    parser.add_argument("--model-cfg", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    parser.add_argument("--merge-strategy", choices=["prompt_set", "positive_union"], default="prompt_set")
    parser.add_argument("--target-clicks", default="1,3,5,10,15")
    parser.add_argument("--target-ious", default="0.85,0.88,0.90,0.93")
    parser.add_argument("--max-clicks", type=int, default=15)
    parser.add_argument("--min-error-area", type=int, default=5)
    parser.add_argument("--visualize-max-images", type=int, default=3)
    parser.add_argument("--single-mask", action="store_true")
    parser.add_argument("--use-mask-input", action="store_true")
    parser.add_argument("--save-patent", action="store_true")
    parser.add_argument("--patent-step", type=int, default=10)
    parser.add_argument("--patent-title", default="TEM interactive segmentation result")
    parser.add_argument("--patent-method-label", default=None)
    parser.add_argument("--patent-reference-root", default=None, help="Root containing per-domain Base patent_npz/Base folders.")
    parser.add_argument("--patent-reference-label", default="Base model")
    return parser


def resolve_checkpoint(args, domain):
    if args.checkpoint_root:
        checkpoint_pattern = str(args.checkpoint_pattern)
        checkpoint_pattern = checkpoint_pattern.replace("{domain}", domain)
        checkpoint_pattern = checkpoint_pattern.replace("__DOMAIN__", domain)
        if checkpoint_pattern == str(args.checkpoint_pattern):
            checkpoint_pattern = domain
        checkpoint = Path(args.checkpoint_root) / checkpoint_pattern
    elif args.checkpoint:
        checkpoint = Path(args.checkpoint)
    else:
        raise ValueError("Pass either --checkpoint or --checkpoint-root.")

    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint for {domain} not found: {checkpoint}")
    return checkpoint


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = build_arg_parser().parse_args()
    script_path = Path(__file__).resolve().parent / "evaluate_em.py"
    sam2_root = Path(args.sam2_root).resolve() if args.sam2_root else Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else sam2_root / "test_result" / "SAM2_EM"

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    for domain in args.domains:
        domain_dir = dataset_root / domain
        image_dir = domain_dir / "test" / "input"
        mask_dir = domain_dir / "test" / "target"
        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(f"{domain} must contain test/input and test/target under {domain_dir}")

        checkpoint = resolve_checkpoint(args, domain)
        run_name = f"{domain}_{args.merge_strategy}"
        domain_output_dir = output_root / run_name
        cmd = [
            sys.executable,
            str(script_path),
            "--image-dir",
            str(image_dir),
            "--mask-dir",
            str(mask_dir),
            "--sam2-root",
            str(sam2_root),
            "--checkpoint",
            str(checkpoint),
            "--dataset-name",
            domain,
            "--run-name",
            run_name,
            "--merge-strategy",
            args.merge_strategy,
            "--target-clicks",
            args.target_clicks,
            "--target-ious",
            args.target_ious,
            "--max-clicks",
            str(args.max_clicks),
            "--min-error-area",
            str(args.min_error_area),
            "--visualize-max-images",
            str(args.visualize_max_images),
        ]
        if args.model_cfg:
            cmd.extend(["--model-cfg", args.model_cfg])
        if args.output_root:
            cmd.extend(["--output-root", str(output_root)])
        else:
            cmd.extend(["--output-root", str(output_root)])
        if args.device:
            cmd.extend(["--device", args.device])
        if args.single_mask:
            cmd.append("--single-mask")
        if args.use_mask_input:
            cmd.append("--use-mask-input")
        if args.save_patent:
            cmd.append("--save-patent")
            cmd.extend(["--patent-step", str(args.patent_step)])
            cmd.extend(["--patent-title", args.patent_title])
            cmd.extend(["--patent-reference-label", args.patent_reference_label])
            if args.patent_method_label:
                cmd.extend(["--patent-method-label", args.patent_method_label])
            if args.patent_reference_root:
                reference_dir = Path(args.patent_reference_root) / domain / "patent_npz" / "Base"
                cmd.extend(["--patent-reference-dir", str(reference_dir)])

        print("\n" + "=" * 80)
        print(f"Evaluating SAM2 on {domain}")
        print("Command:", " ".join(cmd))
        print("=" * 80)
        subprocess.run(cmd, cwd=str(sam2_root), check=True)

    fixed_rows = []
    noc_rows = []
    for domain in args.domains:
        domain_output_dir = output_root / f"{domain}_{args.merge_strategy}"
        fixed_rows.extend(read_csv_rows(domain_output_dir / "sam2_summary_fixed_clicks.csv"))
        noc_rows.extend(read_csv_rows(domain_output_dir / "sam2_summary_noc.csv"))

    write_csv_rows(output_root / f"sam2_all_domains_fixed_clicks_{args.merge_strategy}.csv", fixed_rows)
    write_csv_rows(output_root / f"sam2_all_domains_noc_{args.merge_strategy}.csv", noc_rows)

    print("\n[DONE] SAM2 all-domain evaluation finished.")
    print(f"Combined fixed-click summary: {output_root / f'sam2_all_domains_fixed_clicks_{args.merge_strategy}.csv'}")
    print(f"Combined NoC summary: {output_root / f'sam2_all_domains_noc_{args.merge_strategy}.csv'}")


if __name__ == "__main__":
    main()
