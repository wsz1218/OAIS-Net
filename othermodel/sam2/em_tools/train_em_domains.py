import argparse
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
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune one SAM2 model per DeepPI/OAIS EM domain for whole-image "
            "binary semantic interactive segmentation."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root containing domain_xxx/train/input,target and domain_xxx/test/input,target.",
    )
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--sam2-root", default=None, help="Path to SAM2 repo. Defaults to this script's parent repo.")
    parser.add_argument("--checkpoint", required=True, help="Official SAM2/SAM2.1 checkpoint used for fine-tuning init.")
    parser.add_argument("--data-root", default=None, help="Root for converted SAM2 domain datasets.")
    parser.add_argument("--exp-root", default=None, help="Root for per-domain logs and checkpoints.")
    parser.add_argument("--config-prefix", default="sam2.1_hiera_b+_EM")
    parser.add_argument("--template-name", default="sam2.1_hiera_b+_MOSE_finetune.yaml")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--base-lr", type=float, default=5e-6)
    parser.add_argument("--vision-lr", type=float, default=3e-6)
    parser.add_argument("--phases-per-epoch", type=int, default=1)
    parser.add_argument("--multiplier", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--skip-data-prepare", action="store_true")
    parser.add_argument("--skip-config", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_domain(dataset_root, domain):
    domain_dir = dataset_root / domain
    required = [
        domain_dir / "train" / "input",
        domain_dir / "train" / "target",
        domain_dir / "test" / "input",
        domain_dir / "test" / "target",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{domain} is missing required folders: {missing}")
    return domain_dir


def run_command(cmd, cwd, dry_run):
    print("Command:", " ".join(str(item) for item in cmd))
    if dry_run:
        return
    subprocess.run([str(item) for item in cmd], cwd=str(cwd), check=True)


def main():
    args = build_arg_parser().parse_args()
    default_sam2_root = Path(__file__).resolve().parents[1]
    sam2_root = Path(args.sam2_root).resolve() if args.sam2_root else default_sam2_root
    dataset_root = Path(args.dataset_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else sam2_root / "em_data" / "oais_semantic_domains"
    exp_root = Path(args.exp_root).resolve() if args.exp_root else sam2_root / "sam2_logs" / "em_bplus_semantic_domains"

    prepare_script = sam2_root / "em_tools" / "prepare_em_data.py"
    config_script = sam2_root / "em_tools" / "make_em_config.py"
    train_script = sam2_root / "training" / "train.py"

    for path in [sam2_root, dataset_root, checkpoint, prepare_script, config_script, train_script]:
        if not path.exists():
            raise FileNotFoundError(path)

    domain_dirs = {domain: validate_domain(dataset_root, domain) for domain in args.domains}

    for domain in args.domains:
        domain_dir = domain_dirs[domain]
        data_dir = data_root / domain
        exp_dir = exp_root / domain
        config_name = f"{args.config_prefix}_{domain}.yaml"

        print("\n" + "=" * 80)
        print(f"Training SAM2 EM domain: {domain}")
        print(f"Dataset: {domain_dir}")
        print(f"Converted data: {data_dir}")
        print(f"Experiment: {exp_dir}")
        print(f"Expected checkpoint: {exp_dir / 'checkpoints' / 'checkpoint.pt'}")
        print("=" * 80)

        if not args.skip_data_prepare:
            run_command(
                [
                    sys.executable,
                    prepare_script,
                    "--source",
                    domain_dir,
                    "--output-dir",
                    data_dir,
                ],
                cwd=sam2_root,
                dry_run=args.dry_run,
            )

        if not args.skip_config:
            run_command(
                [
                    sys.executable,
                    config_script,
                    "--sam2-root",
                    sam2_root,
                    "--data-dir",
                    data_dir,
                    "--checkpoint",
                    checkpoint,
                    "--config-name",
                    config_name,
                    "--template-name",
                    args.template_name,
                    "--experiment-log-dir",
                    exp_dir,
                    "--epochs",
                    args.epochs,
                    "--num-workers",
                    args.num_workers,
                    "--num-frames",
                    args.num_frames,
                    "--batch-size",
                    args.batch_size,
                    "--resolution",
                    args.resolution,
                    "--base-lr",
                    args.base_lr,
                    "--vision-lr",
                    args.vision_lr,
                    "--phases-per-epoch",
                    args.phases_per_epoch,
                    "--multiplier",
                    args.multiplier,
                    "--gpus-per-node",
                    args.num_gpus,
                ],
                cwd=sam2_root,
                dry_run=args.dry_run,
            )

        if not args.skip_train:
            run_command(
                [
                    sys.executable,
                    train_script,
                    "-c",
                    f"configs/sam2.1_training/{config_name}",
                    "--use-cluster",
                    0,
                    "--num-gpus",
                    args.num_gpus,
                ],
                cwd=sam2_root,
                dry_run=args.dry_run,
            )

    print("\n[DONE] SAM2 per-domain fine-tuning commands completed.")
    print(f"Checkpoint root for evaluation: {exp_root}")


if __name__ == "__main__":
    main()
