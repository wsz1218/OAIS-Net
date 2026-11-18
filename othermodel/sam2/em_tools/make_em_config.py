import argparse
import re
from pathlib import Path


def yaml_path(path):
    return str(Path(path).resolve()).replace("\\", "/")


def replace_first_scalar(text, key, value):
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*).*$", re.MULTILINE)
    text, count = pattern.subn(rf"\g<1>{value}", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not replace config key: {key}")
    return text


def replace_checkpoint_path(text, checkpoint_path):
    pattern = re.compile(r"^(\s*checkpoint_path:\s*).*$", re.MULTILINE)
    text, count = pattern.subn(rf"\g<1>{yaml_path(checkpoint_path)}", text, count=1)
    if count != 1:
        raise RuntimeError("Could not replace checkpoint_path.")
    return text


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate a SAM2.1 EM semantic fine-tuning config from the official MOSE template."
    )
    parser.add_argument("--sam2-root", default=None, help="Path to local SAM2 repo. Defaults to this script's parent repo.")
    parser.add_argument("--data-dir", default=None, help="Directory created by prepare_em_data.py.")
    parser.add_argument("--checkpoint", required=True, help="Official SAM2.1 checkpoint path.")
    parser.add_argument("--config-name", default="sam2.1_hiera_b+_EM_semantic_finetune.yaml")
    parser.add_argument("--template-name", default="sam2.1_hiera_b+_MOSE_finetune.yaml")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--max-num-objects", type=int, default=1)
    parser.add_argument("--base-lr", type=float, default=5e-6)
    parser.add_argument("--vision-lr", type=float, default=3e-6)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--phases-per-epoch", type=int, default=1)
    parser.add_argument("--multiplier", type=int, default=4)
    parser.add_argument("--experiment-log-dir", default=None)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    return parser


def main():
    default_sam2_root = Path(__file__).resolve().parents[1]
    args = build_arg_parser().parse_args()
    sam2_root = Path(args.sam2_root).resolve() if args.sam2_root else default_sam2_root
    data_dir = Path(args.data_dir).resolve() if args.data_dir else sam2_root / "em_data" / "oais_semantic_train"
    experiment_log_dir = (
        Path(args.experiment_log_dir).resolve()
        if args.experiment_log_dir
        else sam2_root / "sam2_logs" / "em_bplus_semantic_finetune"
    )

    config_dir = sam2_root / "sam2" / "configs" / "sam2.1_training"
    template_path = config_dir / args.template_name
    output_path = config_dir / args.config_name
    checkpoint = Path(args.checkpoint).resolve()

    if not template_path.exists():
        raise FileNotFoundError(f"Template config not found: {template_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    img_folder = data_dir / "JPEGImages"
    gt_folder = data_dir / "Annotations"
    file_list_txt = data_dir / "train.txt"
    for path in [img_folder, gt_folder, file_list_txt]:
        if not path.exists():
            raise FileNotFoundError(f"SAM2 EM data path not found: {path}")

    text = template_path.read_text(encoding="utf-8")
    replacements = {
        "resolution": args.resolution,
        "train_batch_size": args.batch_size,
        "num_train_workers": args.num_workers,
        "num_frames": args.num_frames,
        "max_num_objects": args.max_num_objects,
        "base_lr": args.base_lr,
        "vision_lr": args.vision_lr,
        "phases_per_epoch": args.phases_per_epoch,
        "num_epochs": args.epochs,
        "img_folder": yaml_path(img_folder),
        "gt_folder": yaml_path(gt_folder),
        "file_list_txt": yaml_path(file_list_txt),
        "multiplier": args.multiplier,
        "gpus_per_node": args.gpus_per_node,
        "experiment_log_dir": yaml_path(experiment_log_dir),
    }
    for key, value in replacements.items():
        text = replace_first_scalar(text, key, value)
    text = replace_checkpoint_path(text, checkpoint)

    output_path.write_text(text, encoding="utf-8")
    print("=" * 80)
    print("SAM2 EM semantic fine-tuning config generated.")
    print(f"Config: {output_path.resolve()}")
    print(f"Data: {data_dir.resolve()}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Experiment log dir: {experiment_log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
