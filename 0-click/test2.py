import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import oaisnet_ablation_sta_ltm1 as table3

IMAGE_DIR = "pi_seg/data/datasets/Skeletal_muscle_test/domain_6_Lucchi/test_augmented_8fold/input"
MASK_DIR = "pi_seg/data/datasets/Skeletal_muscle_test/domain_6_Lucchi/test_augmented_8fold/target"
MODEL_PATH = "save/skeletal_muscle_run_Lucchi/model_latest.pth"
MODES = ("Base", "STA-only", "LTM-only", "OAIS-Net")
DIAGNOSTIC_MODES = ("STA+encoder-LTM", "PI-only")
TABLE3_MODES = {
    "Base": "Base",
    "STA-only": "Base+STA",
    "LTM-only": "Base+LTM",
    "OAIS-Net": "OAIS-Net",
}


def mode_config(mode):
    if mode in TABLE3_MODES:
        config = dict(table3.ABLATION_MODES[TABLE3_MODES[mode]])
    elif mode == "STA+encoder-LTM":
        config = dict(table3.ABLATION_MODES["Base+LTM"], sta=True)
    elif mode == "PI-only":
        config = dict(table3.ABLATION_MODES["OAIS-Net"], sta=False)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if config.get("main_protocol"):
        raise ValueError("Zero-click experiment requires Table 3 specialized ablation protocol")
    return config


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_sequences(sequences, output_dir, plot_step):
    count = len(sequences["Base"])
    selected = np.arange(0, count, plot_step)
    if selected[-1] != count - 1:
        selected = np.append(selected, count - 1)
    styles = {
        "Base": ("#c93636", "--", "s"),
        "STA-only": ("#2b805c", "-.", "^"),
        "LTM-only": ("#7b57a1", ":", "D"),
        "OAIS-Net": ("#196699", "-", "o"),
    }
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    for mode in MODES:
        color, line, marker = styles[mode]
        ax.plot(selected + 1, sequences[mode][selected] * 100.0,
                color=color, linestyle=line, marker=marker,
                linewidth=2, markersize=5, label=mode)
    ax.set(xlabel="Image sequence index", ylabel="Zero-click IoU (%)",
           title="Lucchi++ zero-click IoU", xlim=(0.5, count + 0.5))
    ax.set_xticks(selected + 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(output_dir / "zero_click_IoU_sequence.png")
    fig.savefig(output_dir / "zero_click_IoU_sequence.pdf")
    plt.close(fig)


def update_ltm_with_prediction(adaptor, image, points, prediction, gt_mask,
                               mode_config, args, guard_type):
    target = prediction.copy()
    if not args.ltm_rollback:
        table3.run_ltm_update(adaptor, image, points, target, mode_config, args)
        return True

    before_pred = table3.forward_mask(adaptor.model, image, points)
    before_metrics = table3.current_metrics(before_pred, gt_mask)
    before_state = table3.snapshot_trainable_weights(adaptor.model)
    table3.run_ltm_update(adaptor, image, points, target, mode_config, args)
    after_pred = table3.forward_mask(adaptor.model, image, points)
    after_metrics = table3.current_metrics(after_pred, gt_mask)
    accepted = table3.metrics_pass_guard(before_metrics, after_metrics, args, guard_type)
    if not accepted:
        table3.restore_weights(adaptor.model, before_state)
    return accepted


def run_one_image(adaptor, image_path, mask_path, mode_config, args, device):
    _, image = table3.load_image_tensor(image_path, device)
    gt_mask = table3.load_binary_mask(mask_path)
    robot = table3.RobotClicker(gt_mask, min_error_area=args.min_error_area)
    points = table3.empty_points_tensor(device, args.point_slots_per_class)
    clicks = []
    prediction = table3.forward_mask(adaptor.model, image, points)
    metrics = table3.current_metrics(prediction, gt_mask)
    zero_click_iou = metrics["iou"]
    sta_updates = 0
    ltm_attempts = 0
    ltm_accepted = 0
    ltm_trigger = mode_config.get("ltm_trigger", args.ltm_trigger)

    for _ in range(args.max_clicks):
        click = robot.get_next_click(prediction)
        if click is not None:
            clicks.append(click)
            points = table3.append_click_to_tensor(points, click)
            if mode_config["sta"]:
                before_state = None
                before_metrics = None
                if args.sta_rollback:
                    current_state = table3.snapshot_sta_weights(adaptor)
                    current_pred = table3.forward_mask(adaptor.model, image, points)
                    current_metrics = table3.current_metrics(current_pred, gt_mask)
                    if mode_config["ltm"] == "none":
                        initial_state = table3.snapshot_initial_sta_weights(adaptor)
                        table3.restore_weights(adaptor.model, initial_state)
                        base_pred = table3.forward_mask(adaptor.model, image, points)
                        base_metrics = table3.current_metrics(base_pred, gt_mask)
                        if current_metrics["iou"] >= base_metrics["iou"]:
                            before_state = current_state
                            before_metrics = current_metrics
                            table3.restore_weights(adaptor.model, current_state)
                        else:
                            before_state = initial_state
                            before_metrics = base_metrics
                    else:
                        before_state = current_state
                        before_metrics = current_metrics

                if hasattr(adaptor, "ablation_sta_only_mode"):
                    adaptor.ablation_sta_only_mode = mode_config["ltm"] == "none"
                adaptor.adapt_MI(image, points, clicks)
                sta_updates += 1
                if hasattr(adaptor, "ablation_sta_only_mode"):
                    adaptor.ablation_sta_only_mode = False

            prediction = table3.forward_mask(adaptor.model, image, points)
            if mode_config["sta"] and args.sta_rollback and before_state is not None:
                updated = table3.current_metrics(prediction, gt_mask)
                if mode_config["ltm"] == "none":
                    rollback = updated["iou"] + args.sta_rollback_tolerance < before_metrics["iou"]
                else:
                    rollback = not table3.metrics_pass_guard(
                        before_metrics, updated, args, "quality"
                    )
                if rollback:
                    table3.restore_weights(adaptor.model, before_state)
                    prediction = table3.forward_mask(adaptor.model, image, points)
            metrics = table3.current_metrics(prediction, gt_mask)

        if (ltm_trigger == "per-click" and mode_config["ltm"] != "none"
                and clicks and metrics["iou"] >= args.ltm_min_iou):
            accepted = update_ltm_with_prediction(
                adaptor, image, points, prediction, gt_mask, mode_config, args, "quality"
            )
            ltm_attempts += 1
            ltm_accepted += int(accepted)

    if (ltm_trigger == "end-of-image" and mode_config["ltm"] != "none"
            and clicks and metrics["iou"] >= args.ltm_min_iou):
        accepted = update_ltm_with_prediction(
            adaptor, image, points, prediction, gt_mask, mode_config, args, "boundary"
        )
        ltm_attempts += 1
        ltm_accepted += int(accepted)

    if args.reset_short_term_between_images or mode_config.get("reset_sta_after_image", False):
        table3.reset_short_term_weights(adaptor)

    return {
        "zero_click_iou": zero_click_iou,
        "final_iou": metrics["iou"],
        "final_dice": metrics["dice"],
        "clicks_used": len(clicks),
        "sta_updates": sta_updates,
        "ltm_update_attempts": ltm_attempts,
        "ltm_update_accepted": ltm_accepted,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Lucchi++ cross-image zero-click ablation with Table 3 update settings."
    )
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--mask-dir", default=MASK_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--output-root", default=str(ROOT / "零点击消融实验" / "results"))
    parser.add_argument("--adaptor-module", default="pi_seg.inference.oaims_adaptor_ablation")
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default=None)
    parser.add_argument("--max-clicks", type=int, default=15)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--point-slots-per-class", type=int, default=10)
    parser.add_argument("--min-error-area", type=int, default=5)
    parser.add_argument("--plot-step", type=int, default=2)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--sta-rollback", dest="sta_rollback", action="store_true", default=True)
    parser.add_argument("--no-sta-rollback", dest="sta_rollback", action="store_false")
    parser.add_argument("--sta-rollback-tolerance", type=float, default=0.0)
    parser.add_argument("--ltm-rollback", dest="ltm_rollback", action="store_true", default=True)
    parser.add_argument("--no-ltm-rollback", dest="ltm_rollback", action="store_false")
    parser.add_argument("--ltm-iou-tolerance", type=float, default=0.002)
    parser.add_argument("--ltm-dice-tolerance", type=float, default=0.002)
    parser.add_argument("--ltm-hd95-tolerance", type=float, default=1.0)
    parser.add_argument("--ltm-min-accept-iou-gain", type=float, default=0.001)
    parser.add_argument("--ltm-min-accept-dice-gain", type=float, default=0.001)
    parser.add_argument("--ltm-min-accept-hd95-gain", type=float, default=0.5)
    parser.add_argument("--ltm-min-iou", type=float, default=0.65)
    parser.add_argument("--ltm-max-clicks", type=int, default=10)
    parser.add_argument("--pure-ltm-lr", type=float, default=3e-6)
    parser.add_argument("--pure-ltm-steps", type=int, default=1)
    parser.add_argument("--pure-ltm-anchor-weight", type=float, default=1.0)
    parser.add_argument("--reset-short-term-between-images", action="store_true")
    parser.set_defaults(ltm_trigger="per-click")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.max_clicks < 1 or args.num_runs < 1 or args.plot_step < 1:
        parser.error("max-clicks, num-runs, and plot-step must be positive")
    image_dir = ROOT / args.image_dir
    mask_dir = ROOT / args.mask_dir
    model_path = ROOT / args.model_path
    for label, path in (("image-dir", image_dir), ("mask-dir", mask_dir),
                        ("model-path", model_path)):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    images = table3.collect_image_paths(str(image_dir))
    if len(images) < 2:
        raise ValueError("At least two images are required")
    masks = [table3.match_mask_path(path, str(mask_dir)) for path in images]
    missing = [str(path) for path, mask in zip(images, masks) if mask is None]
    if missing:
        raise FileNotFoundError(f"Missing masks: {missing}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    modes = MODES + DIAGNOSTIC_MODES if args.diagnose else MODES
    output_dir = Path(args.output_root) / (
        f"LucchiPP_zero_click_table3_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args))
    config.update({
        "image_dir": str(image_dir), "mask_dir": str(mask_dir),
        "model_path": str(model_path), "device": device,
        "ltm_supervision": "prediction",
        "modes": {mode: mode_config(mode) for mode in modes},
    })
    with open(output_dir / "experiment_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    adaptor_class = table3.load_adaptor_class(args.adaptor_module)
    all_rows = []
    runs = {}
    print(f"Images: {len(images)} | runs: {args.num_runs} | device: {device}")
    print("Table 3 settings; LTM target is current prediction, not GT.")
    if args.sta_rollback or args.ltm_rollback:
        print("GT remains in the Table 3 rollback decisions.")
    for mode in modes:
        table3.set_reproducible(args.seed)
        model = table3.load_ready_model(str(model_path), device)
        adaptor = adaptor_class(model)
        mode_runs = []
        for run in range(1, args.num_runs + 1):
            table3.set_reproducible(args.seed)
            adaptor.reset_weights()
            sequence = []
            for index, (image_path, mask_path) in enumerate(zip(images, masks), start=1):
                result = run_one_image(
                    adaptor, image_path, mask_path, mode_config(mode), args, device
                )
                sequence.append(result["zero_click_iou"])
                all_rows.append({
                    "mode": mode, "run": run, "image_index": index,
                    "image": Path(image_path).name, **result,
                })
                print(f"[{mode} run {run}] {index}/{len(images)} "
                      f"{Path(image_path).name}: zero-click IoU={result['zero_click_iou']:.4f}")
            mode_runs.append(sequence)
        runs[mode] = np.asarray(mode_runs, dtype=float)
        del adaptor, model
        if device == "cuda":
            torch.cuda.empty_cache()

    first = runs["Base"][:, 0]
    if not all(np.allclose(values[:, 0], first, atol=1e-6) for values in runs.values()):
        raise ValueError("First-image zero-click scores differ across modes")
    means = {mode: values.mean(axis=0) for mode, values in runs.items()}
    per_image = []
    for index, image_path in enumerate(images):
        row = {"image_index": index + 1, "image": Path(image_path).name}
        for mode in modes:
            values = runs[mode][:, index]
            row[mode] = float(values.mean())
            row[f"{mode}_std_across_runs"] = float(values.std(ddof=0))
            row[f"{mode}_gain_vs_base"] = float(
                (values - runs["Base"][:, index]).mean()
            )
        per_image.append(row)
    write_csv(output_dir / "per_image_zero_click_iou.csv", per_image)
    write_csv(output_dir / "per_mode_details.csv", all_rows)

    summary = []
    for mode in modes:
        following = runs[mode][:, 1:]
        run_means = following.mean(axis=1)
        config = mode_config(mode)
        summary.append({
            "mode": mode,
            "num_runs": args.num_runs,
            "num_following_images": following.shape[1],
            "sta_reset_after_image": config["reset_sta_after_image"],
            "ltm_trigger": config.get("ltm_trigger", "none"),
            "ltm_supervision": "prediction" if config["ltm"] != "none" else "none",
            "mean_zero_click_iou": float(run_means.mean()),
            "std_across_run_means": float(run_means.std(ddof=0)),
            "std_across_images": float(following.mean(axis=0).std(ddof=0)),
            "mean_gain_vs_base": float(
                (following - runs["Base"][:, 1:]).mean()
            ),
        })
    write_csv(output_dir / "following_images_summary.csv", summary)

    if args.diagnose:
        contrasts = (
            ("STA effect with pure LTM", "STA+encoder-LTM", "LTM-only"),
            ("PI without STA", "PI-only", "LTM-only"),
            ("STA effect with PI", "OAIS-Net", "PI-only"),
        )
        rows = []
        for label, first_mode, second_mode in contrasts:
            differences = runs[first_mode][:, 1:] - runs[second_mode][:, 1:]
            rows.append({
                "contrast": label, "first_mode": first_mode,
                "second_mode": second_mode,
                "mean_zero_click_iou_difference": float(differences.mean()),
                "first_mode_wins": int((differences > 0).sum()),
                "num_observations": differences.size,
            })
        write_csv(output_dir / "diagnostic_contrasts.csv", rows)

    plot_sequences(means, output_dir, args.plot_step)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
