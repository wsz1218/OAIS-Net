import os
# Set CUDA_VISIBLE_DEVICES in the shell when needed, for example:
# CUDA_VISIBLE_DEVICES=1 python oaims_test_iou_last.py
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "20260916"
import random
import glob
import cv2
import torch
import numpy as np
from scipy.ndimage import label
import re
import matplotlib.pyplot as plt
import time
from datetime import datetime
from types import SimpleNamespace
import pandas as pd

from model import initialize_model
from oaisnet_ablation_sta_ltm1 import run_one_image
from pi_seg.inference.oaims_adaptor_ablation import OAIMSAdaptor

# ================= 数据集配置 =================
# Modify the three paths of each dataset to match the setup used for its Table 1/3 results.
DATASET_CONFIGS = {
    "hela_cells": {
        "label": "HeLa_Cells",
        "image_dir": "pi_seg/data/datasets/Skeletal_muscle_test/domain_2_cells/test/input",
        "mask_dir": "pi_seg/data/datasets/Skeletal_muscle_test/domain_2_cells/test/target",
        "model_path": "save/skeletal_muscle_run_cells/model_latest.pth",
        "max_clicks": 15,
        "num_runs": 5,
        "target_clicks": [1, 3, 5, 10, 15],
        "target_ious": [0.85, 0.88, 0.90, 0.93],
        "visualize_max_images": 3,
    },
    "mouse_skeletal_muscle": {
        "label": "Mouse_Skeletal_Muscle",
        "image_dir": "pi_seg/data/datasets/Skeletal_muscle_test/domain_3_muscle/test/input",
        "mask_dir": "pi_seg/data/datasets/Skeletal_muscle_test/domain_3_muscle/test/target",
        "model_path": "save/skeletal_muscle_run_muscle/model_latest.pth",
        "max_clicks": 15,
        "num_runs": 5,
        "target_clicks": [1, 3, 5, 10, 15],
        "target_ious": [0.85, 0.88, 0.90, 0.93],
        "visualize_max_images": 3,
    },
    "nanoparticles_tem": {
        "label": "Nanoparticles_TEM",
        "image_dir": "pi_seg/data/datasets/Skeletal_muscle_test/domain_1_nano/test/input",
        "mask_dir": "pi_seg/data/datasets/Skeletal_muscle_test/domain_1_nano/test/target",
        "model_path": "save/skeletal_muscle_run_nanocells/model_latest.pth",
        "max_clicks": 15,
        "num_runs": 5,
        "target_clicks": [1, 3, 5, 10, 15],
        "target_ious": [0.85, 0.88, 0.90, 0.93],
        "visualize_max_images": 3,
    },
    "lucchi": {
        "label": "LucchiPP",
        "image_dir": "pi_seg/data/datasets/Skeletal_muscle_test/domain_6_Lucchi/test/input",
        "mask_dir": "pi_seg/data/datasets/Skeletal_muscle_test/domain_6_Lucchi/test/target",
        "model_path": "save/skeletal_muscle_run_Lucchi/model_latest.pth",
        "max_clicks": 15,
        "num_runs": 5,
        "target_clicks": [1, 3, 5, 10, 15],
        "target_ious": [0.85, 0.88, 0.90, 0.93],
        "visualize_max_images": 3,
    },
}

# Keep one name here to test a single dataset, or list all four for sequential evaluation.
RUN_DATASETS = [
    "hela_cells",
    "mouse_skeletal_muscle",
    "nanoparticles_tem",
    "lucchi",
]

OUTPUT_ROOT = "test_result"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 20260916


# Match the original Table 3 evaluator. The full OAIS mode intentionally uses
# the specialized adaptor and rollback policy rather than the legacy Table 1 PI path.
TABLE3_BASE_CONFIG = {
    "sta": False,
    "ltm": "none",
    "reset_sta_after_image": False,
    "main_protocol": False,
}
TABLE3_OAIS_CONFIG = {
    "sta": True,
    "ltm": "original",
    "ltm_trigger": "per-click",
    "reset_sta_after_image": False,
    "main_protocol": False,
}


def set_reproducible(seed=SEED):
    """Configure deterministic execution before model construction or adaptation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)


# ================= 核心工具函数 =================

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


# ★ 核心修改 1：将 Dice 计算公式替换为真正的 IoU 计算公式
def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return (1.0 * intersection) / union if union > 0 else 1.0


def save_paper_visual(
    img_gray, gt_mask, pred_mask, all_clicks, img_name, step, mode,
    iou_score, dice_score, sub_exp_dir,
):
    """
    将对比图保存到本次实验子文件夹的 visuals 目录下
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    img_rgb = cv2.cvtColor((img_gray * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    for (y, x, is_pos) in all_clicks:
        color = (0, 255, 0) if is_pos == 1 else (255, 0, 0)
        cv2.circle(img_rgb, (x, y), 3, color, -1)
        cv2.circle(img_rgb, (x, y), 6, color, 1)

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Image + {len(all_clicks)} Clicks", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title("Ground Truth (GT)", fontsize=14)
    axes[1].axis("off")

    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    overlay_rgb = img_rgb.copy()
    true_positive = np.logical_and(pred, gt)
    false_positive = np.logical_and(pred, np.logical_not(gt))
    false_negative = np.logical_and(np.logical_not(pred), gt)
    overlay_rgb[true_positive] = overlay_rgb[true_positive] * 0.5 + np.array([0, 255, 0]) * 0.5
    overlay_rgb[false_positive] = overlay_rgb[false_positive] * 0.5 + np.array([255, 0, 0]) * 0.5
    overlay_rgb[false_negative] = overlay_rgb[false_negative] * 0.5 + np.array([0, 0, 255]) * 0.5

    axes[2].imshow(overlay_rgb)
    axes[2].set_title(
        f"{mode} (IoU: {iou_score * 100:.1f}%, Dice: {dice_score * 100:.1f}%)",
        fontsize=12,
    )
    axes[2].axis("off")

    plt.tight_layout()

    vis_mode_dir = os.path.join(sub_exp_dir, "visuals", mode)
    os.makedirs(vis_mode_dir, exist_ok=True)
    save_path = os.path.join(vis_mode_dir, f"{img_name}_step{step}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close('all')


# ================= 机器人点击策略 =================

class RobotClicker:
    def __init__(self, gt_mask):
        self.gt_mask = gt_mask
        self.clicked_mask = np.zeros_like(gt_mask, dtype=np.uint8)

    def get_next_click(self, pred_mask):
        error_mask = (pred_mask != self.gt_mask).astype(np.uint8)
        error_mask[self.clicked_mask == 1] = 0
        labeled_array, num_features = label(error_mask)
        if num_features == 0: return None
        comp_sizes = np.bincount(labeled_array.ravel())
        comp_sizes[0] = 0
        max_id = np.argmax(comp_sizes)
        if comp_sizes[max_id] < 5: return None
        largest_comp_mask = (labeled_array == max_id).astype(np.uint8)
        dist = cv2.distanceTransform(largest_comp_mask, cv2.DIST_L2, 5)
        dist[self.clicked_mask == 1] = 0
        max_dist = np.max(dist)
        if max_dist < 1: return None
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        self.clicked_mask[y, x] = 1
        return (y, x, 1 if self.gt_mask[y, x] == 1 else 0)


def append_click_to_tensor(points_tensor, click, click_idx):
    y, x, is_pos = click
    new_points = points_tensor.clone()
    num_points = new_points.size(1) // 2

    offset = 0 if is_pos == 1 else num_points

    for i in range(offset, offset + num_points):
        if new_points[0, i, 0] == -1.0:
            new_points[0, i, 0] = y
            new_points[0, i, 1] = x
            new_points[0, i, 2] = 1.0
            break

    return new_points


def load_ready_model(model_path):
    print(f"📦 正在从 {model_path} 加载权重...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = initialize_model()
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    if isinstance(model, torch.nn.DataParallel): model = model.module
    model.load_state_dict(new_state_dict, strict=False)
    return model.to(DEVICE).eval()


# ================= 核心评估流程 =================

def make_table3_args(eval_config):
    """Use the same defaults as oaisnet_ablation_sta_ltm1.py."""
    return SimpleNamespace(
        target_clicks=eval_config["target_clicks"],
        target_ious=eval_config["target_ious"],
        max_clicks=eval_config["max_clicks"],
        point_slots_per_class=10,
        min_error_area=5,
        sta_rollback=True,
        sta_rollback_tolerance=0.0,
        ltm_rollback=True,
        ltm_iou_tolerance=0.002,
        ltm_dice_tolerance=0.002,
        ltm_hd95_tolerance=1.0,
        ltm_min_accept_iou_gain=0.001,
        ltm_min_accept_dice_gain=0.001,
        ltm_min_accept_hd95_gain=0.5,
        ltm_min_iou=0.65,
        ltm_max_clicks=10,
        pure_ltm_lr=3e-6,
        pure_ltm_steps=1,
        pure_ltm_anchor_weight=1.0,
        ltm_trigger="per-click",
        reset_short_term_between_images=False,
        with_noc=True,
    )


def run_fixed_clicks_evaluation(
    adaptor, img_paths, mask_dir, mode_cfg, eval_config, sub_exp_dir=None, mode_name=None,
):
    """Evaluate Base or OAIS with the Table 3 specialized evaluation path."""
    adaptor.reset_weights()
    args = make_table3_args(eval_config)
    target_clicks = eval_config["target_clicks"]
    target_ious = eval_config["target_ious"]
    iou_records = {k: [] for k in target_clicks}
    dice_records = {k: [] for k in target_clicks}
    noc_records = {target_iou: [] for target_iou in target_ious}
    latencies = []

    for img_idx, img_path in enumerate(img_paths):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        possible_masks = glob.glob(os.path.join(mask_dir, base_name + ".*"))
        if not possible_masks:
            continue

        visualization_callback = None
        if (
            sub_exp_dir is not None
            and mode_name is not None
            and img_idx < eval_config["visualize_max_images"]
        ):
            def visualization_callback(image_np, gt_mask, pred_mask, all_clicks, img_name, step, metrics):
                save_paper_visual(
                    image_np / 255.0,
                    gt_mask,
                    pred_mask,
                    all_clicks,
                    img_name,
                    step,
                    mode_name,
                    metrics["iou"],
                    metrics["dice"],
                    sub_exp_dir,
                )

        result = run_one_image(
            adaptor,
            img_path,
            possible_masks[0],
            mode_cfg,
            args,
            DEVICE,
            visualization_callback=visualization_callback,
        )
        for click_count in target_clicks:
            iou_records[click_count].append(result["fixed_clicks"][click_count]["iou"])
            dice_records[click_count].append(result["fixed_clicks"][click_count]["dice"])
        for target_iou in target_ious:
            noc_records[target_iou].append(result["noc"][target_iou])
        latencies.append(result["click_latency_ms"])

    avg_ious = {k: np.mean(v) for k, v in iou_records.items()}
    avg_dices = {k: np.mean(v) for k, v in dice_records.items()}
    avg_latency = np.mean(latencies) if latencies else 0
    return avg_ious, avg_dices, avg_latency, iou_records, dice_records, noc_records


# ================= 多数据集评估 =================

def run_dataset(dataset_key, dataset_config):
    set_reproducible()
    dataset_name = dataset_config["label"]
    image_dir = dataset_config["image_dir"]
    mask_dir = dataset_config["mask_dir"]
    model_path = dataset_config["model_path"]
    target_clicks = dataset_config["target_clicks"]
    target_ious = dataset_config["target_ious"]
    num_runs = dataset_config["num_runs"]

    for field_name, path in {
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "model_path": model_path,
    }.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[{dataset_key}] Invalid {field_name}: {path}. "
                "Update DATASET_CONFIGS before running the evaluation."
            )

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_exp_name = f"Results_{dataset_name}_IoU_Dice_{timestamp}"
    sub_exp_dir = os.path.join(OUTPUT_ROOT, sub_exp_name)
    os.makedirs(sub_exp_dir, exist_ok=True)

    def log_write(text):
        print(text)
        with open(os.path.join(sub_exp_dir, "test_log.txt"), "a", encoding="utf-8") as f:
            f.write(text + "\n")

    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.*")), key=natural_sort_key)
    if not img_paths:
        raise RuntimeError(f"[{dataset_key}] No images found in: {image_dir}")

    log_write(f"\nDataset: {dataset_name}")
    log_write(f"Images: {len(img_paths)}")
    log_write(f"Settings: max_clicks={dataset_config['max_clicks']}, num_runs={num_runs}")
    model = load_ready_model(model_path)
    adaptor = OAIMSAdaptor(model)

    csv_results = []

    log_write("\n▶ 运行 Base Model (Table 3 specialized evaluation path)...")
    adaptor.reset_weights()
    base_ious, base_dices, base_lat, _, _, base_noc_records = run_fixed_clicks_evaluation(
        adaptor,
        img_paths,
        mask_dir,
        TABLE3_BASE_CONFIG,
        dataset_config,
        sub_exp_dir=sub_exp_dir,
        mode_name="Base",
    )
    for k in target_clicks:
        csv_results.append({
            "Mode": "Base",
            "Clicks": k,
            "IoU": base_ious[k],
            "Dice": base_dices[k],
            "Latency_ms": base_lat,
        })

    log_write(f"\n▶ 运行 OAIS-Net (Table 3 specialized evaluation path, Run 1-{num_runs})...")
    oaims_iou_samples = {k: [] for k in target_clicks}
    oaims_dice_samples = {k: [] for k in target_clicks}
    oaims_noc_samples = {target_iou: [] for target_iou in target_ious}
    oaims_lats = []

    for i in range(num_runs):
        log_write(f"   - 正在执行实验 Run {i + 1}/{num_runs}...")
        set_reproducible()
        adaptor.reset_weights()

        run_ious, run_dices, run_lat, run_iou_samples, run_dice_samples, run_noc_samples = run_fixed_clicks_evaluation(
            adaptor,
            img_paths,
            mask_dir,
            TABLE3_OAIS_CONFIG,
            dataset_config,
            sub_exp_dir=sub_exp_dir if i == 0 else None,
            mode_name="OAIS-Net" if i == 0 else None,
        )
        oaims_lats.append(run_lat)
        for k in target_clicks:
            oaims_iou_samples[k].extend(run_iou_samples[k])
            oaims_dice_samples[k].extend(run_dice_samples[k])
            csv_results.append({
                "Mode": f"OAIMS_Run{i + 1}",
                "Clicks": k,
                "IoU": run_ious[k],
                "Dice": run_dices[k],
                "Latency_ms": run_lat,
            })
        for target_iou in target_ious:
            oaims_noc_samples[target_iou].extend(run_noc_samples[target_iou])

    oaims_iou_avg = {k: np.mean(v) for k, v in oaims_iou_samples.items()}
    oaims_iou_std = {k: np.std(v) for k, v in oaims_iou_samples.items()}
    oaims_dice_avg = {k: np.mean(v) for k, v in oaims_dice_samples.items()}
    oaims_dice_std = {k: np.std(v) for k, v in oaims_dice_samples.items()}
    base_noc_avg = {target_iou: np.mean(values) for target_iou, values in base_noc_records.items()}
    oaims_noc_avg = {target_iou: np.mean(values) for target_iou, values in oaims_noc_samples.items()}
    oaims_noc_std = {target_iou: np.std(values) for target_iou, values in oaims_noc_samples.items()}

    log_write("\n🏆 " + "=" * 55)
    log_write("   最终结果对比报告 (核心指标: IoU and Dice)")
    log_write("=" * 60)
    log_write("   点击次数 | Base IoU | OAIS IoU | Base Dice | OAIS Dice")
    log_write("-" * 60)
    for k in target_clicks:
        base_iou = base_ious[k] * 100
        oaims_iou = oaims_iou_avg[k] * 100
        oaims_iou_sd = oaims_iou_std[k] * 100
        base_dice = base_dices[k] * 100
        oaims_dice = oaims_dice_avg[k] * 100
        oaims_dice_sd = oaims_dice_std[k] * 100
        log_write(
            f"   @ {k:2d} 次   | {base_iou:.2f}% | {oaims_iou:.2f}% ± {oaims_iou_sd:.2f}% | "
            f"{base_dice:.2f}% | {oaims_dice:.2f}% ± {oaims_dice_sd:.2f}%"
        )
    log_write("-" * 60)
    log_write("\n   NoC Summary")
    log_write("   Target IoU | Base NoC | OAIS-Net NoC")
    log_write("-" * 60)
    for target_iou in target_ious:
        log_write(
            f"      {target_iou:.2f}    | {base_noc_avg[target_iou]:.2f} | "
            f"{oaims_noc_avg[target_iou]:.2f} ± {oaims_noc_std[target_iou]:.2f}"
        )
    log_write("-" * 60)
    log_write(f"⏱️  平均自适应响应耗时 (OAIMS): {np.mean(oaims_lats):.2f} ms")
    log_write("=" * 60)

    df = pd.DataFrame(csv_results)
    df.to_csv(os.path.join(sub_exp_dir, f"detailed_metrics_IoU_Dice_{dataset_name}.csv"), index=False)

    noc_results = []
    for target_iou in target_ious:
        noc_results.append({
            "Mode": "Base",
            "Target_IoU": target_iou,
            "NoC_mean": base_noc_avg[target_iou],
            "NoC_std": np.std(base_noc_records[target_iou]),
            "Num_samples": len(base_noc_records[target_iou]),
        })
        noc_results.append({
            "Mode": "OAIS-Net",
            "Target_IoU": target_iou,
            "NoC_mean": oaims_noc_avg[target_iou],
            "NoC_std": oaims_noc_std[target_iou],
            "Num_samples": len(oaims_noc_samples[target_iou]),
        })
    pd.DataFrame(noc_results).to_csv(
        os.path.join(sub_exp_dir, f"noc_metrics_{dataset_name}.csv"), index=False
    )

    log_write(f"\n✅ 评估完成！请在目录中查看结果: {sub_exp_dir}")


def main():
    for dataset_key in RUN_DATASETS:
        if dataset_key not in DATASET_CONFIGS:
            raise KeyError(f"Unknown dataset key in RUN_DATASETS: {dataset_key}")
        run_dataset(dataset_key, DATASET_CONFIGS[dataset_key])

if __name__ == "__main__":
    main()
