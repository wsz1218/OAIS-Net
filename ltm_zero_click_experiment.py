import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import glob
import cv2
import torch
import numpy as np
from scipy.ndimage import label
import re
import csv
import argparse
import matplotlib.pyplot as plt
import seaborn as sns  # 画图利器
import time
from datetime import datetime

from model import initialize_model
# 确保你的 Adaptor 导入路径是正确的
from pi_seg.inference.oaims_adaptor_shiying import OAIMSAdaptor

# ================= 路径配置 =================
IMAGE_DIR = "pi_seg/data/datasets/Skeletal_muscle_test/domain_2_cells/test_augmented_8fold/input"
MASK_DIR = "pi_seg/data/datasets/Skeletal_muscle_test/domain_2_cells/test_augmented_8fold/target"
MODEL_PATH = "save/skeletal_muscle_run_cells/model_latest.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_IOUS = [0.90]
MAX_CLICKS = 15
PLOT_STEP = 2
VISUALIZE_MAX_IMAGES = 3  # 保存前几张图的掩码可视化


# ================= 核心工具函数 =================
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return (1.0 * intersection) / union if union > 0 else 1.0


# ================= 绝美学术折线图生成器 =================
def plot_sequential_adaptation(base_seq, oaims_seq, target_iou, save_dir, plot_step=2):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    num_images = len(base_seq)
    all_indices = np.arange(1, num_images + 1)
    selected = np.arange(0, num_images, max(1, plot_step))
    if num_images > 0 and selected[-1] != num_images - 1:
        selected = np.append(selected, num_images - 1)
    image_indices = all_indices[selected]
    base_seq = np.asarray(base_seq)[selected]
    oaims_seq = np.asarray(oaims_seq)[selected]

    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)

    # Static baseline zero-click IoU.
    ax.plot(image_indices, np.asarray(base_seq) * 100.0, marker='s', markersize=7, linestyle='--',
            color='#d62728', linewidth=2, label='Baseline (Static Model)')

    # OAIS-Net zero-click IoU after sequential online adaptation.
    ax.plot(image_indices, np.asarray(oaims_seq) * 100.0, marker='o', markersize=7, linestyle='-',
            color='#1f77b4', linewidth=2.5, label='OAIS-Net')

    ax.set_xlabel('Image Sequence Index', fontsize=16, fontweight='bold')
    ax.set_ylabel('Zero-click Initial IoU (%)', fontsize=16, fontweight='bold')
    ax.set_title('Zero-click Initial IoU in Sequential EM Images', fontsize=18, fontweight='bold',
                 pad=15)

    ax.set_xticks(image_indices)
    ax.set_xlim(0.5, num_images + 0.5)
    ax.set_ylim(40, 100)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2,
              frameon=True, shadow=False, fontsize=14, borderpad=0.8)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "zero_click_IoU_sequence.png")
    plt.savefig(save_path, format='png', bbox_inches='tight')
    plt.savefig(os.path.splitext(save_path)[0] + '.pdf', format='pdf', bbox_inches='tight')
    plt.close('all')
    print(f"📈 序列自适应折线图已生成，保存至: {save_path}")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ================= 机器人点击机制 =================
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


def load_ready_model():
    print(f"📦 正在从 {MODEL_PATH} 加载权重...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = initialize_model()
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    return model.to(DEVICE).eval()


# ================= 单轮连续测试引擎 =================
def run_dataset_evaluation(adaptor, img_paths, target_iou, max_clicks, sub_exp_dir, use_online_learning=True,
                           mode_name="Base"):
    adaptor.reset_weights()  # 测试起点：基座归零
    total_clicks_list = []
    zero_click_iou_list = []
    detail_rows = []

    for img_idx, img_path in enumerate(img_paths):
        # ★ 跨图绝杀：清空上一张图残余的 Adam 动量，但不清空 Encoder 记忆！
        if use_online_learning:
            adaptor.clear_momentum()

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        possible_masks = glob.glob(os.path.join(MASK_DIR, base_name + ".*"))
        if not possible_masks: continue

        gt_mask = (cv2.imread(possible_masks[0], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        img_np = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0

        robot = RobotClicker(gt_mask)
        clicks_used = 0
        all_clicks = []
        points_tensor = torch.full((1, 20, 3), -1.0, device=DEVICE)

        with torch.no_grad():
            out = adaptor.model(image_tensor, points_tensor, training=False)
            curr_pred_mask = (torch.sigmoid(out['instances']) > 0.5).cpu().numpy()[0, 0].astype(np.uint8)
        current_iou = calculate_iou(curr_pred_mask, gt_mask)
        zero_click_iou = current_iou
        zero_click_iou_list.append(zero_click_iou)

        # 交互纠错循环
        while current_iou < target_iou and clicks_used < max_clicks:
            click = robot.get_next_click(curr_pred_mask)
            if click is None: break

            all_clicks.append(click)
            points_tensor = append_click_to_tensor(points_tensor, click, clicks_used)
            clicks_used += 1

            if use_online_learning:
                adaptor.adapt_MI(image_tensor, points_tensor, all_clicks)  # 短时剧烈微调 (KD + CCG)

            with torch.no_grad():
                out = adaptor.model(image_tensor, points_tensor, training=False)
                curr_pred_mask = (torch.sigmoid(out['instances']) > 0.5).cpu().numpy()[0, 0].astype(np.uint8)

            current_iou = calculate_iou(curr_pred_mask, gt_mask)

        # Consolidate the final prediction once, after interaction ends.
        if use_online_learning and clicks_used > 0:
            final_pred_mask = curr_pred_mask.copy()
            adaptor.adapt_PI(image_tensor, points_tensor, final_pred_mask, max_clicks=10)

        final_clicks = clicks_used if current_iou >= target_iou else max_clicks
        total_clicks_list.append(final_clicks)
        detail_rows.append({
            "image_index": img_idx + 1,
            "image": os.path.basename(img_path),
            "zero_click_iou": zero_click_iou,
            "final_iou": current_iou,
            "clicks_used": final_clicks,
        })

        print(
            f"  -> [{mode_name}] 图片 {img_idx + 1}/{len(img_paths)} "
            f"0-click IoU: {zero_click_iou:.4f}, 最终 IoU: {current_iou:.4f}, "
            f"点击: {final_clicks:02d}")

    return np.mean(zero_click_iou_list), zero_click_iou_list, detail_rows


# ================= 主函数 =================
def main():
    global IMAGE_DIR, MASK_DIR, MODEL_PATH, MAX_CLICKS

    parser = argparse.ArgumentParser(
        description="Evaluate zero-click initial IoU during sequential STA/LTM adaptation."
    )
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--mask-dir", default=MASK_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--output-root", default="test_result")
    parser.add_argument("--max-clicks", type=int, default=MAX_CLICKS)
    parser.add_argument(
        "--plot-step", type=int, default=PLOT_STEP,
        help="Plot every Nth sequence image; CSV files always retain all images.",
    )
    parser.add_argument("--target-iou", type=float, default=TARGET_IOUS[0])
    args = parser.parse_args()

    IMAGE_DIR = args.image_dir
    MASK_DIR = args.mask_dir
    MODEL_PATH = args.model_path
    MAX_CLICKS = args.max_clicks
    TARGET_IOUS[:] = [args.target_iou]

    path_parts = os.path.normpath(IMAGE_DIR).split(os.sep)
    dataset_name = path_parts[-3] if len(path_parts) >= 3 else "Dataset"
    root_dir = args.output_root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_exp_dir = os.path.join(root_dir, f"Results_{dataset_name}_SingleRun_{timestamp}")
    os.makedirs(sub_exp_dir, exist_ok=True)

    def log_write(text):
        print(text)
        with open(os.path.join(sub_exp_dir, "test_log.txt"), "a", encoding="utf-8") as f:
            f.write(text + "\n")

    img_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.*")), key=natural_sort_key)
    model = load_ready_model()
    adaptor = OAIMSAdaptor(model)
    all_results = []

    log_write("=" * 70)
    log_write("🚀 启动单轮连续自适应零点击 IoU 测试")
    log_write("=" * 70)

    for target_iou in TARGET_IOUS:
        log_write(f"\n🎯 正在评测指标: zero-click initial IoU (interaction target: {target_iou:.2f})")
        log_write("-" * 50)

        # 1. 跑 Baseline (拿参照心电图)
        log_write("▶ 运行 Base Model (静态基线，无记忆)...")
        base_avg, base_seq, base_rows = run_dataset_evaluation(
            adaptor, img_paths, target_iou, MAX_CLICKS, sub_exp_dir,
            use_online_learning=False, mode_name="Base"
        )
        log_write(f"   => Base Model 平均 zero-click IoU: {base_avg:.4f}")

        # 2. 跑 OAIMS (拿断崖下降图，只跑一轮！)
        log_write("\n▶ 运行 OAIMS 模型 (开启在线进化，单向穿透测试)...")
        oaims_avg, oaims_seq, oaims_rows = run_dataset_evaluation(
            adaptor, img_paths, target_iou, MAX_CLICKS, sub_exp_dir,
            use_online_learning=True, mode_name="OAIS-Net"
        )
        log_write(f"   => OAIS-Net 平均 zero-click IoU: {oaims_avg:.4f}")

        # 3. 极简 CSV 记录
        improvement = oaims_avg - base_avg
        all_results.append({
            "Target_IoU": f"{target_iou:.2f}",
            "Base_zero_click_IoU": f"{base_avg:.4f}",
            "OAIS-Net_zero_click_IoU": f"{oaims_avg:.4f}",
            "Gain": f"{improvement:+.4f}"
        })

        # 4. Generate the zero-click IoU sequence figure.
        plot_sequential_adaptation(
            base_seq, oaims_seq, target_iou, sub_exp_dir,
            plot_step=max(1, args.plot_step),
        )

        write_csv(os.path.join(sub_exp_dir, "base_zero_click_iou.csv"), base_rows)
        write_csv(os.path.join(sub_exp_dir, "oais_zero_click_iou.csv"), oaims_rows)

        summary_rows = []
        for image_index, img_path in enumerate(img_paths):
            summary_rows.append({
                "image_index": image_index + 1,
                "image": os.path.basename(img_path),
                "base_zero_click_iou": base_seq[image_index],
                "oais_zero_click_iou": oaims_seq[image_index],
                "gain": oaims_seq[image_index] - base_seq[image_index],
            })
        write_csv(os.path.join(sub_exp_dir, "zero_click_iou_summary.csv"), summary_rows)

        log_write("=" * 50)

    # 导出精简版 CSV
    csv_filename = os.path.join(sub_exp_dir, f"metrics_summary_{dataset_name}.csv")
    fieldnames = ["Target_IoU", "Base_zero_click_IoU", "OAIS-Net_zero_click_IoU", "Gain"]
    with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    log_write(f"\n✅ 评估彻底完成！请立刻去 {sub_exp_dir} 文件夹查看你的降维打击折线图！")


if __name__ == "__main__":
    main()
