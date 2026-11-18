"""Aggregate zero-click IoU results from the four EM datasets into one figure."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_ROOT = Path("test_result") / "A零点击"
OUTPUT_DIR = RESULT_ROOT

DATASETS = [
    ("HeLa Cells", "Results_domain_2_cells_SingleRun_*", 1),
    ("Mouse Skeletal Muscle", "Results_domain_3_muscle_SingleRun_*", 2),
    ("Nanoparticles TEM", "Results_domain_4_fuzzy_blobs_SingleRun_*", 2),
    ("Lucchi++", "Results_domain_6_Lucchi_SingleRun_*", 2),
]


def read_iou_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_result_dir(pattern):
    matches = sorted(RESULT_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No result directory matches: {pattern}")
    return matches[-1]


def select_indices(count, plot_step):
    if count == 0:
        return np.array([], dtype=int)
    selected = np.arange(0, count, plot_step, dtype=int)
    if selected[-1] != count - 1:
        selected = np.append(selected, count - 1)
    return selected


def load_dataset(pattern, plot_step):
    result_dir = find_result_dir(pattern)
    base_rows = read_iou_csv(result_dir / "base_zero_click_iou.csv")
    oais_rows = read_iou_csv(result_dir / "oais_zero_click_iou.csv")
    count = min(len(base_rows), len(oais_rows))
    if count == 0:
        raise ValueError(f"No rows found in {result_dir}")

    selected = select_indices(count, plot_step)
    base_values = np.array(
        [float(base_rows[index]["zero_click_iou"]) * 100.0 for index in selected]
    )
    oais_values = np.array(
        [float(oais_rows[index]["zero_click_iou"]) * 100.0 for index in selected]
    )
    x_values = selected + 1
    return x_values, base_values, oais_values, count


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
    })

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=300)
    axes = axes.ravel()
    colors = {"base": "#d62728", "oais": "#1f77b4"}

    for axis, (title, pattern, plot_step) in zip(axes, DATASETS):
        x_values, base_values, oais_values, total_count = load_dataset(
            pattern, plot_step
        )
        axis.plot(
            x_values, base_values, color=colors["base"], linestyle="--",
            marker="s", markersize=7, linewidth=2.4,
            label="Baseline (Static Model)",
        )
        axis.plot(
            x_values, oais_values, color=colors["oais"], linestyle="-",
            marker="o", markersize=7, linewidth=2.6,
            label="OAIS-Net (Online Adaptive)",
        )
        axis.set_title(title, fontsize=18, fontweight="bold", pad=12)
        axis.set_xlabel("Image Sequence Index", fontsize=15, fontweight="bold", labelpad=7)
        axis.set_ylabel("Zero-click Initial IoU (%)", fontsize=15, fontweight="bold", labelpad=8)
        axis.set_ylim(40, 100)
        axis.set_xlim(0.5, max(x_values) + 0.5)
        axis.set_xticks(x_values)
        axis.tick_params(axis="both", labelsize=12, width=1.2, length=5)
        axis.grid(True, color="#d9d9d9", linewidth=0.9, alpha=0.75)
        for spine in axis.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#555555")
        print(f"{title}: plotted {len(x_values)} points from {total_count} images")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=2, fontsize=14, frameon=True,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.8, h_pad=3.0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "zero_click_IoU_all_datasets_no_shading.png"
    pdf_path = OUTPUT_DIR / "zero_click_IoU_all_datasets_no_shading.pdf"
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
