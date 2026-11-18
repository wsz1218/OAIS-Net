"""Plot four Lucchi++ zero-click IoU series from the selected result files."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "零点击消融实验" / "results"
SOURCES = {
    "history": ROOT / "test_result" / "A零点击" /
    "Results_domain_6_Lucchi_SingleRun_20260907_164116" /
    "zero_click_iou_summary.csv",
    "sta": RESULTS / "LucchiPP_zero_click_ablation_20261109_160946" /
    "per_image_zero_click_iou.csv",
    "ltm": RESULTS / "LucchiPP_zero_click_table3_20261109_175612" /
    "per_image_zero_click_iou.csv",
}
OUTPUT = RESULTS / "LucchiPP_zero_click_four_modes"


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    history = read_rows(SOURCES["history"])
    sta = read_rows(SOURCES["sta"])
    ltm = read_rows(SOURCES["ltm"])
    if not history or len(history) != len(sta) or len(history) != len(ltm):
        raise ValueError("Source files have different numbers of images")

    for i, (h, s, l) in enumerate(zip(history, sta, ltm), start=1):
        expected = (str(i), h["image"])
        if (h["image_index"], h["image"]) != expected:
            raise ValueError(f"Historical image order is inconsistent at {i}")
        if (s["image_index"], s["image"]) != expected:
            raise ValueError(f"STA-only image does not match at {i}")
        if (l["image_index"], l["image"]) != expected:
            raise ValueError(f"LTM-only image does not match at {i}")

    series = {
        "Base": [float(row["base_zero_click_iou"]) * 100 for row in history],
        "STA-only": [float(row["STA-only"]) * 100 for row in sta],
        "LTM-only": [float(row["LTM-only"]) * 100 for row in ltm],
        "OAIS-Net": [float(row["oais_zero_click_iou"]) * 100 for row in history],
    }
    if any(not 0 <= value <= 100 for values in series.values() for value in values):
        raise ValueError("IoU values must be in the range [0, 1] before scaling")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.linewidth": 0.9,
        "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(11.2, 5.7), dpi=300)
    x = list(range(1, len(history) + 1))
    styles = {
        "Base": ("#d53a36", "--", "s"),
        "STA-only": ("#258259", "-.", "^"),
        "LTM-only": ("#8258a6", ":", "D"),
        "OAIS-Net": ("#1769a6", "-", "o"),
    }
    for name, values in series.items():
        color, line, marker = styles[name]
        ax.plot(x, values, label=name, color=color, linestyle=line,
                marker=marker, markersize=4.5, linewidth=2.0)

    ax.set_xlim(0.5, len(x) + 0.5)
    ax.set_ylim(40, 100)
    ax.set_xticks(list(range(1, len(x) + 1, 2)) + [len(x)])
    ax.set_yticks(range(40, 101, 10))
    ax.set_xlabel("Image sequence index", fontsize=13)
    ax.set_ylabel("Zero-click initial IoU (%)", fontsize=13)
    ax.grid(True, color="#e3e3e3", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10)
    fig.suptitle("Lucchi++", fontsize=16, y=0.98)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=4, frameon=False, fontsize=11, columnspacing=1.6)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.82)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Plotted {len(x)} aligned images: {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
