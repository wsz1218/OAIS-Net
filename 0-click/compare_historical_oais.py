"""Compare the historical Lucchi++ OAIS run with the newer ablation runs.

The source protocols differ. Outputs are descriptive and must not be used as a
matched-protocol ablation without rerunning the historical arm.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parent
OLD_RUN = "Results_domain_6_Lucchi_SingleRun_20260907_164116"
NEW_RUN = "LucchiPP_zero_click_ablation_20261109_160946"
OUTPUT_DIR = EXPERIMENT_DIR / "results" / "LucchiPP_mixed_protocol_comparison"


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def source_paths():
    matches = list(ROOT.glob(f"test_result/*/{OLD_RUN}/zero_click_iou_summary.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one historical Lucchi++ result, found {len(matches)}")
    new_path = EXPERIMENT_DIR / "results" / NEW_RUN / "per_image_zero_click_iou.csv"
    if not new_path.is_file():
        raise FileNotFoundError(new_path)
    return matches[0], new_path


def combine_rows(old_rows, new_rows):
    if not old_rows or len(old_rows) != len(new_rows):
        raise ValueError("The two runs must contain the same nonempty image sequence")

    combined = []
    for old, new in zip(old_rows, new_rows):
        if (old["image_index"], old["image"]) != (new["image_index"], new["image"]):
            raise ValueError(f"Image sequence mismatch: {old['image']} vs {new['image']}")
        row = {
            "index": int(new["image_index"]),
            "image": new["image"],
            "Base": float(new["Base"]),
            "STA-only": float(new["STA-only"]),
            "LTM-only": float(new["LTM-only"]),
            "Historical OAIS-Net": float(old["oais_zero_click_iou"]),
        }
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0
               for key, value in row.items() if key not in ("index", "image")):
            raise ValueError(f"Invalid IoU in image {row['image']}")
        combined.append(row)
    return combined


def write_table(rows, old_path, new_path):
    columns = ("Base", "STA-only", "LTM-only", "Historical OAIS-Net")
    lines = [
        "# Lucchi++ zero-click IoU by image",
        "",
        "| Index | Image | Base (%) | STA-only (%) | LTM-only (%) | Historical OAIS-Net (%) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = " | ".join(f"{row[name] * 100:.2f}" for name in columns)
        lines.append(f"| {row['index']} | {row['image']} | {metrics} |")

    lines.extend([
        "",
        "## Following images (2-20)",
        "",
        "| Mode | Mean IoU (%) | Image SD (percentage points) |",
        "|---|---:|---:|",
    ])
    for name in columns:
        values = np.asarray([row[name] for row in rows[1:]], dtype=float) * 100.0
        lines.append(f"| {name} | {values.mean():.2f} | {values.std(ddof=0):.2f} |")

    lines.extend([
        "",
        "**Protocol warning:** Historical OAIS-Net used GT-supervised, per-click PI; "
        "the newer arms used prediction-supervised, end-of-image LTM where applicable. "
        "This is a descriptive juxtaposition, not a matched-protocol ablation.",
        "",
        f"Historical source: `{old_path.relative_to(ROOT).as_posix()}`  ",
        f"New source: `{new_path.relative_to(ROOT).as_posix()}`",
        "",
    ])
    (OUTPUT_DIR / "per_image_zero_click_iou_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def plot(rows):
    x = np.asarray([row["index"] for row in rows])
    styles = (
        ("Base", "#c43c39", "--", "s"),
        ("STA-only", "#287f59", "-.", "^"),
        ("LTM-only", "#7859a6", ":", "D"),
        ("Historical OAIS-Net*", "#17639b", "-", "o"),
    )
    fig, ax = plt.subplots(figsize=(13.2, 6.8), dpi=300)
    for label, color, linestyle, marker in styles:
        key = label.rstrip("*")
        y = np.asarray([row[key] for row in rows]) * 100.0
        ax.plot(x, y, label=label, color=color, linestyle=linestyle,
                marker=marker, markersize=5.5, linewidth=2.1)

    ax.set(xlim=(0.5, len(rows) + 0.5), ylim=(70, 100),
           xlabel="Image sequence index", ylabel="Zero-click initial IoU (%)")
    ax.set_xticks(x)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    fig.suptitle("Lucchi++ zero-click IoU by image", fontsize=17, y=0.985)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=4,
               frameon=False, fontsize=11)
    fig.text(0.08, 0.06,
             "* Historical OAIS-Net: GT-supervised per-click PI. New arms: "
             "prediction-supervised end-of-image LTM where applicable.\n"
             "Mixed protocols; curves are descriptive, not a matched ablation.",
             ha="left", va="bottom", fontsize=9.5, color="#333333")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.19)
    fig.savefig(OUTPUT_DIR / "zero_click_iou_mixed_protocol.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "zero_click_iou_mixed_protocol.pdf")
    plt.close(fig)


def main():
    old_path, new_path = source_paths()
    rows = combine_rows(read_rows(old_path), read_rows(new_path))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_table(rows, old_path, new_path)
    plot(rows)
    print(f"Compared {len(rows)} aligned images; outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
