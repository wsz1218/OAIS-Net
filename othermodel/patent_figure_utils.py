import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def natural_sort_key(value):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"([0-9]+)", str(value))
    ]


def get_font(size=22):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ]
    for font_path in candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def save_patent_npz(img_gray, gt_mask, pred_mask, clicks, img_name, step, mode, iou, output_root):
    out_dir = Path(output_root) / "patent_npz" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    clicks_arr = (
        np.array(clicks, dtype=np.int32)
        if len(clicks) > 0
        else np.zeros((0, 3), dtype=np.int32)
    )
    save_path = out_dir / f"{img_name}_step{step}.npz"
    np.savez_compressed(
        save_path,
        img=img_gray.astype(np.float32),
        gt=(gt_mask > 0).astype(np.uint8),
        pred=(pred_mask > 0).astype(np.uint8),
        clicks=clicks_arr,
        iou=np.array(iou, dtype=np.float32),
    )
    return save_path


def make_bw_result_panel(img_gray, gt_mask, pred_mask, clicks=None, size=320):
    img = (img_gray * 255).astype(np.uint8) if img_gray.max() <= 1 else img_gray.astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    panel = img_rgb.copy()

    panel[pred == 1] = (
        panel[pred == 1] * 0.65 + np.array([170, 170, 170]) * 0.35
    ).astype(np.uint8)

    yy, xx = np.indices(pred.shape)
    fp = np.logical_and(pred == 1, gt == 0)
    fn = np.logical_and(pred == 0, gt == 1)
    panel[np.logical_and(fp, (xx - yy) % 7 < 2)] = [0, 0, 0]
    panel[np.logical_and(fn, (xx + yy) % 7 < 2)] = [60, 60, 60]

    contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, (0, 0, 0), 2)

    if clicks is not None:
        for y, x, is_pos in clicks:
            y, x, is_pos = int(y), int(x), int(is_pos)
            if is_pos == 1:
                cv2.circle(panel, (x, y), 5, (0, 0, 0), 2)
            else:
                cv2.line(panel, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 0), 2)
                cv2.line(panel, (x - 5, y + 5), (x + 5, y - 5), (0, 0, 0), 2)

    panel = cv2.resize(panel, (size, size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(panel)


def draw_centered_lines(draw, x, y, box_w, lines, font, line_gap=4):
    cur_y = y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x + (box_w - tw) // 2, cur_y), line, fill="black", font=font)
        cur_y += th + line_gap


def make_patent_comparison(
    reference_npz,
    method_npz,
    save_path,
    fig_title,
    reference_label="Base model",
    method_label="Comparison method",
):
    reference = np.load(reference_npz)
    method = np.load(method_npz)

    img = reference["img"]
    gt = reference["gt"]
    reference_pred = reference["pred"]
    method_pred = method["pred"]
    clicks = method["clicks"]
    reference_iou = float(reference["iou"]) * 100
    method_iou = float(method["iou"]) * 100

    size = 320
    gap = 24
    margin = 32
    title_h = 58
    label_h = 74

    font_title = get_font(26)
    font_label = get_font(20)

    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
    p1 = Image.fromarray(cv2.resize(img_uint8, (size, size), interpolation=cv2.INTER_AREA)).convert("RGB")
    gt_uint8 = ((gt > 0) * 255).astype(np.uint8)
    p2 = Image.fromarray(cv2.resize(gt_uint8, (size, size), interpolation=cv2.INTER_NEAREST)).convert("RGB")
    p3 = make_bw_result_panel(img, gt, reference_pred, clicks=None, size=size)
    p4 = make_bw_result_panel(img, gt, method_pred, clicks=clicks, size=size)

    panels = [p1, p2, p3, p4]
    labels = [
        ["(a) Original TEM image"],
        ["(b) Ground truth"],
        [f"(c) {reference_label}", f"IoU={reference_iou:.2f}%"],
        [f"(d) {method_label}", f"IoU={method_iou:.2f}%"],
    ]

    width = margin * 2 + size * 4 + gap * 3
    height = title_h + size + label_h + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), fig_title, fill="black", font=font_title)

    x = margin
    y = title_h
    for panel, label_lines in zip(panels, labels):
        canvas.paste(panel, (x, y))
        draw.rectangle((x, y, x + size - 1, y + size - 1), outline="black", width=2)
        draw_centered_lines(draw, x, y + size + 8, size, label_lines, font_label)
        x += size + gap

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path, dpi=(300, 300))
    return save_path


def generate_patent_figures(
    output_root,
    method_mode,
    step=10,
    fig_title="TEM interactive segmentation result",
    reference_mode="Base",
    reference_dir=None,
    reference_label="Base model",
    method_label=None,
):
    output_root = Path(output_root)
    method_label = method_label or method_mode
    method_dir = output_root / "patent_npz" / method_mode
    reference_dir = Path(reference_dir) if reference_dir else output_root / "patent_npz" / reference_mode
    patent_dir = output_root / "patent_figures" / method_mode

    if not method_dir.is_dir():
        print(f"[Patent Figure] method npz dir not found: {method_dir}")
        return 0
    if not reference_dir.is_dir():
        print(f"[Patent Figure] reference npz dir not found: {reference_dir}")
        print("[Patent Figure] npz files were saved, but comparison figures were skipped.")
        return 0

    patent_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for method_npz in sorted(method_dir.glob(f"*_step{step}.npz"), key=lambda p: natural_sort_key(p.name)):
        reference_npz = reference_dir / method_npz.name
        if not reference_npz.exists():
            continue
        out_png = patent_dir / method_npz.name.replace(".npz", "_patent_bw_iou.png")
        make_patent_comparison(
            reference_npz=reference_npz,
            method_npz=method_npz,
            save_path=out_png,
            fig_title=fig_title,
            reference_label=reference_label,
            method_label=method_label,
        )
        count += 1

    print(f"[Patent Figure] generated {count} figures in: {patent_dir}")
    return count
