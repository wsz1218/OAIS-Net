import os
import time
import re
from copy import deepcopy
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from model import initialize_model
from pi_seg.inference.oaims_adaptor_KD_EWC_IAIoU import OAIMSAdaptor


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
POINT_SLOTS_PER_CLASS = 10
POS_COLOR = (0, 220, 50)
NEG_COLOR = (255, 40, 30)
MASK_BLUE = np.array([40, 130, 255], dtype=np.float32)

BASE_MODEL = None
OAIS_MODEL = None
OAIS_CORE = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DEVICE = "cpu"
MODEL_PATH = None


def rel_path(path):
    try:
        return os.path.relpath(path, os.getcwd()).replace("\\", "/")
    except ValueError:
        return path.replace("\\", "/")


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def dropdown_choices(paths):
    return [(rel_path(path), path) for path in paths]


def scan_model_checkpoints():
    roots = ["save", "SAVE_DIR"]
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for current_root, _, files in os.walk(root):
            for name in files:
                lower = name.lower()
                if lower.endswith(".pth") or lower.endswith(".pth.tar"):
                    found.append(os.path.abspath(os.path.join(current_root, name)))
    return dropdown_choices(sorted(set(found), key=lambda p: rel_path(p).lower()))


def has_images(folder):
    try:
        return any(os.path.splitext(name)[1].lower() in VALID_IMAGE_EXTS for name in os.listdir(folder))
    except OSError:
        return False


def scan_image_folders():
    roots = ["pi_seg/data/datasets"]
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for current_root, dirs, _ in os.walk(root):
            if os.path.basename(current_root).lower() == "input" and has_images(current_root):
                found.append(os.path.abspath(current_root))
            if "input" in dirs:
                input_dir = os.path.join(current_root, "input")
                if has_images(input_dir):
                    found.append(os.path.abspath(input_dir))
    return dropdown_choices(sorted(set(found), key=lambda p: rel_path(p).lower()))


def load_state_dict_compatible(model, checkpoint):
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    clean_state = {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
    if isinstance(model, nn.DataParallel):
        clean_state = {f"module.{key}": value for key, value in clean_state.items()}
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    return len(missing), len(unexpected)


def active_module(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def load_model_instance(checkpoint, target_device):
    model = initialize_model()
    load_state_dict_compatible(model, checkpoint)
    model.to(target_device).eval()
    return model


def load_models(model_path):
    global BASE_MODEL, OAIS_MODEL, OAIS_CORE, MODEL_PATH

    if not model_path or not os.path.exists(model_path):
        return "Model checkpoint does not exist."

    try:
        start = time.perf_counter()
        checkpoint = torch.load(model_path, map_location=DEVICE)
        BASE_MODEL = load_model_instance(checkpoint, BASE_DEVICE)
        OAIS_MODEL = load_model_instance(checkpoint, DEVICE)
        OAIS_CORE = active_module(OAIS_MODEL)
        OAIS_CORE.oaims_adaptor = OAIMSAdaptor(OAIS_CORE)
        MODEL_PATH = model_path
        elapsed = time.perf_counter() - start
        return (
            f"Loaded paired models in {elapsed:.1f}s: Base on {BASE_DEVICE}, "
            f"OAIS-Net on {DEVICE}. Checkpoint: {os.path.basename(model_path)}"
        )
    except Exception as exc:
        BASE_MODEL = None
        OAIS_MODEL = None
        OAIS_CORE = None
        return f"Failed to load paired models: {exc}"


def refresh_models():
    choices = scan_model_checkpoints()
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value), f"Found {len(choices)} checkpoints."


def refresh_folders():
    choices = scan_image_folders()
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value), f"Found {len(choices)} image folders."


def load_folder(folder_path):
    if not folder_path or not os.path.isdir(folder_path):
        return gr.update(choices=[], value=None), [], "Image folder does not exist."

    paths = [
        os.path.abspath(os.path.join(folder_path, name))
        for name in sorted(os.listdir(folder_path), key=natural_key)
        if os.path.splitext(name)[1].lower() in VALID_IMAGE_EXTS
    ]
    if not paths:
        return gr.update(choices=[], value=None), [], "No supported image files were found."
    return gr.update(choices=paths, value=paths[0]), paths, f"Loaded {len(paths)} images."


def read_rgb_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def empty_mask_like(image):
    return np.zeros(image.shape[:2], dtype=np.uint8)


def mask_to_rgb(mask):
    if mask is None:
        return None
    mask_2d = mask[:, :, 0] if getattr(mask, "ndim", 0) == 3 else mask
    return np.stack([mask_2d.astype(np.uint8)] * 3, axis=-1)


def load_image(image_path):
    if not image_path or not os.path.exists(image_path):
        return None, None, None, None, None, [], 0, "Image path does not exist."

    try:
        reset_oais_short_term_branch()
        image = read_rgb_image(image_path)
        empty = empty_mask_like(image)
        empty_rgb = mask_to_rgb(empty)
        return image, image.copy(), empty_rgb, empty_rgb, empty, [], 0, f"Ready: {os.path.basename(image_path)}"
    except Exception as exc:
        return None, None, None, None, None, [], 0, f"Failed to load image: {exc}"


def move_image(current_path, image_paths, step):
    if not image_paths:
        return gr.update(), None, None, None, None, None, [], 0, "No image list is loaded."
    if current_path not in image_paths:
        next_path = image_paths[0]
    else:
        next_path = image_paths[(image_paths.index(current_path) + step) % len(image_paths)]
    loaded = load_image(next_path)
    return gr.update(value=next_path), *loaded


def previous_image(current_path, image_paths):
    return move_image(current_path, image_paths, -1)


def next_image(current_path, image_paths):
    return move_image(current_path, image_paths, 1)


def image_to_tensor(original_img, target_device):
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    gray = np.expand_dims(gray, axis=-1)
    return T.functional.to_tensor(gray).type(torch.float32).unsqueeze(0).to(target_device)


def points_tensor_from_clicks(clicks, target_device=DEVICE):
    pos = [(int(point[1]), int(point[0]), 1.0) for point, label in clicks if label == 1]
    neg = [(int(point[1]), int(point[0]), 1.0) for point, label in clicks if label == 0]
    pos = pos[:POINT_SLOTS_PER_CLASS] + [(-1, -1, -1)] * max(0, POINT_SLOTS_PER_CLASS - len(pos))
    neg = neg[:POINT_SLOTS_PER_CLASS] + [(-1, -1, -1)] * max(0, POINT_SLOTS_PER_CLASS - len(neg))
    return torch.tensor([pos + neg], dtype=torch.float32, device=target_device)


def all_clicks_for_adaptor(clicks):
    return [(int(point[1]), int(point[0]), int(label)) for point, label in clicks]


def forward_mask(model, image_tensor, points_tensor):
    with torch.no_grad():
        output = model(image_tensor, points_tensor, training=False)
        logits = output["instances"] if isinstance(output, dict) and "instances" in output else output
        return (torch.sigmoid(logits).detach().cpu().numpy()[0, 0] > 0.5).astype(np.uint8) * 255


def reset_oais_branch():
    if OAIS_CORE is not None and hasattr(OAIS_CORE, "oaims_adaptor"):
        OAIS_CORE.oaims_adaptor.reset_weights()


def reset_oais_short_term_branch():
    if OAIS_CORE is None or not hasattr(OAIS_CORE, "oaims_adaptor"):
        return
    adaptor = OAIS_CORE.oaims_adaptor
    decoder_ids = {id(param) for param in getattr(adaptor, "decoder_params", [])}
    if not decoder_ids or not hasattr(adaptor, "initial_state_dict"):
        return
    with torch.no_grad():
        for name, param in OAIS_CORE.named_parameters():
            if id(param) in decoder_ids and name in adaptor.initial_state_dict:
                param.copy_(adaptor.initial_state_dict[name])
    if hasattr(adaptor, "optimizer"):
        adaptor.optimizer.state.clear()


def draw_clicks(original_img, clicks):
    if original_img is None:
        return None
    image = original_img.copy()
    for point, label in clicks:
        color = POS_COLOR if label == 1 else NEG_COLOR
        cv2.circle(image, tuple(map(int, point)), 4, color, -1)
        cv2.circle(image, tuple(map(int, point)), 8, color, 1)
    return image


def draw_overlay(original_img, mask, clicks):
    if original_img is None:
        return None
    image = original_img.copy().astype(np.float32)
    if mask is not None:
        alpha = (mask > 0).astype(np.float32)[..., None] * 0.30
        image = image * (1.0 - alpha) + MASK_BLUE * alpha
    return draw_clicks(np.clip(image, 0, 255).astype(np.uint8), clicks)


def segment_base(original_img, clicks):
    try:
        if BASE_MODEL is None:
            return None
        image_tensor = image_to_tensor(original_img, BASE_DEVICE)
        points_tensor = points_tensor_from_clicks(clicks, BASE_DEVICE)
        return forward_mask(BASE_MODEL, image_tensor, points_tensor)
    except Exception:
        return None


def segment_oais_new_click(original_img, clicks):
    try:
        if OAIS_MODEL is None or OAIS_CORE is None:
            return None
        image_tensor = image_to_tensor(original_img, DEVICE)
        points_tensor = points_tensor_from_clicks(clicks, DEVICE)
        OAIS_CORE.oaims_adaptor.adapt_MI(image_tensor, points_tensor, all_clicks_for_adaptor(clicks))
        return forward_mask(OAIS_MODEL, image_tensor, points_tensor)
    except Exception:
        return None


def replay_oais(original_img, clicks):
    if OAIS_MODEL is None or OAIS_CORE is None or not clicks:
        return None
    reset_oais_short_term_branch()
    mask = None
    for idx in range(1, len(clicks) + 1):
        prefix = clicks[:idx]
        mask = segment_oais_new_click(original_img, prefix)
    return mask


def make_status(clicks, elapsed, mode):
    return (
        f"{mode} | clicks={len(clicks)} | elapsed={elapsed:.2f}s | "
        "same manual clicks are applied to Base and OAIS-Net."
    )


def add_click(original_img, clicks, point_type, evt: gr.SelectData):
    if original_img is None:
        return None, None, None, None, clicks, len(clicks), "Load an image first."
    if BASE_MODEL is None or OAIS_MODEL is None:
        overlay = draw_clicks(original_img, clicks)
        return overlay, None, None, None, clicks, len(clicks), "Load a paired model first."
    if evt is None or evt.index is None:
        overlay = draw_clicks(original_img, clicks)
        return overlay, None, None, None, clicks, len(clicks), "No click coordinate received."

    label = 0 if point_type == "negative" else 1
    clicks = list(clicks)
    clicks.append((tuple(evt.index), label))

    start = time.perf_counter()
    base_mask = segment_base(original_img, clicks)
    oais_mask = segment_oais_new_click(original_img, clicks)
    elapsed = time.perf_counter() - start

    input_overlay = draw_clicks(original_img, clicks)
    base_overlay = draw_overlay(original_img, base_mask, clicks)
    oais_overlay = draw_overlay(original_img, oais_mask, clicks)
    return (
        input_overlay,
        base_overlay,
        oais_overlay,
        oais_mask,
        clicks,
        len(clicks),
        make_status(clicks, elapsed, "Click completed"),
    )


def undo_click(original_img, clicks):
    if original_img is None:
        return None, None, None, None, clicks, 0, "Load an image first."
    clicks = list(clicks)
    if clicks:
        clicks.pop()

    start = time.perf_counter()
    if clicks:
        base_mask = segment_base(original_img, clicks)
        oais_mask = replay_oais(original_img, clicks)
        input_overlay = draw_clicks(original_img, clicks)
        base_overlay = draw_overlay(original_img, base_mask, clicks)
        oais_overlay = draw_overlay(original_img, oais_mask, clicks)
    else:
        reset_oais_branch()
        empty = empty_mask_like(original_img)
        empty_rgb = mask_to_rgb(empty)
        base_mask = empty
        oais_mask = empty
        input_overlay = original_img.copy()
        base_overlay = empty_rgb
        oais_overlay = empty_rgb
    elapsed = time.perf_counter() - start
    return input_overlay, base_overlay, oais_overlay, oais_mask, clicks, len(clicks), make_status(clicks, elapsed, "Undo completed")


def clear_clicks(original_img):
    reset_oais_short_term_branch()
    if original_img is None:
        return None, None, None, None, [], 0, "Load an image first."
    empty = empty_mask_like(original_img)
    empty_rgb = mask_to_rgb(empty)
    return original_img.copy(), empty_rgb, empty_rgb, empty, [], 0, "Clicks cleared. OAIS-Net short-term adaptation was reset."


def reset_models_and_clicks(original_img):
    reset_oais_branch()
    if original_img is None:
        return None, None, None, None, [], 0, "OAIS-Net branch reset. Load an image to continue."
    empty = empty_mask_like(original_img)
    empty_rgb = mask_to_rgb(empty)
    return original_img.copy(), empty_rgb, empty_rgb, empty, [], 0, "OAIS-Net branch reset to the loaded checkpoint."


def complete_frame(original_img, oais_mask, clicks):
    if original_img is None or oais_mask is None:
        return "Nothing to complete."
    if OAIS_CORE is None or not hasattr(OAIS_CORE, "oaims_adaptor"):
        return "Load OAIS-Net model first."
    if not clicks:
        return "Add clicks before writing PI memory."

    try:
        image_tensor = image_to_tensor(original_img, DEVICE)
        points_tensor = points_tensor_from_clicks(clicks, DEVICE)
        OAIS_CORE.oaims_adaptor.adapt_PI(image_tensor, points_tensor, oais_mask, max_clicks=10)
        return "PI memory updated from the current OAIS-Net mask."
    except Exception as exc:
        return f"Complete frame failed: {exc}"


def save_comparison(original_img, base_overlay, oais_overlay, image_path, output_dir):
    if original_img is None or base_overlay is None or oais_overlay is None:
        return "No comparison to save."
    output_dir = output_dir or os.path.join("save", "gui_comparisons")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path or "comparison"))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    panel = np.concatenate([original_img, base_overlay, oais_overlay], axis=1)
    save_path = os.path.join(output_dir, f"{base_name}_base_vs_oais_{timestamp}.png")
    cv2.imwrite(save_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    return f"Saved comparison figure to: {save_path}"


def build_ui():
    model_choices = scan_model_checkpoints()
    folder_choices = scan_image_folders()
    default_model = model_choices[0][1] if model_choices else None
    default_folder = folder_choices[0][1] if folder_choices else None

    css = """
    .status-box textarea { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
    .compact-button button { min-height: 38px; }
    """

    with gr.Blocks(css=css, title="OAIS-Net Manual Click Comparison") as demo:
        clicks = gr.State(value=[])
        image_paths = gr.State(value=[])
        original_image = gr.State(value=None)
        current_oais_mask = gr.State(value=None)

        gr.Markdown("## OAIS-Net Manual Click Comparison Workbench")

        with gr.Row():
            model_path = gr.Dropdown(
                choices=model_choices,
                value=default_model,
                label="Model checkpoint",
                interactive=True,
                scale=4,
            )
            load_model_btn = gr.Button("Load Paired Models", variant="primary", scale=1)
            refresh_models_btn = gr.Button("Refresh Models", scale=1)
            reset_branch_btn = gr.Button("Reset OAIS-Net Branch", variant="stop", scale=1)

        with gr.Row():
            folder_input = gr.Dropdown(
                choices=folder_choices,
                value=default_folder,
                label="Image folder",
                interactive=True,
                scale=3,
            )
            load_folder_btn = gr.Button("Load Folder", scale=1)
            refresh_folders_btn = gr.Button("Refresh Folders", scale=1)
            image_dropdown = gr.Dropdown(label="Image sequence", choices=[], interactive=True, scale=3)

        status = gr.Textbox(
            label=f"Status - running on {DEVICE}",
            value="Load paired models and an image folder to start.",
            lines=3,
            interactive=False,
            elem_classes="status-box",
        )

        with gr.Row():
            with gr.Column(scale=5):
                input_image = gr.Image(
                    label="Input + shared manual clicks",
                    type="numpy",
                    height=520,
                    interactive=False,
                    show_download_button=False,
                )
                with gr.Row():
                    click_type = gr.Radio(
                        choices=[("Positive foreground", "positive"), ("Negative background", "negative")],
                        value="positive",
                        label="Click type",
                    )
                    click_count = gr.Number(label="Clicks", value=0, interactive=False)
                with gr.Row(elem_classes="compact-button"):
                    prev_btn = gr.Button("Previous")
                    next_btn = gr.Button("Next")
                    undo_btn = gr.Button("Undo")
                    clear_btn = gr.Button("Clear Clicks")

            with gr.Column(scale=5):
                base_view = gr.Image(
                    label="Base without online adaptation",
                    type="numpy",
                    height=520,
                    interactive=False,
                    show_download_button=True,
                )

            with gr.Column(scale=5):
                oais_view = gr.Image(
                    label="OAIS-Net with online adaptation",
                    type="numpy",
                    height=520,
                    interactive=False,
                    show_download_button=True,
                )

        with gr.Row():
            output_dir = gr.Textbox(label="Output folder", value="save/gui_comparisons", scale=4)
            complete_btn = gr.Button("Complete Frame + PI Memory", variant="primary", scale=1)
            save_btn = gr.Button("Save Comparison Figure", scale=1)

        load_model_btn.click(load_models, [model_path], [status])
        refresh_models_btn.click(refresh_models, [], [model_path, status])
        reset_branch_btn.click(
            reset_models_and_clicks,
            [original_image],
            [input_image, base_view, oais_view, current_oais_mask, clicks, click_count, status],
        )

        refresh_folders_btn.click(refresh_folders, [], [folder_input, status])
        load_folder_btn.click(
            load_folder,
            [folder_input],
            [image_dropdown, image_paths, status],
        ).then(
            load_image,
            [image_dropdown],
            [original_image, input_image, base_view, oais_view, current_oais_mask, clicks, click_count, status],
        )

        image_dropdown.change(
            load_image,
            [image_dropdown],
            [original_image, input_image, base_view, oais_view, current_oais_mask, clicks, click_count, status],
        )

        prev_btn.click(
            previous_image,
            [image_dropdown, image_paths],
            [image_dropdown, original_image, input_image, base_view, oais_view, current_oais_mask, clicks, click_count, status],
        )
        next_btn.click(
            next_image,
            [image_dropdown, image_paths],
            [image_dropdown, original_image, input_image, base_view, oais_view, current_oais_mask, clicks, click_count, status],
        )

        input_image.select(
            add_click,
            [original_image, clicks, click_type],
            [input_image, base_view, oais_view, current_oais_mask, clicks, click_count, status],
            concurrency_limit=1,
            trigger_mode="always_last",
        )

        undo_btn.click(
            undo_click,
            [original_image, clicks],
            [input_image, base_view, oais_view, current_oais_mask, clicks, click_count, status],
        )

        clear_btn.click(
            clear_clicks,
            [original_image],
            [input_image, base_view, oais_view, current_oais_mask, clicks, click_count, status],
        )

        complete_btn.click(complete_frame, [original_image, current_oais_mask, clicks], [status])
        save_btn.click(save_comparison, [original_image, base_view, oais_view, image_dropdown, output_dir], [status])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7861, share=False)
