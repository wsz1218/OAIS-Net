import argparse
import os

import torch

from gui.callbacks import set_dataset_roots, set_device
from gui.layout import build_gradio_ui


def parse_args():
    parser = argparse.ArgumentParser(description="Launch OAIS-Net interactive segmentation GUI.")
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=None,
        help=(
            "Dataset root or image folder to scan. Can be used more than once. "
            "Default: pi_seg/data/datasets. You can also set OAIMS_DATASET_ROOTS, "
            "separated by os.pathsep."
        ),
    )
    parser.add_argument("--server-name", default="127.0.0.1", help="Gradio server host.")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    return parser.parse_args()


def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


ARGS = parse_args()
DEVICE = resolve_device()
set_device(DEVICE)

dataset_roots = ARGS.dataset_root
if dataset_roots is None and os.environ.get("OAIMS_DATASET_ROOTS"):
    dataset_roots = os.environ["OAIMS_DATASET_ROOTS"].split(os.pathsep)
set_dataset_roots(dataset_roots or ["pi_seg/data/datasets"])

CANVAS_SIZE = (640, 640)
POINT_COLORS = [(0, 0.85, 0.2), (1, 0.15, 0.1), (1, 0.85, 0.1)]

demo = build_gradio_ui(DEVICE, CANVAS_SIZE, POINT_COLORS)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        server_name=ARGS.server_name,
        server_port=ARGS.server_port,
        share=ARGS.share,
    )
