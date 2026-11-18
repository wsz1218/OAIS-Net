"""Build the existing OAIS-Net architecture without machine-specific paths."""
from pathlib import Path

import torch
from torch import nn
import segmentation_models_pytorch as smp

from seg.model.is_deepPI_model import DeepPI
from seg.model import initializer


def resolve_device(requested="auto"):
    if requested in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this PyTorch installation.")
    return device


def read_checkpoint(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_unet(encoder_checkpoint=None):
    unet = smp.Unet("resnet50", in_channels=1, encoder_weights=None, classes=1)
    unet.encoder.conv1 = nn.Conv2d(65, 64, 3, stride=1, padding=1, bias=False)
    block = unet.decoder.blocks[3]
    block.conv1[0] = nn.Conv2d(128, 64, 3, padding=1, bias=False)
    block.conv1[1] = nn.BatchNorm2d(64)
    block.conv2[0] = nn.Conv2d(64, 64, 3, padding=1, bias=False)
    block.conv2[1] = nn.BatchNorm2d(64)
    unet.decoder.blocks = unet.decoder.blocks[:4]
    unet.segmentation_head = nn.Sequential()
    if encoder_checkpoint:
        checkpoint = read_checkpoint(encoder_checkpoint)
        state = checkpoint.get("state_dict", checkpoint)
        own = unet.encoder.state_dict()
        matched = {}
        for name, value in state.items():
            for prefix in ("module.", "feature_extractor.unet.", "encoder_q.", "encoder."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            if name in own and value.shape == own[name].shape:
                matched[name] = value
        if len(matched) < 10:
            raise ValueError("Too few matching ResNet encoder tensors in the pretraining checkpoint.")
        unet.encoder.load_state_dict(matched, strict=False)
        print(f"Loaded {len(matched)}/{len(own)} encoder tensors; the 65-channel input stem stays initialized.")
    return unet


def initialize_model(device="auto", encoder_checkpoint=None):
    model = DeepPI(with_aux_output=False, use_leaky_relu=True, use_rgb_conv=False,
                   use_disks=True, with_prev_mask=False)
    model.apply(initializer.XavierGluon(rnd_type="gaussian", magnitude=2.0))
    model.feature_extractor.unet = load_unet(encoder_checkpoint)
    return model.to(resolve_device(device))


def load_model_weights(model, checkpoint):
    state = checkpoint.get("state_dict", checkpoint)
    state = {name.removeprefix("module."): value for name, value in state.items()}
    model.load_state_dict(state, strict=True)
