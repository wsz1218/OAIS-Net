"""Prediction-supervised STA and encoder-only, end-of-image LTM."""
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from seg.inference.oaims_adaptor_KD_EWC_IAIoU import OAIMSAdaptor as LossFunctions


@dataclass
class OnlineSettings:
    sta_lr: float = 1e-4
    ltm_lr: float = 1e-5
    sta_steps: int = 2
    ltm_steps: int = 2
    click_weight: float = 10.0
    negative_boost: float = 1.5
    kd_weight: float = 1.0
    sta_anchor: float = 0.05
    ltm_anchor: float = 0.5
    intensity_tau: float = 0.2
    min_component: int = 50


class OnlineAdaptor(LossFunctions):
    def __init__(self, model, settings=None):
        self.model = model
        self.settings = settings or OnlineSettings()
        self.device = next(model.parameters()).device
        self.encoder_params = list(model.feature_extractor.unet.encoder.parameters())
        self.decoder_params = (list(model.feature_extractor.unet.decoder.parameters())
                               + list(model.feature_extractor.head.parameters()))
        self.initial_state_dict = {name: p.detach().clone() for name, p in model.named_parameters()}
        self.sta_optimizer = torch.optim.Adam(self.decoder_params, lr=self.settings.sta_lr)
        self.ltm_optimizer = torch.optim.Adam(self.encoder_params, lr=self.settings.ltm_lr)

    def _set_mode(self, mode):
        active = self.decoder_params if mode == "mi" else self.encoder_params
        ids = {id(p) for p in active}
        for p in self.model.parameters():
            p.requires_grad_(id(p) in ids)
            p.grad = None
        self.model.eval()

    def clear_momentum(self):
        self.sta_optimizer.state.clear()
        self.ltm_optimizer.state.clear()

    def adapt_MI(self, image, points, clicks):
        if not clicks:
            return
        self._set_mode("mi")
        cfg = self.settings
        h, w = image.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(h, device=self.device),
                                torch.arange(w, device=self.device), indexing="ij")
        with torch.no_grad():
            reference = torch.sigmoid(self.model(image, points, training=False)["instances"])
            binary = (reference[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, count):
                if stats[i, cv2.CC_STAT_AREA] < cfg.min_component:
                    binary[labels == i] = 0
            target = torch.from_numpy(binary).float()[None, None].to(self.device)
            kd_mask = torch.ones_like(target)
            pos_r, neg_r, sigma = ((15, 10, 12.0) if len(clicks) <= 3 else
                                   (10, 7, 8.0) if len(clicks) <= 6 else (7, 5, 5.0))
            for y, x, positive in clicks:
                region = (yy-y).square() + (xx-x).square() <= (pos_r if positive else neg_r)**2
                target[0, 0, region] = float(positive)
                kd_mask[0, 0, region] = 0
        weight = cfg.click_weight * (cfg.negative_boost if any(not c[2] for c in clicks) else 1)
        with torch.enable_grad():
            for _ in range(cfg.sta_steps):
                self.sta_optimizer.zero_grad(set_to_none=True)
                logits = self.model(image, points, training=False)["instances"]
                click_loss = self._iaccg_loss(logits, target, [(y, x) for y, x, _ in clicks],
                                             sigma, image, cfg.intensity_tau)
                kd = F.binary_cross_entropy_with_logits(logits, reference, reduction="none")
                kd = (kd * kd_mask).sum() / (kd_mask.sum() + 1e-5)
                loss = weight * click_loss + cfg.kd_weight * kd + cfg.sta_anchor * self._anchor_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder_params, 1.0)
                self.sta_optimizer.step()

    def adapt_ltm(self, image, points, final_prediction):
        """No ground-truth argument: the target must be the final model prediction."""
        self._set_mode("pi_memory")
        target = final_prediction.detach().float().clone()
        empty = torch.full_like(points, -1.0)
        with torch.enable_grad():
            for _ in range(self.settings.ltm_steps):
                self.ltm_optimizer.zero_grad(set_to_none=True)
                logits = self.model(image, empty, training=False)["instances"]
                loss = self._dice_bce_loss(logits, target) + self.settings.ltm_anchor * self._anchor_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder_params, 1.0)
                self.ltm_optimizer.step()
