import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pi_seg.inference.oaims_adaptor_KD_EWC_IAIoU import OAIMSAdaptor as OriginalOAIMSAdaptor


class OAIMSAdaptor(OriginalOAIMSAdaptor):
    """Ablation-oriented adaptor.

    The original adaptor is kept unchanged for reproducing the main OAIS-Net
    results. This subclass makes the STA-only branch conservative enough for
    module ablation: image-internal adaptation updates only lightweight output
    parameters and uses smaller local pseudo-label regions.
    """

    def __init__(self, model, lr=1e-4, beta=10.0, sigma=15.0):
        super().__init__(model, lr=lr, beta=beta, sigma=sigma)
        self.ablation_sta_only_mode = False
        self.sta_params = []
        self.sta_param_names = []
        self.body_decoder_params = []

        for name, param in self.model.named_parameters():
            lower_name = name.lower()
            if (
                "maps_transform" in lower_name
                or "feature_extractor.head" in lower_name
                or "feature_extractor.unet.decoder.blocks.3" in lower_name
                or "feature_extractor.unet.decoder.blocks.2" in lower_name
                or "head" in lower_name
                or "output" in lower_name
                or "final" in lower_name
            ):
                self.sta_params.append(param)
                self.sta_param_names.append(name)
            elif "decoder" in lower_name:
                self.body_decoder_params.append(param)

        if not self.sta_params:
            self.sta_params = list(self.decoder_params)
            self.sta_param_names = ["decoder_params_fallback"]

    def _set_mode(self, mode):
        if mode == "mi":
            sta_ids = {id(p) for p in self.sta_params}
            for p in self.encoder_params:
                p.requires_grad = id(p) in sta_ids
                if not p.requires_grad:
                    p.grad = None
            for p in self.decoder_params:
                p.requires_grad = id(p) in sta_ids
                if not p.requires_grad:
                    p.grad = None
        else:
            super()._set_mode(mode)

    def _anchor_loss_trainable(self):
        reg_loss = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.initial_state_dict:
                reg_loss = reg_loss + torch.sum((param - self.initial_state_dict[name]) ** 2)
        return reg_loss

    def _click_params(self, click_count):
        if click_count <= 3:
            return 10, 8, 7.0
        if click_count <= 6:
            return 8, 6, 5.5
        return 6, 4, 4.0

    def _local_click_bce(self, logits, all_clicks, h, w):
        if not all_clicks:
            return torch.tensor(0.0, device=self.device)

        yy, xx = torch.meshgrid(
            torch.arange(h, device=self.device),
            torch.arange(w, device=self.device),
            indexing="ij",
        )
        target = torch.sigmoid(logits.detach()).clone()
        weight = torch.zeros_like(logits)

        recent_clicks = all_clicks[-5:] if len(all_clicks) > 5 else all_clicks
        for y, x, is_pos in recent_clicks:
            radius = 8 if is_pos == 1 else 6
            local_mask = ((yy - y) ** 2 + (xx - x) ** 2) <= radius ** 2
            target[0, 0, local_mask] = 1.0 if is_pos == 1 else 0.0
            weight[0, 0, local_mask] = 1.0

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (bce * weight).sum() / (weight.sum() + 1e-5)

    def adapt_MI(self, image, points_t, all_clicks):
        self._set_mode("mi")

        with torch.enable_grad():
            self.model.eval()
            b, c, h, w = image.shape
            gy, gx = torch.meshgrid(
                torch.arange(h, device=self.device),
                torch.arange(w, device=self.device),
                indexing="ij",
            )

            with torch.no_grad():
                out_initial = self.model(image, points_t, training=False)
                logits_base = out_initial["instances"].detach()
                prob_base = torch.sigmoid(logits_base)
                pseudo_gt_np = (prob_base > 0.5).cpu().numpy()[0, 0].astype(np.uint8)

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pseudo_gt_np, connectivity=8)
                for label_idx in range(1, num_labels):
                    if stats[label_idx, cv2.CC_STAT_AREA] < 50:
                        pseudo_gt_np[labels == label_idx] = 0

                pseudo_gt = torch.tensor(
                    pseudo_gt_np,
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(0).unsqueeze(0)
                kd_mask = torch.ones_like(logits_base)

                pos_r, neg_r, sigma = self._click_params(len(all_clicks))
                for y, x, is_pos in all_clicks:
                    radius = pos_r if is_pos == 1 else neg_r
                    override = ((gy - y) ** 2 + (gx - x) ** 2) <= radius ** 2
                    pseudo_gt[0, 0, override] = 1.0 if is_pos == 1 else 0.0
                    kd_mask[0, 0, override] = 0.0

            trainable_params = [p for p in self.sta_params if p.requires_grad]
            sta_optimizer = torch.optim.Adam(trainable_params, lr=1e-4 if self.ablation_sta_only_mode else 8e-5)

            recent_clicks = all_clicks[-5:] if len(all_clicks) > 5 else all_clicks
            loss_mi = torch.tensor(0.0, device=self.device)

            for _ in range(2):
                sta_optimizer.zero_grad(set_to_none=True)
                output = self.model(image, points_t, training=False)
                logits_current = output["instances"]

                if recent_clicks:
                    click_coords = [(click[0], click[1]) for click in recent_clicks]
                    loss_click = self._iaccg_loss(
                        logits_current,
                        pseudo_gt,
                        click_coords,
                        sigma,
                        image,
                        intensity_threshold=0.15,
                    )
                else:
                    loss_click = torch.tensor(0.0, device=self.device)

                kd = F.binary_cross_entropy_with_logits(logits_current, prob_base, reduction="none")
                loss_kd = (kd * kd_mask).sum() / (kd_mask.sum() + 1e-5)
                loss_anchor = self._anchor_loss_trainable()
                loss_local = self._local_click_bce(logits_current, all_clicks, h, w)

                if self.ablation_sta_only_mode:
                    loss_mi = 10.0 * loss_click + 1.5 * loss_local + 0.25 * loss_kd + 0.03 * loss_anchor
                else:
                    loss_mi = 10.0 * loss_click + 0.4 * loss_kd + 0.05 * loss_anchor
                loss_mi.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                sta_optimizer.step()

        return float(loss_mi.detach().cpu())
