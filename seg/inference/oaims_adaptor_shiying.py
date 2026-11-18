import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label
import cv2
import gc

class OAIMSAdaptor:
    def __init__(self, model, lr=1e-4, beta=10.0, sigma=15.0):
        self.model = model
        self.beta = beta
        self.sigma = sigma
        self.device = next(model.parameters()).device

        self.encoder_params = []
        self.decoder_params = []

        for name, param in self.model.named_parameters():
            if 'decoder' in name.lower() or 'head' in name.lower():
                self.decoder_params.append(param)
            else:
                self.encoder_params.append(param)

        self.optimizer = torch.optim.Adam([
            {'params': self.encoder_params, 'lr': 1e-5},
            {'params': self.decoder_params, 'lr': lr}
        ])

        self.initial_state_dict = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters() if param.requires_grad
        }
        self.is_memory_activated = False

    def reset_weights(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.initial_state_dict:
                    param.copy_(self.initial_state_dict[name])
        self.optimizer.state.clear()
        self.is_memory_activated = False
        print("🔄 [OAIMS] 模型已重置为纯净基座。")

    def clear_momentum(self):
        self.optimizer.state.clear()

    def _set_mode(self, mode):
        if mode == 'mi':
            for p in self.encoder_params:
                p.requires_grad = False
                p.grad = None
            for p in self.decoder_params:
                p.requires_grad = True
        elif mode == 'pi_memory':
            for p in self.encoder_params:
                p.requires_grad = True
            for p in self.decoder_params:
                p.requires_grad = False
                p.grad = None

    def _anchor_loss(self):
        reg_loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.initial_state_dict:
                reg_loss += torch.sum((param - self.initial_state_dict[name]) ** 2)
        return reg_loss

    def _dice_bce_loss(self, pred_logits, target_mask):
        bce = F.binary_cross_entropy_with_logits(pred_logits, target_mask)
        pred_probs = torch.sigmoid(pred_logits)
        intersection = (pred_probs * target_mask).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + target_mask.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
        return bce + dice.mean()

    def _iaccg_loss(self, pred_logits, target_mask, clicks, sigma_spatial, image, intensity_threshold=0.2):
        B, C, H, W = pred_logits.shape
        g_map = torch.zeros((B, 1, H, W), device=self.device)
        y_arr = torch.arange(H, device=self.device).view(-1, 1)
        x_arr = torch.arange(W, device=self.device).view(1, -1)
        img_tensor = image[0, 0]

        for (cy, cx) in clicks:
            dist_sq = (y_arr - cy) ** 2 + (x_arr - cx) ** 2
            spatial_g = torch.exp(-dist_sq / (2 * sigma_spatial ** 2))
            y_min, y_max = max(0, cy - 1), min(H, cy + 2)
            x_min, x_max = max(0, cx - 1), min(W, cx + 2)
            click_intensity = torch.median(img_tensor[y_min:y_max, x_min:x_max])
            intensity_diff = torch.abs(img_tensor - click_intensity)
            intensity_weight = torch.exp(- (intensity_diff / intensity_threshold) ** 2)
            intensity_weight = torch.clamp(intensity_weight, min=0.1, max=1.0)
            g_map[0, 0] += spatial_g * intensity_weight

        g_map = torch.clamp(g_map, 0, 1)
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_mask, reduction='none')
        return (bce_loss * g_map).sum() / (g_map.sum() + 1e-5)

    def adapt_MI(self, image, points_t, all_clicks):
        self._set_mode('mi')
        with torch.enable_grad():
            self.model.eval()
            B, C, H, W = image.shape
            gy, gx = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')

            has_negative_click = any(click[2] == 0 for click in all_clicks)
            self.optimizer.param_groups[1]['lr'] = 1e-4
            current_beta = self.beta * 1.5 if has_negative_click else self.beta

            with torch.no_grad():
                out_initial = self.model(image, points_t, training=False)
                logits_base = out_initial['instances'].detach()
                prob_base = torch.sigmoid(logits_base)
                pseudo_gt_np = (prob_base > 0.5).cpu().numpy()[0, 0].astype(np.uint8)

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pseudo_gt_np, connectivity=8)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] < 50:
                        pseudo_gt_np[labels == i] = 0

                pseudo_gt = torch.tensor(pseudo_gt_np, device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                kd_mask = torch.ones_like(logits_base)

                click_count = len(all_clicks)
                if click_count <= 3:
                    pos_r, neg_r, sig = 15, 10, 12.0
                elif click_count <= 6:
                    pos_r, neg_r, sig = 10, 7, 8.0
                else:
                    pos_r, neg_r, sig = 7, 5, 5.0

                for click in all_clicks:
                    y, x, is_pos = click[0], click[1], click[2]
                    override_mask = ((gy - y) ** 2 + (gx - x) ** 2) <= (neg_r ** 2 if is_pos == 0 else pos_r ** 2)
                    pseudo_gt[0, 0, override_mask] = 0.0 if is_pos == 0 else 1.0
                    kd_mask[0, 0, override_mask] = 0.0

            for _ in range(2):
                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(image, points_t, training=False)
                logits_current = out['instances']

                loss_ccg = 0
                if len(all_clicks) > 0:
                    clicks_coords = [(click[0], click[1]) for click in all_clicks]
                    loss_ccg = self._iaccg_loss(logits_current, pseudo_gt, clicks_coords, sig, image, intensity_threshold=0.2)

                bce_loss_kd = F.binary_cross_entropy_with_logits(logits_current, prob_base, reduction='none')
                loss_kd = (bce_loss_kd * kd_mask).sum() / (kd_mask.sum() + 1e-5)
                loss_ewc = self._anchor_loss()

                lambda_kd = 1.0
                lambda_ewc = 0.05
                loss_mi = current_beta * loss_ccg + lambda_kd * loss_kd + lambda_ewc * loss_ewc
                loss_mi.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder_params, max_norm=1.0)
                self.optimizer.step()

        return loss_mi.item()

    def adapt_PI(self, image, points_1, final_mask, max_clicks=10):
        self.is_memory_activated = True
        if isinstance(final_mask, np.ndarray): final_mask = final_mask[:, :, 0] if final_mask.ndim == 3 else final_mask
        final_mask_tensor = torch.tensor(final_mask, device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        final_mask_tensor = (final_mask_tensor > 0.5).float()

        with torch.enable_grad():
            self.model.eval()
            self._set_mode('mi')

            self.optimizer.param_groups[1]['lr'] = 1e-5
            self.optimizer.zero_grad(set_to_none=True)
            out_1 = self.model(image, points_1, training=False)
            loss_s1 = self._dice_bce_loss(out_1['instances'], final_mask_tensor)
            loss_s1.backward()
            torch.nn.utils.clip_grad_norm_(self.decoder_params, max_norm=1.0)
            self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                pred_1_probs = torch.sigmoid(self.model(image, points_1, training=False)['instances'])
                pred_1_mask = (pred_1_probs > 0.5).cpu().numpy()[0, 0]

            generated_clicks = self._generate_error_clicks(pred_1_mask, final_mask_tensor.cpu().numpy()[0, 0], max_clicks)
            if len(generated_clicks) > 0:
                points_hat = self._append_clicks(points_1, generated_clicks)
                out_hat = self.model(image, points_hat, training=False)
                loss_s2_df = self._dice_bce_loss(out_hat['instances'], final_mask_tensor)

                loss_s2_ccg = self._iaccg_loss(
                    out_hat['instances'], final_mask_tensor, [(y, x) for y, x, _ in generated_clicks],
                    self.sigma, image, intensity_threshold=0.2
                )
                loss_s2 = loss_s2_df + self.beta * loss_s2_ccg
                loss_s2.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder_params, max_norm=1.0)
                self.optimizer.step()

            self._set_mode('pi_memory')
            self.optimizer.param_groups[0]['lr'] = 1e-5

            empty_points = torch.full_like(points_1, -1.0)
            for _ in range(2):
                self.optimizer.zero_grad(set_to_none=True)
                out_empty = self.model(image, empty_points, training=False)
                loss_empty = self._dice_bce_loss(out_empty['instances'], final_mask_tensor)
                loss_anchor = 0.5 * self._anchor_loss()
                loss_total = loss_empty + loss_anchor
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder_params, max_norm=1.0)
                self.optimizer.step()

        gc.collect()
        torch.cuda.empty_cache()
        return loss_empty.item() if 'loss_empty' in locals() else loss_s1.item()

    def _generate_error_clicks(self, pred_mask, gt_mask, max_clicks):
        error_mask = (pred_mask != gt_mask).astype(np.uint8)
        labeled_array, num_features = label(error_mask)
        clicks = []
        if num_features == 0: return clicks
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0
        sorted_components = np.argsort(component_sizes)[::-1]
        for i in range(min(max_clicks, num_features)):
            comp_id = sorted_components[i]
            if component_sizes[comp_id] < 10: continue
            comp_mask = (labeled_array == comp_id).astype(np.uint8)
            dist_transform = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            max_idx = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
            clicks.append((max_idx[0], max_idx[1], gt_mask[max_idx[0], max_idx[1]]))
        return clicks

    def _append_clicks(self, points_tensor, new_clicks):
        new_points = points_tensor.clone()
        num_points = new_points.size(1) // 2
        for (y, x, is_pos) in new_clicks:
            offset = 0 if is_pos else num_points
            for i in range(offset, offset + num_points):
                if new_points[0, i, 0] == -1:
                    new_points[0, i, 0] = y
                    new_points[0, i, 1] = x
                    new_points[0, i, 2] = 1.0
                    break
        return new_points