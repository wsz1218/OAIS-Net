import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label
import cv2


class OAIMSAdaptor:
    # 恢复稍大一点的学习率 1e-4，因为我们只更新单次
    def __init__(self, model, lr=1e-4, beta=10.0, sigma=15.0):
        self.model = model
        self.beta = beta
        self.sigma = sigma
        self.device = next(model.parameters()).device

        # 1. 初始绝对防御：全部冻结
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. 黄金分割点：精准解冻 Decoder 和 Head，死保 Encoder
        trainable_params = []
        if hasattr(self.model, 'feature_extractor') and hasattr(self.model.feature_extractor, 'unet'):
            # 解冻解码器 (Decoder)：让它学会如何重新组合特征
            if hasattr(self.model.feature_extractor.unet, 'decoder'):
                for param in self.model.feature_extractor.unet.decoder.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)

        if hasattr(self.model, 'feature_extractor') and hasattr(self.model.feature_extractor, 'head'):
            # 解冻分类头 (Head)
            for param in self.model.feature_extractor.head.parameters():
                param.requires_grad = True
                trainable_params.append(param)

        self.optimizer = torch.optim.Adam(trainable_params, lr=lr)

        self.initial_state_dict = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters() if param.requires_grad
        }

    def reset_weights(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.initial_state_dict:
                    param.copy_(self.initial_state_dict[name])
        self.optimizer.state.clear()
        print("🔄 [OAIMS] 模型权重和优化器状态已成功重置为基座初始状态！")

    def _dice_bce_loss(self, pred_logits, target_mask):
        bce = F.binary_cross_entropy_with_logits(pred_logits, target_mask)
        pred_probs = torch.sigmoid(pred_logits)
        intersection = (pred_probs * target_mask).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + target_mask.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
        return bce + dice.mean()

    def _get_gaussian_map(self, shape, clicks, sigma):
        B, C, H, W = shape
        g_map = torch.zeros((B, C, H, W), device=self.device)
        y_arr = torch.arange(H, device=self.device).view(-1, 1)
        x_arr = torch.arange(W, device=self.device).view(1, -1)
        for (cy, cx) in clicks:
            g_map += torch.exp(-((y_arr - cy) ** 2 + (x_arr - cx) ** 2) / (2 * sigma ** 2))
        return torch.clamp(g_map, 0, 1)

    def _ccg_loss(self, pred_logits, target_mask, clicks, sigma):
        g_map = self._get_gaussian_map(pred_logits.shape, clicks, sigma)
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_mask, reduction='none')
        return (bce_loss * g_map).mean()

    def adapt_MI(self, image, points_t, latest_click):
        y, x, is_pos = latest_click

        with torch.enable_grad():
            self.model.eval()  # 死保 BN 层统计量

            with torch.no_grad():
                out_initial = self.model(image, points_t, training=False)
                pseudo_gt = (torch.sigmoid(out_initial['instances']) > 0.5).float()

            B, C, H, W = pseudo_gt.shape
            gy, gx = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device),
                                    indexing='ij')

            # 半径设为 7：恰到好处的指引，既不误伤友军，也不至于微弱
            override_mask = ((gy - y) ** 2 + (gx - x) ** 2) <= 7 ** 2

            if is_pos == 1:
                pseudo_gt[0, 0, override_mask] = 1.0
            else:
                pseudo_gt[0, 0, override_mask] = 0.0

                # ★ 恢复极速响应：只执行 1 次梯度下降！★
            self.optimizer.zero_grad()
            out = self.model(image, points_t, training=False)
            logits = out['instances']

            loss_df = self._dice_bce_loss(logits, pseudo_gt)
            loss_ccg = self._ccg_loss(logits, pseudo_gt, [(y, x)], self.sigma)

            loss_mi = loss_df + self.beta * loss_ccg
            loss_mi.backward()
            self.optimizer.step()

        return loss_mi.item()

    def adapt_PI(self, image, points_1, final_mask, max_clicks=10):
        with torch.enable_grad():
            self.model.eval()

            # 防爆清洗：严格将 GUI 传来的像素转为 0.0 和 1.0
            if isinstance(final_mask, np.ndarray):
                if final_mask.ndim == 3:
                    final_mask = final_mask[:, :, 0]

            final_mask_tensor = torch.tensor(final_mask, device=self.device, dtype=torch.float32).unsqueeze(
                0).unsqueeze(0)
            final_mask_tensor = (final_mask_tensor > 0.5).float()

            self.optimizer.zero_grad()
            out_1 = self.model(image, points_1, training=False)
            loss_s1 = self._dice_bce_loss(out_1['instances'], final_mask_tensor)
            loss_s1.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            with torch.no_grad():
                pred_1_probs = torch.sigmoid(self.model(image, points_1, training=False)['instances'])
                pred_1_mask = (pred_1_probs > 0.5).cpu().numpy()[0, 0]

            generated_clicks = self._generate_error_clicks(pred_1_mask, final_mask_tensor.cpu().numpy()[0, 0],
                                                           max_clicks)

            if len(generated_clicks) == 0:
                return loss_s1.item()

            points_hat = self._append_clicks(points_1, generated_clicks)

            out_hat = self.model(image, points_hat, training=False)
            loss_s2_df = self._dice_bce_loss(out_hat['instances'], final_mask_tensor)
            loss_s2_ccg = self._ccg_loss(out_hat['instances'], final_mask_tensor,
                                         [(y, x) for y, x, _ in generated_clicks], self.sigma)

            loss_s2 = loss_s2_df + self.beta * loss_s2_ccg
            loss_s2.backward()
            self.optimizer.step()

        return loss_s2.item()

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