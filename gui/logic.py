import torch
import numpy as np
import cv2
from torchvision import transforms as T
from copy import deepcopy


def run_inference(model, original_img, sel_pix, device='cpu'):
    model.eval()

    positive_points_li = [(p[0][1], p[0][0], i) for i, p in enumerate(sel_pix) if p[1] == 1]
    negative_points_li = [(p[0][1], p[0][0], i) for i, p in enumerate(sel_pix) if p[1] == 0]
    points_nd = get_points_nd(positive_points_li, negative_points_li, device)

    input_img = deepcopy(original_img)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    input_img = np.expand_dims(input_img, axis=-1)
    input_img = T.functional.to_tensor(input_img).type(torch.float32).unsqueeze(0).to(device)

    # =====================================================================
    # ★ 核心通电：触发 OAIMS 交互中适应 (MI) ★
    # =====================================================================
    if hasattr(model, 'oaims_adaptor') and len(sel_pix) >= 2:
        print(f"⚡ 触发 MI 交互中自适应 (第 {len(sel_pix)} 次点击)...")

        # 提取最新一次点击的真实坐标和它的标签 (y, x, 1/0)
        latest_click = (int(sel_pix[-1][0][1]), int(sel_pix[-1][0][0]), int(sel_pix[-1][1]))

        # 连环重锤：使用带有新点击的 points_nd 和明确标签，强制模型更新 3 次
        loss_val = model.oaims_adaptor.adapt_MI(input_img, points_nd, latest_click)
        print(f"✅ MI 强制更新完成! CCG Loss: {loss_val:.4f}")
    # =====================================================================

    with torch.no_grad():
        outputs = model.forward(image=input_img, mask=None, points=points_nd, training=False)

        if isinstance(outputs, dict) and 'instances' in outputs:
            logits = outputs['instances']
        elif isinstance(outputs, dict) and 'samples' in outputs:
            logits = outputs['samples'][0] if isinstance(outputs['samples'], list) else outputs['samples']
        else:
            logits = outputs

        o_mask = torch.sigmoid(logits)

    return o_mask


def get_points_nd(pos_clicks, neg_clicks, device='cpu'):
    limit = 1000
    max_points = max(len(pos_clicks), len(neg_clicks))
    max_points = min(limit, max_points) if limit else max_points
    max_points = max(1, max_points)

    pos = pos_clicks + [(-1, -1, -1)] * (max_points - len(pos_clicks))
    neg = neg_clicks + [(-1, -1, -1)] * (max_points - len(neg_clicks))
    return torch.tensor([pos + neg], dtype=torch.float32, device=device)


def get_contour(mask):
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    gray = cv2.cvtColor(np.stack([mask] * 3, axis=2).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def get_mito_info(image, mask, contour):
    x, y, w, h = cv2.boundingRect(contour)
    area = np.ceil(cv2.contourArea(contour))
    crop = image[y:y + h, x:x + w]

    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mask.ndim == 2:
        maskgray = mask
    elif mask.ndim == 3:
        maskgray = mask[:, :, 0]
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    maskgray = cv2.cvtColor(np.stack([maskgray] * 3, axis=2).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    filled = np.zeros(imgray.shape, dtype=imgray.dtype)
    filled = cv2.drawContours(filled, [contour], -1, (255, 255, 255), -1)
    _, inv_mask = cv2.threshold(filled, 127, 255, cv2.THRESH_BINARY_INV)
    subtracted = cv2.subtract(imgray, cv2.bitwise_and(imgray, inv_mask))
    _, cristae = cv2.threshold(subtracted, 127, 255, cv2.THRESH_BINARY)

    unique = np.unique(cristae, return_counts=True)[1]
    back_area = 0 if len(unique) < 2 else unique[1]
    crista_ratio = 1 - (back_area / area)

    return area, w, h, crista_ratio, crop


def get_mito_info_lst(image, mask, contours):
    result = []
    for c in contours:
        area, w, h, cr, crop = get_mito_info(image, mask, c)
        result.append({
            'area': area,
            'width': w,
            'height': h,
            'crista_ratio': cr,
            'crop_object': crop
        })
    return result