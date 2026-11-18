import os
import pickle

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from time import time
from pathlib import Path
from datetime import timedelta

from pi_seg.inference.clicker.clicker_deepPI import Clicker as PI_Clicker
from pi_seg.inference.clicker.clicker import Clicker
from pi_seg.utils.exp_imports.default import *
from pi_seg.utils.vis import draw_probmap, draw_with_blend_and_clicks, add_tag


def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array([_get_noc(iou_arr, iou_thr)
                               for iou_arr in all_ious], dtype=np.int)

        score = scores_arr.mean()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        over_max_list.append(over_max)

    return noc_list, over_max_list


def get_prediction_vis_callback(logs_path, dataset_name, prob_thresh):
    save_path = logs_path / 'predictions_vis' / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    def callback(image, gt_mask, pred_probs, sample_id, click_indx, clicks_list):
        sample_path = save_path / f'{sample_id}_{click_indx}.jpg'
        prob_map = draw_probmap(pred_probs)
        image_with_mask = draw_with_blend_and_clicks(image, pred_probs > prob_thresh, clicks_list=clicks_list)
        cv2.imwrite(str(sample_path), np.concatenate((image_with_mask, prob_map), axis=1)[:, :, ::-1])

    return callback


def check_divide_zero(value):
    if value > 1:
        return -1

    return value


def fn_performance(label_, output):
    FP = len(np.where(output - label_ == 1)[0])
    FN = len(np.where(output - label_ == -1)[0])
    TP = len(np.where(output + label_ == 2)[0])
    TN = len(np.where(output + label_ == 0)[0])

    accuracy = (TP + TN) / (TP + FN + FP + TN + 1e-6)
    DSC = 2 * TP / (2 * TP + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    jarcard = TP / (TP + FP + FN + 1e-6)
    jarcard_b = TN / (TN + FP + FN + 1e-6)
    iou_o = (jarcard + jarcard_b) / 2

    accuracy = check_divide_zero(accuracy)
    DSC = check_divide_zero(DSC)
    precision = check_divide_zero(precision)
    recall = check_divide_zero(recall)
    jarcard = check_divide_zero(jarcard)
    jarcard_b = check_divide_zero(jarcard_b)
    iou_o = check_divide_zero(iou_o)

    result = {'accuracy': accuracy, 'DSC': DSC, 'precision': precision, 'recall': recall, 'iou_f': jarcard,
              'iou_b': jarcard_b, 'iou_o': iou_o}

    return result


def evaluate(predictor, sample, sample_cnt=16, index=0, min_clicks=1, max_clicks=30, pred_thr=0.5, max_iou_thr=1.0,
             vis=False, save_dir='./experiments', uncertainty=False):
    image, gt_mask, init_mask = sample.image, sample.gt_mask, sample.init_mask

    clicker = PI_Clicker(gt_mask=gt_mask)

    if uncertainty is False:
        clicker = Clicker(gt_mask=gt_mask)

    pred_mask = torch.zeros_like(torch.tensor(gt_mask)).cuda()
    pred_mask = pred_mask.unsqueeze(0)
    pred_mask = pred_mask.unsqueeze(0)
    prev_mask = pred_mask

    uncertainty_map = torch.zeros_like(torch.tensor(gt_mask)).cuda()
    prev_uncertainty_map = uncertainty_map
    out_image = None

    ious_list = []
    ious_b_list = []
    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            vis_pred = prev_mask

            if uncertainty:
                clicker.make_next_click(pred_mask, uncertainty_map=prev_uncertainty_map, top_uncertainty=True)
            else:
                clicker.make_next_click(pred_mask)

            pred_probs, uncertainty_map, ps = predictor.get_prediction(clicker, sample_cnt=sample_cnt,
                                                                       prev_mask=prev_mask)
            pred_mask = pred_probs > pred_thr

            prev_uncertainty_map = torch.tensor(uncertainty_map).cuda()

            iou = fn_performance(gt_mask, pred_mask)['iou_f']
            ious_list.append(iou)

            iou_b = fn_performance(gt_mask, pred_mask)['iou_b']
            ious_b_list.append(iou_b)

            prev_mask = torch.tensor(pred_mask).cuda()
            prev_mask = prev_mask.unsqueeze(0)
            prev_mask = prev_mask.unsqueeze(0)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

            if vis:
                clicks_list = clicker.get_clicks()
                last_y, last_x = predictor.last_y, predictor.last_x
                out_image = vis_result_base(save_dir, index, image, pred_mask, gt_mask, init_mask, iou,
                                            click_indx + 1, clicks_list, vis_pred, last_y, last_x, uncertainty_map)

    if vis:
        return np.array(ious_list, dtype=np.float32), np.array(ious_b_list,
                                                               dtype=np.float32), pred_probs, pred_mask, uncertainty_map, out_image, clicker, ps

    return np.array(ious_list, dtype=np.float32), np.array(ious_b_list,
                                                           dtype=np.float32), pred_probs, pred_mask, uncertainty_map, clicker, ps


def get_contour(mask, thresh=20):
    tmp_mask = np.array(mask, dtype=np.uint8)
    contour, hierarchy = cv2.findContours(tmp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = [x for x in contour if len(x) > thresh]

    return contour


def get_mito_info(contour):
    x0, y0 = zip(*np.squeeze(contour))
    x, y, w, h = cv2.boundingRect(contour)
    area = np.ceil(cv2.contourArea(contour))

    return area, w, h


def get_mito_info_lst(contours):
    mito_info_lst = []
    for contour in contours:
        tmp_info = {}
        area, w, h = get_mito_info(contour)

        tmp_info['Area'] = area
        tmp_info['witdh'] = w
        tmp_info['height'] = h

        mito_info_lst.append(tmp_info)

    return mito_info_lst


fn_tonumpy = lambda x: x.to('cpu').detach().numpy()


# 将此函数添加到 pi_seg/utils/val_utils.py 文件的最底部

def evaluate_with_oaims(predictor, sample, index=0, max_clicks=20, max_iou_thr=0.9,
                        device='cuda', vis=False, save_dir=None, **kwargs):
    """
    【对齐 GUI 逻辑版】测试脚本
    完全模拟 logic.py 中的坐标转换和图像处理，确保模型在测试时看到的输入与 GUI 一致。
    """
    image_raw, gt_mask_raw = sample.image, sample.gt_mask

    # 1. 标签预处理：治愈机器人眼盲，确保 Clicker 找得到目标
    gt_mask_np = (np.array(gt_mask_raw) > 0).astype(np.int32)
    clicker = Clicker(gt_mask=gt_mask_np)

    # 2. 图像预处理：模拟 logic.py，但增加通道数判断
    # 检查维度：如果是 (H, W, 1) 或者 (H, W)，scn 就是 1
    if len(image_raw.shape) == 3 and image_raw.shape[2] == 3:
        # 只有确实是 3 通道图像时才进行 BGR2GRAY
        input_img_gray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
    elif len(image_raw.shape) == 3 and image_raw.shape[2] == 1:
        # 如果是 (H, W, 1)，去掉最后一个轴
        input_img_gray = np.squeeze(image_raw, axis=2)
    else:
        # 本身就是 (H, W) 的灰度图
        input_img_gray = image_raw

    # 转换为 Tensor，保持 1, 1, H, W 结构
    input_img_pt = torch.from_numpy(input_img_gray).float().unsqueeze(0).unsqueeze(0).to(device)

    if input_img_pt.max() > 1.0:
        input_img_pt /= 255.0

    # 设置模型输入
    predictor.set_input_image(image_raw)

    ious_list = []
    pred_mask_torch = torch.zeros_like(input_img_pt)

    for click_indx in range(max_clicks):
        # 机器人模拟点击：根据当前预测和 GT 的差异产生新点
        clicker.make_next_click(pred_mask_torch)

        # ---------------------------------------------------------
        # 3. 坐标转换：【关键】严格模拟 logic.py 的 get_points_nd 逻辑
        # ---------------------------------------------------------
        clicks = clicker.get_clicks()
        # logic.py 习惯：(y, x, index) 格式，这里我们构造出 points_nd
        pos_clicks = [(c.coords[0], c.coords[1], i) for i, c in enumerate(clicks) if c.is_positive]
        neg_clicks = [(c.coords[0], c.coords[1], i) for i, c in enumerate(clicks) if not c.is_positive]

        # 调用你在 logic.py 里的同名工具函数（假设已经在 val_utils 可视范围内，或直接写逻辑）
        # 这里直接手动构造对齐 logic.py 的维度 [1, N, 3]
        max_len = len(clicks)
        points_nd = np.full((1, max_len, 3), -1, dtype=np.float32)
        for i, c in enumerate(clicks):
            points_nd[0, i, 0] = c.coords[1]  # x
            points_nd[0, i, 1] = c.coords[0]  # y
            points_nd[0, i, 2] = 1 if c.is_positive else 0
        points_nd = torch.from_numpy(points_nd).to(device)

        # ---------------------------------------------------------
        # 4. 触发 OAIMS 帧内更新 (Mid-Interaction)
        # ---------------------------------------------------------
        if click_indx > 0:
            all_clicks_tuple = [(c.coords[0], c.coords[1], int(c.is_positive)) for c in clicks]
            predictor.oaims_adaptor.adapt_MI(input_img_pt, points_nd, all_clicks_tuple)

        # 获取预测结果
        with torch.no_grad():
            pred_probs = predictor.get_prediction(clicker)

            # --- 🛡️ 安全处理：兼容 Tensor 和 NumPy ---
        if torch.is_tensor(pred_probs):
            # 如果是张量，执行剥离计算图的操作
            pred_probs_np = pred_probs.detach().cpu().numpy().squeeze()
        else:
            # 如果已经是 NumPy 数组，直接 squeeze
            pred_probs_np = np.squeeze(pred_probs)

        # 二值化生成掩码
        pred_mask_np = (pred_probs_np > 0.5).astype(np.int32)
        iou = fn_performance(gt_mask_np, pred_mask_np)['iou_f']
        ious_list.append(iou)

        # 更新用于下一次点击对比的 mask
        pred_mask_torch = torch.from_numpy(pred_mask_np).float().unsqueeze(0).unsqueeze(0).to(device)

        print(f"\r  [Image {index}] Click {click_indx + 1}/{max_clicks} | IoU: {iou:.4f}", end="", flush=True)

        if iou >= max_iou_thr:
            print()
            break

    # ---------------------------------------------------------
    # 6. 触发跨帧记忆 (Post-Interaction)：解决你说的“第二张唤醒”问题
    # ---------------------------------------------------------
    if len(clicks) > 0:
        # 提取第一下点击，构造 [1, 1, 3] 的坐标张量
        p1 = np.full((1, 1, 3), -1, dtype=np.float32)
        p1[0, 0, 0], p1[0, 0, 1], p1[0, 0, 2] = clicks[0].coords[1], clicks[0].coords[0], 1
        points_1 = torch.from_numpy(p1).to(device)

        predictor.oaims_adaptor.adapt_PI(input_img_pt, points_1, pred_mask_np)

    return np.array(ious_list), None, pred_probs, pred_mask_torch, clicker