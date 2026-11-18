# gui(alrealy)/callbacks.py

import os
import json
import base64
import cv2
import torch
import datetime
import numpy as np
from PIL import Image
from copy import deepcopy
import gradio as gr
from .logic import (
    run_inference,
    get_contour,
    get_mito_info_lst,
    get_points_nd
)
import torchvision.transforms as T
from pi_seg.inference.oaims_adaptor import OAIMSAdaptor
import os

# Visualization
point_colors = [(1, 0, 0), (0, 1, 0), (1, 1, 0)]
canvas_size = (640, 640)

_model = None
_device = None


def register_model(model, device):
    """Registers the segmentation model and device (CPU/GPU) for inference."""
    global _model, _device
    _model = model
    _device = device

    # ★ 挂载 OAIMS 引擎 ★
    if not hasattr(_model, 'oaims_adaptor'):
        _model.oaims_adaptor = OAIMSAdaptor(_model)
        print("🚀 OAIMS 引擎已成功挂载！准备实时进化！")


def select_file(evt: gr.SelectData, files):
    selected_image = np.asarray(Image.open(files[evt.index].name).convert('RGB'))

    preprocessed_image_ = selected_image.copy()
    input_image_new = gr.ImageEditor(value=preprocessed_image_, height=canvas_size[1], label='Input',
                                     type='numpy', show_label=False, interactive=False, transforms=[],
                                     eraser=False, brush=False)

    curr_file_name = os.path.splitext(os.path.basename(files[evt.index].name))[0]
    tmp_save_results = {'file_name': curr_file_name}
    save_results_str = json.dumps(tmp_save_results)

    # 返回参数严格匹配 file_explorer.select 的 outputs
    return (selected_image, preprocessed_image_, input_image_new, [], [], None, False,
            curr_file_name, 0, save_results_str)


def select_point(original_img, sel_pix, point_type, evt: gr.SelectData, processed_mask_li,
                 output_mask_inter, save_results, user_name='unknown', consumption_time=0):
    minute, second = divmod(consumption_time, 60)
    consumption_time_txt = f'{minute:02d}:{second:02d}'
    brush = gr.Brush(colors=['#000000'], default_color="#000000", color_mode="fixed", default_size=5)

    with torch.no_grad():
        img = original_img.copy()
        label = 0 if point_type == 'negative_point' else 1
        sel_pix.append((evt.index, label))

        # 仅接收单张 mask 返回
        o_mask = run_inference(_model, original_img, sel_pix, _device)
        o_mask = (o_mask.detach().cpu().numpy()[0, 0] > 0.5).astype(np.uint8) * 255
        processed_mask_li.append(o_mask)

        stack_mask = np.stack([o_mask] * 3, axis=2) * (0, 0, 1)
        img = (img / 255 * 0.8 + stack_mask / 255 * 0.2).astype(np.float32)

        for point, lbl in sel_pix:
            cv2.circle(img, point, radius=5, color=point_colors[lbl], thickness=-1)

        marker_img = np.clip(img * 255, 0, 255).astype(np.uint8)
        b64_mask = base64.b64encode(cv2.imencode('.png', o_mask)[1]).decode('utf-8')

        tmp_save_results = {} if save_results is None else json.loads(save_results)
        result_entry = {
            'consumption_time': consumption_time_txt,
            'num_points': len(sel_pix),
            'points_coord': sel_pix,
            'mask': b64_mask,
        }

        if not output_mask_inter:
            tmp_save_results['user'] = user_name
            tmp_save_results['results'] = [result_entry]
            save_results_str = json.dumps(tmp_save_results)

            output_mask_new = gr.ImageEditor(
                value=o_mask, height=canvas_size[1], label='Mask', type='numpy',
                interactive=True, transforms=[], eraser=False, brush=brush,
                show_label=False, show_download_button=True, elem_classes='output_mask_ie'
            )
            return marker_img, output_mask_new, o_mask, True, len(sel_pix), save_results_str

        tmp_save_results['results'].append(result_entry)
        save_results_str = json.dumps(tmp_save_results)

        return marker_img, o_mask, o_mask, True, len(sel_pix), save_results_str


def edit_mask(edit_img):
    return edit_img['composite']


def undo_points(original_img, sel_pix, processed_mask_li):
    brush = gr.Brush(colors=['#000000'], default_color="#000000", color_mode="fixed", default_size=5)

    if not isinstance(original_img, np.ndarray):
        return gr.Warning("Please upload an image.")

    if len(sel_pix) < 1 or len(processed_mask_li) < 1:
        return original_img, None, None, False, 0

    processed_mask_li.pop()

    if len(processed_mask_li):
        temp_mask = processed_mask_li[-1]
    else:
        sel_pix.clear()
        return original_img, None, None, False, 0

    temp_img = original_img.copy().astype(np.float32) / 255.0
    stack_mask = np.zeros_like(temp_img, dtype=np.float32)
    stack_mask[..., 2] = temp_mask.astype(np.float32) / 255.0
    temp_img = temp_img * 0.8 + stack_mask * 0.2

    sel_pix.pop()

    for point, label in sel_pix:
        cv2.circle(temp_img, point, radius=5, color=point_colors[label], thickness=-1)

    temp_img = np.clip(temp_img * 255, 0, 255).astype(np.uint8)

    output_mask_new = gr.ImageEditor(value=temp_mask, height=canvas_size[1], label='Mask',
                                     type='numpy', interactive=True, transforms=[], eraser=False,
                                     brush=brush, show_label=False, show_download_button=True)

    return temp_img, output_mask_new, temp_mask, True, len(sel_pix)


def reset_image(img):
    if isinstance(img, np.ndarray) is not True:
        gr.Warning("Please upload image.")
        # 这里为了防错，统一返回与 clear_image 一致的结构
        return [None] * 12

    preprocessed_image_ = img.copy()
    output_mask_new = gr.ImageEditor(value=None, height=canvas_size[1], label='Mask', type='numpy',
                                     interactive=False, show_label=False, show_download_button=False)

    return [
        img,  # original_image
        preprocessed_image_,  # preprocessed_image
        preprocessed_image_,  # input_image
        [],  # selected_points
        [],  # processed_mask_li
        output_mask_new,  # output_mask
        None,  # edited_mask
        None,  # crop_object_gallery
        None,  # num_object
        None,  # result_table
        False,  # output_mask_inter
        0,  # num_points
    ]


def clear_image():
    input_image = gr.ImageEditor(value=None, height=canvas_size[1], label='Input', type='numpy',
                                 interactive=False, show_label=False, transforms=[], eraser=False, brush=False)
    output_mask = gr.ImageEditor(value=None, height=canvas_size[1], label='Mask', type='numpy',
                                 interactive=False, show_label=False, show_download_button=False)

    return (None, None, input_image, [], [], output_mask, None, None, None, None, False, 0)


# 找到 complete_edit 函数
def complete_edit(original_image, edited_mask, sel_pix, curr_file_name, save_results, consumption_time=0):
    # ★ 触发 OAIMS 跨帧记忆沉淀 ★
    global _model, _device
    if hasattr(_model, 'oaims_adaptor') and len(sel_pix) > 0:
        print("🧠 正在执行跨帧记忆沉淀 (PI Adaptation)...")
        from .logic import get_points_nd
        import torchvision.transforms as T

        # 提取第一个点
        first_lbl = sel_pix[0][1]
        first_coord = (sel_pix[0][0][1], sel_pix[0][0][0], 0)
        points_1 = get_points_nd([first_coord], [], _device) if first_lbl == 1 else get_points_nd([], [first_coord],
                                                                                                  _device)

        # 转换图像
        input_img = deepcopy(original_image)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img = np.expand_dims(input_img, axis=-1)
        input_img = T.functional.to_tensor(input_img).type(torch.float32).unsqueeze(0).to(_device)

        # 更新权重
        _model.oaims_adaptor.adapt_PI(input_img, points_1, edited_mask)
        print("✅ 当前帧特征已成功写入模型权重！去测下一张图吧！")

    output_mask_new = gr.ImageEditor(value=edited_mask, height=canvas_size[1], type='numpy',
                                     interactive=False, show_label=False, show_download_button=True)
    return output_mask_new, False


def quantify_object(image, edited_mask):
    contours = get_contour(edited_mask)
    mito_info_lst = get_mito_info_lst(image, edited_mask, contours)

    results, crops = [], []
    for i, m in enumerate(mito_info_lst, start=1):
        if m['area'] > 10:
            results.append([i, m['width'], m['height'], m['area']])
            crops.append(m['crop_object'])

    return len(results), results, crops


def stop_timer():
    return True


def get_time_consumption(consumption_time, stop_flag):
    if stop_flag:
        consumption_time -= 1
        stop_flag = False
    consumption_time += 1
    minute, second = divmod(consumption_time, 60)
    consumption_time_txt = '{:02d}:{:02d}'.format(minute, second)
    return consumption_time_txt, consumption_time, stop_flag


def clear_time():
    consumption_time = 0
    minute, second = divmod(consumption_time, 60)
    consumption_time_txt = '{:02d}:{:02d}'.format(minute, second)
    return consumption_time_txt, 0, True

def trigger_reset_weights():
    """
    前端按钮调用的回调函数，用于重置 OAIMS 外挂的权重。
    """
    global _model
    if _model is not None and hasattr(_model, 'oaims_adaptor'):
        _model.oaims_adaptor.reset_weights()
        # 弹出一个前端提示框告诉用户重置成功
        gr.Info("Model weights successfully reset to original state!")
    else:
        gr.Warning("OAIMS Adaptor is not initialized.")


def save_finetuned_weights():
    """
    将当前内存中经过 OAIMS 微调的模型权重持久化保存到硬盘。
    """
    global _model
    if _model is not None:
        try:
            # 设定保存目录
            save_dir = "save"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "model_test.pth")

            # 保存模型的 state_dict
            torch.save({"state_dict": _model.state_dict()}, save_path)

            # 触发前端提示
            gr.Info(f"微调权重已成功保存至: {save_path}")
            print(f"💾 [OAIMS] 权重已保存至: {save_path}")
        except Exception as e:
            gr.Warning(f"保存权重失败: {str(e)}")
            print(f"❌ [OAIMS] 保存权重失败: {str(e)}")
    else:
        gr.Warning("模型未初始化，无法保存。")


