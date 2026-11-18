import os
import numpy as np
import torch
import random
import torch.nn.functional as F
from tqdm import tqdm

from pi_seg.utils.train_utils import *
from config import (
    device, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, VALIDATION_STEP,
    NUM_MAX_POINTS, SAVE_DIR
)


def train_and_validate(model, train_loader, val_loader, optimizer,
                       start_epoch=1, best_iou=0.0, best_epoch=0,
                       train_loss_history=None, val_metrics_history=None):
    os.makedirs(SAVE_DIR, exist_ok=True)

    if train_loss_history is None: train_loss_history = []
    if val_metrics_history is None: val_metrics_history = []

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_loss = []

        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data['points']

            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
            num_points = random.randint(0, NUM_MAX_POINTS)

            # 1. 模拟交互阶段 (无梯度)
            with torch.no_grad():
                model.eval()
                for click_idx in range(num_points):
                    # 判断是否有 with_prev_mask 属性
                    use_prev = hasattr(model.module if hasattr(model, 'module') else model, 'with_prev_mask') and \
                               (model.module.with_prev_mask if hasattr(model, 'module') else model.with_prev_mask)
                    net_input = torch.cat((image, prev_output), dim=1) if use_prev else image

                    # 极速前向传播
                    outputs = model(image=net_input, mask=gt_mask, points=points, training=False)
                    logits = outputs['instances']
                    prev_output = torch.sigmoid(logits)

                    # 纯确定性落点
                    points = get_next_points_removeall(prev_output, gt_mask, points, click_idx + 1)

            # 2. 实际训练更新阶段 (有梯度)
            model.train()
            use_prev = hasattr(model.module if hasattr(model, 'module') else model, 'with_prev_mask') and \
                       (model.module.with_prev_mask if hasattr(model, 'module') else model.with_prev_mask)
            net_input = torch.cat((image, prev_output), dim=1) if use_prev else image

            outputs = model(image=net_input, mask=gt_mask, points=points, training=True)
            logits = outputs['instances']

            # ★ 核心：纯净的二值交叉熵损失，代替臃肿的 ELBO ★
            loss = F.binary_cross_entropy_with_logits(logits, gt_mask.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        avg_train_loss = np.mean(train_loss)
        train_loss_history.append(avg_train_loss)

        if epoch % VALIDATION_STEP == 0:
            val_metrics = validate_model(model, val_loader)
            val_metrics['epoch'] = epoch
            val_metrics_history.append(val_metrics)

            val_iou = val_metrics["iou"]

            if val_iou > best_iou:
                best_iou = val_iou
                best_epoch = epoch
                best_model_path = os.path.join(SAVE_DIR, "model_best_iou.pth")
                torch.save({
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": val_metrics,
                }, best_model_path)
                print(f"Saved best IoU model at {best_model_path} (IoU: {best_iou:.4f})")

        latest_model_path = os.path.join(SAVE_DIR, "model_latest.pth")
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss_history": train_loss_history,
            "val_metrics_history": val_metrics_history
        }, latest_model_path)

        print(
            f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Best Model: Epoch {best_epoch} (IoU: {best_iou:.4f})")

    print(f"Training finished. Latest model saved at {latest_model_path}")


def validate_model(model, val_loader):
    accuracy_lst, iou_lst, recall_lst, precision_lst, dsc_lst = [], [], [], [], []

    with torch.no_grad():
        model.eval()
        for batch_data in tqdm(val_loader, desc="Validation"):
            batch_data = {k: v.to(device) for k, v in batch_data.items()}
            image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data['points']

            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
            for click_idx in range(NUM_MAX_POINTS):
                use_prev = hasattr(model.module if hasattr(model, 'module') else model, 'with_prev_mask') and \
                           (model.module.with_prev_mask if hasattr(model, 'module') else model.with_prev_mask)
                net_input = torch.cat((image, prev_output), dim=1) if use_prev else image

                outputs = model(image=net_input, mask=gt_mask, points=points, training=False)
                logits = outputs['instances']
                prev_output = torch.sigmoid(logits)

                points = get_next_points_removeall(prev_output, gt_mask, points, click_idx + 1)

            # Calculate performance metrics
            for res, tar in zip(prev_output, gt_mask):
                np_restored = fn_threshold(fn_tonumpy(res))
                np_target = fn_threshold(fn_tonumpy(tar))

                performance_dict = fn_performance(np_restored, np_target)
                accuracy_lst.append(performance_dict['accuracy'])
                iou_lst.append(performance_dict['iou_f'])
                recall_lst.append(performance_dict['recall'])
                precision_lst.append(performance_dict['precision'])
                dsc_lst.append(performance_dict['DSC'])

    metrics = {
        "accuracy": np.mean(accuracy_lst),
        "iou": np.mean(iou_lst),
        "recall": np.mean(recall_lst),
        "precision": np.mean(precision_lst),
        "dsc": np.mean(dsc_lst),
    }

    print(
        f"IoU: {metrics['iou']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, DSC: {metrics['dsc']:.4f}")
    return metrics