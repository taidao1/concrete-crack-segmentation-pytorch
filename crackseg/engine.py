"""单轮训练、验证及整图验证。"""

from pathlib import Path

import torch
from tqdm import tqdm

from .metrics import countspixels
from .inference import pred_compare


def train_one_epoch(model, loader, loss_fn, optimizer, device, freeze_bn_stats=False):
    model.train()
    if freeze_bn_stats:
        # 只固定运行统计；BN 的 weight / bias 仍保留梯度。
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
    loss_sum = 0.0
    sample_count = 0
    tp = 0
    fn = 0
    fp = 0
    for batch_images, batch_labels in loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        scores = model(batch_images)
        loss = loss_fn(scores, batch_labels)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        loss_sum += batch_loss * len(batch_images)
        sample_count += len(batch_images)
        pred = scores.detach().sigmoid() >= 0.5
        target = batch_labels >= 0.5
        batch_tp, batch_fp, batch_fn = countspixels(pred, target)
        tp += batch_tp
        fn += batch_fn
        fp += batch_fp

    denominator = tp + fn + fp

    if sample_count == 0:
        raise ValueError("训练数据不能为空")
    if denominator == 0:
        iou = None
    else:
        iou = tp / denominator

    return loss_sum / sample_count, iou


def validation(model, loader, loss_fn, device):
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    tp = 0
    fn = 0
    fp = 0
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            scores = model(batch_images)
            loss = loss_fn(scores, batch_labels)
            batch_loss = loss.item()
            loss_sum += batch_loss * len(batch_images)
            sample_count += len(batch_images)
            pred = scores.detach().sigmoid() >= 0.5
            target = batch_labels >= 0.5
            batch_tp, batch_fp, batch_fn = countspixels(pred, target)
            tp += batch_tp
            fn += batch_fn
            fp += batch_fp

        denominator = tp + fn + fp

    if sample_count == 0:
        raise ValueError("验证数据不能为空")
    if denominator == 0:
        iou = None
    else:
        iou = tp / denominator

    return loss_sum / sample_count, iou


def w_image_evaluate(images_path, r_masks_path, val_ids, model):
    tp = 0
    fp = 0
    fn = 0
    suffix = ".jpg"

    for number in tqdm(val_ids, desc="整图验证", unit="image"):
        image_number = number + suffix
        image_path = Path(images_path) / image_number
        if not image_path.is_file():
            image_path = image_path.with_suffix(".JPG")
        r_mask_path = Path(r_masks_path) / image_number
        if not r_mask_path.is_file():
            r_mask_path = r_mask_path.with_suffix(".JPG")
        one_tp, one_fp, one_fn = pred_compare(image_path, r_mask_path, model)
        tp += one_tp
        fp += one_fp
        fn += one_fn

    denominator = tp + fp + fn
    iou = tp / denominator if denominator > 0 else None
    return iou
