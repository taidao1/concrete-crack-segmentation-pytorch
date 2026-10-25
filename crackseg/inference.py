"""整图滑窗预测与裂缝存在判断。"""

import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor

from .data import norm, std
from .metrics import countspixels


def gen_coordinates(image):
    height, width = image.shape[-2:]
    a = 0
    b = 0
    height_list = []
    width_list = []

    while a < height:
        if a + 512 > height:
            a = height - 512
            if a < 0:
                raise ValueError("目前暂不支持尺寸小于512的图")
            height_list.append(a)
            break
        if a + 512 <= height:
            height_list.append(a)
            a += 512

    while b < width:
        if b + 512 > width:
            b = width - 512
            if b < 0:
                raise ValueError("目前暂不支持尺寸小于512的图")
            width_list.append(b)
            break
        if b + 512 <= width:
            width_list.append(b)
            b += 512
    return height_list, width_list


def img_crop(image, a, b):
    '''裁剪图片'''
    c_image = image[:,a:a+512,b:b+512]
    return c_image


def forecast_funct(image, model):
    height, width = image.shape[-2:]
    score_sum = torch.zeros(
        (height, width),
        dtype=torch.float32,
        device=image.device
    )
    count_map = torch.zeros_like(score_sum)

    height_list, width_list = gen_coordinates(image)

    model.eval()
    model_device = next(model.parameters()).device
    with torch.no_grad():
        for i in height_list:
            for j in width_list:
                c_image = img_crop(image, i, j)
                logits = model(c_image.unsqueeze(0).to(model_device))
                scores = logits.sigmoid()[0, 0].to(score_sum.device)
                score_sum[i:i+512, j:j+512] += scores
                count_map[i:i+512, j:j+512] += 1
        if (count_map == 0).any():
            raise RuntimeError("存在未被窗口覆盖的像素")
        mean_scores = score_sum / count_map
        mask = mean_scores >= 0.5
    return mask


def if_exist(mask, threshold_rate = 0.003):
    if mask.sum() / mask.numel() >= threshold_rate:
        return True
    else:
        return False


def img_predict(image_path, model):
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image_tensor = pil_to_tensor(image)

    image_tensor = norm(image_tensor)
    image_tensor = std(image_tensor)



    mask = forecast_funct(image_tensor, model)


    return mask, image_tensor


def pred_compare(image_path, r_mask_path, model):
    mask, image_tensor = img_predict(image_path, model)
    with Image.open(r_mask_path) as source:
        r_mask = ImageOps.exif_transpose(source).convert("L")
        mask_tensor = pil_to_tensor(r_mask)
        target = mask_tensor.squeeze(0) >= 128

    tp, fp, fn = countspixels(mask, target)



    return tp, fp, fn
