"""训练U-Net裂缝分割模型。"""

import argparse
import random
import torch
from torch import nn
from tqdm import tqdm

from crackseg.paths import PROJECT_ROOT
from crackseg.models import setmodel
from crackseg.data import image_list
from crackseg.loaders import split_list, makedatasets, makedataloader
from crackseg.engine import train_one_epoch, validation, w_image_evaluate
from crackseg.checkpoints import save_model


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    project_root = PROJECT_ROOT
    rgb_path = project_root / "data" / "raw" / "metu_v1" / "rgb"
    BW_path = project_root / "data" / "raw" / "metu_v1" / "BW"

    device = torch.device("cuda")
    model = setmodel().to(device)
    sample_ids = image_list(rgb_path, BW_path)
    train_ids, val_ids, test_ids = split_list(sample_ids)
    train_dataset, val_dataset, test_dataset = makedatasets(rgb_path, BW_path)
    val_loader = makedataloader(val_dataset, False)
    train_loader = makedataloader(train_dataset)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    data_list = []
    max_iou = -1

    for epoch in range(20):
        data_dict = {}
        mean_loss, iou = train_one_epoch(model, tqdm(train_loader, desc="训练", unit="batch"), loss_fn, optimizer, device)
        print(f"第{epoch}次训练平均损失为{mean_loss}")
        print(f"第{epoch}次训练IoU为{iou}")
        print("模型设备：", next(model.parameters()).device)
        data_dict["训练轮数："] = epoch
        data_dict["训练平均损失："] = mean_loss
        data_dict["训练IoU："] = iou
        data_list.append(data_dict)

        mean_loss, iou = validation(model, tqdm(val_loader, desc="验证", unit="batch"), loss_fn, device)
        data_list.append({"验证平均损失：":mean_loss, "验证IoU：":iou})
        print(f"验证平均损失为{mean_loss}")
        print(f"验证IoU为{iou}")

        val_iou = w_image_evaluate(rgb_path, BW_path, val_ids, model)

        print(f"整图验证IoU为{val_iou}")
        data_dict["整图验证IoU："] = val_iou

        if val_iou is not None and val_iou > max_iou:
            max_iou = val_iou
            data_dict["保存最佳模型iou为："] = max_iou
            save_path = project_root
            save_model(model, optimizer, save_path, data_list)


if __name__ == "__main__":
    main()
