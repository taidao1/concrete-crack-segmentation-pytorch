"""从检查点恢复训练，并固定BN运行统计。"""

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
from crackseg.checkpoints import save_model, load_model


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
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    data_list = []
    load_path = project_root / "outputs" / "run_20261024_174146_218961"
    model, optimizer, data_list = load_model(model, optimizer, load_path)
    for group in optimizer.param_groups:
        group["lr"] = 3e-5

    print("实际学习率：", [group["lr"] for group in optimizer.param_groups])
    for record in data_list:
        if "训练轮数：" in record:
            start_count = record["训练轮数："] + 1
        if "保存最佳模型iou为：" in record:
            max_iou = record["保存最佳模型iou为："]

    for epoch in range(5):
        data_dict = {}
        mean_loss, iou = train_one_epoch(
            model, tqdm(train_loader, desc="训练", unit="batch"), loss_fn, optimizer,
            device, freeze_bn_stats=True
        )
        print(f"第{epoch + start_count}次训练平均损失为{mean_loss}")
        print(f"第{epoch + start_count}次训练IoU为{iou}")
        print("模型设备：", next(model.parameters()).device)
        data_dict["训练轮数："] = epoch + start_count
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
