"""评价训练集和验证集的固定裁剪，并显示含裂缝样例。"""

import argparse
import torch
from torch import nn
from tqdm import tqdm

from crackseg.paths import PROJECT_ROOT
from crackseg.models import setmodel
from crackseg.data import image_list
from crackseg.loaders import split_list, makedatasets, makedataloader
from crackseg.engine import validation
from crackseg.checkpoints import load_model
from crackseg.visualization import visu


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    project_root = PROJECT_ROOT
    rgb_path = project_root / "data" / "raw" / "metu_v1" / "rgb"
    BW_path = project_root / "data" / "raw" / "metu_v1" / "BW"

    device = torch.device("cuda")
    model = setmodel().to(device)
    sample_ids = image_list(rgb_path, BW_path)
    train_ids, val_ids, test_ids = split_list(sample_ids)
    train_dataset, val_dataset, test_dataset = makedatasets(rgb_path, BW_path, rand_crop=False)
    val_loader = makedataloader(val_dataset, False)
    train_loader = makedataloader(train_dataset, False)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    data_list = []
    load_path = project_root / "outputs" / "run_20261024_181934_060182"
    model, optimizer, data_list = load_model(model, optimizer, load_path)

    mean_loss, iou = validation(model, tqdm(train_loader, desc="训练集验证", unit="batch"), loss_fn, device)
    data_list.append({"训练集验证平均损失：":mean_loss, "训练集验证IoU：":iou})
    print(f"训练集验证平均损失为{mean_loss}")
    print(f"训练集验证IoU为{iou}")

    mean_loss, iou = validation(model, tqdm(val_loader, desc="验证", unit="batch"), loss_fn, device)
    data_list.append({"验证集验证平均损失：":mean_loss, "验证集验证IoU：":iou})
    print(f"验证集验证平均损失为{mean_loss}")
    print(f"验证集验证IoU为{iou}")

    model.eval()
    shown = False

    for images, masks in val_loader:
        with torch.no_grad():
            preds = model(images.to(device)).sigmoid() >= 0.5
        for i in range(len(images)):
            if masks[i].sum().item() > 0:
                print("真实裂缝像素数：", masks[i].sum().item())
                print("预测裂缝像素数：", preds[i].sum().item())
                visu(images[i], preds[i], masks[i])
                shown = True
                break
        if shown:
            break


if __name__ == "__main__":
    main()
