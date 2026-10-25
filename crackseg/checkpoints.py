"""保存和加载模型、优化器及训练记录。"""

from datetime import datetime
import json

import torch


def save_model(model, optimizer, save_path, data_list):
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = save_path / "outputs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    model_save_path = run_dir / "crack_segmentation_model.pth"
    opti_save_path = run_dir / "crack_segmentation_optim.pth"


    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), opti_save_path)

    with open(run_dir / "stats.txt", "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

    print(f"保存模型位置为{run_dir}")


def load_model(model, optimizer, save_path):
    model_path = save_path / "crack_segmentation_model.pth"
    optim_path = save_path / "crack_segmentation_optim.pth"

    model_state = torch.load(model_path, map_location="cuda:0", weights_only=True)
    model.load_state_dict(model_state)

    optim_state = torch.load(optim_path, weights_only=True)
    optimizer.load_state_dict(optim_state)

    with open(save_path / "stats.txt", "r", encoding="utf-8") as f:
        data_list = json.load(f)

    return model, optimizer, data_list


def loadmod(model, save_path):
    model_path = save_path / "crack_segmentation_model.pth"

    model_state = torch.load(model_path, map_location="cuda:0", weights_only=True)
    model.load_state_dict(model_state)

    return model
