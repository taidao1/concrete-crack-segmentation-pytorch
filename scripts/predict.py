"""预测单张照片的裂缝mask，显示结果并判断是否存在裂缝。"""

import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from crackseg.models import setmodel
from crackseg.inference import img_predict, if_exist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setmodel(None).to(device)
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    model_state = torch.load(
        args.model,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(model_state)

    mask, image_tensor = img_predict(args.image, model)
    fig, axes = plt.subplots(1,2)
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(mask.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Mask")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()

    plt.show()
    if if_exist(mask):
        print("存在裂缝")
    else:
        print("未检测到裂缝")


if __name__ == "__main__":
    main()
