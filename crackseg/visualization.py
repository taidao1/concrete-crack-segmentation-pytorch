"""原图、真实标注和预测结果的显示。"""

import matplotlib.pyplot as plt
import torch


def visu(image, p_mask, r_mask):
    """显示单张样本：image 为标准化的 [3,H,W]，p_mask/r_mask 为预测/真实的 [1,H,W] 二值图。"""
    # Dataset 做过 (RGB - mean) / std；显示前还原为 0～1 的 RGB。
    image = image.detach().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image * std + mean).clamp(0, 1)

    # imshow 的 RGB 输入是 [H,W,3]，灰度输入是 [H,W]。
    image = image.permute(1, 2, 0).numpy()
    p_mask = p_mask.detach().cpu().squeeze(0).numpy()
    r_mask = r_mask.detach().cpu().squeeze(0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(r_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Ground truth")
    axes[2].imshow(p_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()
