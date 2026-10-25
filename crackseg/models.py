"""模型创建。"""

import segmentation_models_pytorch as smp


def setmodel(type="imagenet"):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=type,
        in_channels=3,
        classes=1,
        activation=None
    )
    return model
