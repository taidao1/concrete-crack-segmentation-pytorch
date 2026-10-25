"""成对图像读取、预处理与自定义 Dataset。"""

import random
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor


def pair_samples(rgb_path, BW_path, number="001"):
    '''这个函数用于从数据集取出一对图像数据并将其转化为tensor张量'''
    suffix = ".jpg"
    image_number = number + suffix
    rgb_path = Path(rgb_path) / image_number
    if not rgb_path.is_file():
        rgb_path = rgb_path.with_suffix(".JPG")
    BW_path = Path(BW_path) / image_number
    if not BW_path.is_file():
        BW_path = BW_path.with_suffix(".JPG")
    with Image.open(rgb_path) as source:
        image = ImageOps.exif_transpose(source)
        image = image.convert("RGB")
        rgb_tensor = pil_to_tensor(image)
    with Image.open(BW_path) as source:
        image = ImageOps.exif_transpose(source)
        image = image.convert("L")
        BW_tensor = pil_to_tensor(image)
    return rgb_tensor, BW_tensor


def image_list(rgb_path, BW_path):
    '''对应数据集对应的编号列表'''
    rgb_list = []
    BW_list = []
    for rgb_images in rgb_path.iterdir():
        if rgb_images.is_file() and rgb_images.suffix.lower() == ".jpg":
            rgb_list.append(rgb_images.stem)
    for BW_images in BW_path.iterdir():
        if BW_images.is_file() and BW_images.suffix.lower() == ".jpg":
            BW_list.append(BW_images.stem)
    rgb_list = sorted(rgb_list)
    BW_list = sorted(BW_list)
    if rgb_list == BW_list:
        return rgb_list
    else:
        raise ValueError("列表不一致")


def binary(images, threshold=128):
    '''二值化mask图像'''
    images = images >= threshold
    images = images.float()
    return images


def norm(images):
    '''归一化原始图像'''
    images = images.float() / 255.0
    return images


def std(images):
    '''标准化原始图像'''
    normalizer = v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return normalizer(images)


def rand_range(image, H, W):
    height, width = image.shape[-2:]
    if H > height or W > width:
        raise ValueError("请输入小于照片尺寸的裁剪大小")
    a = random.randint(0,height-H)
    c = random.randint(0,width-W)
    b = a + H
    d = c + W
    return a, b, c, d


def cropped(image, mask, a=100, b=612, c=200, d=712):
    '''裁剪图片'''
    height, width = image.shape[-2:]
    if not (0 <= a < b <= height and 0 <= c < d <= width):
        raise IndexError("超出图像边界")
    image = image[:,a:b,c:d]
    mask = mask[:,a:b,c:d]
    return image, mask


class image_datasets(Dataset):
    '''这个dataset类会根据导入原图路径，mask路径和编号对原图进行归一和裁剪，对mask图像进行二值化和裁剪'''
    def __init__(self, rgb_path, BW_path, sample_ids, random_crop = False):
        self.list = sample_ids
        self.rgb_path = rgb_path
        self.BW_path = BW_path
        self.random_crop = random_crop

    def __len__(self):
        return len(self.list)

    def __getitem__(self, key):
        rgb, mask = pair_samples(self.rgb_path, self.BW_path, self.list[key])
        rgb = norm(rgb)
        rgb = std(rgb)
        mask = binary(mask)
        if self.random_crop:
            a, b, c, d = rand_range(rgb, 512, 512)
            rgb, mask = cropped(rgb, mask, a, b, c, d)
        else:
            rgb, mask = cropped(rgb, mask)
        return rgb, mask
