"""按指定划分评价整图分割和图片级裂缝判断。"""

import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor

from crackseg.paths import PROJECT_ROOT
from crackseg.models import setmodel
from crackseg.checkpoints import loadmod
from crackseg.inference import img_predict, if_exist
from crackseg.metrics import countspixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('val', 'test'), default='val', help='val用于验证；test用于冻结版本的最终评价。')
    parser.add_argument('--model-dir', type=Path, default=PROJECT_ROOT / 'outputs' / 'run_20261024_181934_060182', help='包含crack_segmentation_model.pth的目录。')
    args = parser.parse_args()
    project_root = PROJECT_ROOT
    sample_ids_path = project_root / "data" / f"{args.split}_ids.txt"
    sample_ids = sample_ids_path.read_text(encoding="utf-8").splitlines()
    images_path = project_root / "data" / "raw" / "metu_v1" / "rgb"
    r_masks_path = project_root / "data" / "raw" / "metu_v1" / "BW"
    load_path = args.model_dir

    device = torch.device("cuda")
    model = setmodel(None).to(device)
    model = loadmod(model, load_path)

    tp = 0
    fp = 0
    fn = 0
    image_tp = 0
    image_fp = 0
    image_fn = 0
    image_tn = 0
    crack_pixel_threshold = 0.003
    suffix = ".jpg"

    for number in tqdm(sample_ids, desc="整图测试" if args.split == "test" else "整图验证", unit="image"):
        image_number = number + suffix
        image_path = Path(images_path) / image_number
        if not image_path.is_file():
            image_path = image_path.with_suffix(".JPG")
        r_mask_path = Path(r_masks_path) / image_number
        if not r_mask_path.is_file():
            r_mask_path = r_mask_path.with_suffix(".JPG")
        mask, _ = img_predict(image_path, model)
        with Image.open(r_mask_path) as source:
            r_mask = ImageOps.exif_transpose(source).convert("L")
            target = pil_to_tensor(r_mask).squeeze(0) >= 128

        one_tp, one_fp, one_fn = countspixels(mask, target)

        tp += one_tp
        fp += one_fp
        fn += one_fn

        predicted_exists = if_exist(mask, threshold_rate=crack_pixel_threshold)
        actual_exists = target.any().item()
        if actual_exists and predicted_exists:
            image_tp += 1
        elif not actual_exists and predicted_exists:
            image_fp += 1
        elif actual_exists and not predicted_exists:
            image_fn += 1
        else:
            image_tn += 1

    denominator = tp + fp + fn
    iou = tp / denominator if denominator > 0 else None
    Precision = tp / (tp + fp) if tp + fp > 0 else None
    Recall = tp / (tp + fn) if tp + fn > 0 else None

    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else None
    print(f"像素 TP={tp}, FP={fp}, FN={fn}")
    print(f"全局 F1：{f1}")

    print(f"预测全局IoU：{iou}\n预测全局Precision：{Precision}\n预测全局Recall：{Recall}")

    total_images = image_tp + image_fp + image_fn + image_tn
    positive_images = image_tp + image_fn
    negative_images = image_fp + image_tn
    predicted_positive_images = image_tp + image_fp
    image_accuracy = (image_tp + image_tn) / total_images if total_images > 0 else None
    image_precision = image_tp / predicted_positive_images if predicted_positive_images > 0 else None
    image_recall = image_tp / positive_images if positive_images > 0 else None
    image_false_positive_rate = image_fp / negative_images if negative_images > 0 else None

    print(f"图片级判断阈值：{crack_pixel_threshold:.2%}")
    print(f"图片总数：{total_images}，真实有裂缝：{positive_images}，真实无裂缝：{negative_images}")
    print(f"预测有裂缝：{predicted_positive_images}，预测无裂缝：{image_fn + image_tn}")
    print(f"图片级TP（正确检出）：{image_tp}，FP（误报）：{image_fp}，FN（漏检）：{image_fn}，TN（正确排除）：{image_tn}")
    print(f"图片级准确率：{image_accuracy}\n图片级Precision：{image_precision}\n图片级Recall：{image_recall}")
    print(f"无裂缝图片误报率：{image_false_positive_rate}")
    if negative_images == 0:
        print("当前数据没有无裂缝图片，无法评价误报率；此时准确率仅反映有裂缝图片的检出情况。")


if __name__ == "__main__":
    main()
