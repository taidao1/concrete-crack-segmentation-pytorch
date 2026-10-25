"""检查三张外部照片的裂缝判断，结果写入outputs。"""

import argparse
import json
from time import perf_counter
import torch
from PIL import Image

from crackseg.paths import PROJECT_ROOT
from crackseg.models import setmodel
from crackseg.checkpoints import loadmod
from crackseg.inference import img_predict, if_exist


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    project_root = PROJECT_ROOT
    source_path = project_root / "reports" / "evaluation" / "external_presence_smoke" / "sources.json"
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    report_path = project_root / "outputs" / "external_presence_check"
    report_path.mkdir(parents=True, exist_ok=True)
    load_path = project_root / "outputs" / "run_20261024_181934_060182"
    threshold_rate = 0.003
    started = perf_counter()
    model = loadmod(setmodel(None).to("cuda"), load_path)
    cases = []
    tn, fp = 0, 0

    for sample in sources["images"]:
        image_path = project_root / sample["path"]
        before_prediction = perf_counter()
        mask, _ = img_predict(image_path, model)
        predicted_exists = if_exist(mask, threshold_rate=threshold_rate)
        elapsed = perf_counter() - before_prediction
        predicted_pixels = int(mask.sum().item())
        mask_path = report_path / (image_path.stem + "_mask.png")
        Image.fromarray(mask.cpu().numpy().astype("uint8") * 255).save(mask_path)
        if sample["expected_has_crack"] is False:
            fp += int(predicted_exists)
            tn += int(not predicted_exists)
        case = {
            "image": sample["path"],
            "expected_has_crack": sample["expected_has_crack"],
            "label_basis": sample["label_basis"],
            "size_hw": list(mask.shape),
            "predicted_has_crack": predicted_exists,
            "predicted_crack_pixels": predicted_pixels,
            "predicted_crack_fraction": predicted_pixels / mask.numel(),
            "prediction_seconds": elapsed,
            "mask": mask_path.relative_to(project_root).as_posix(),
        }
        cases.append(case)
        print(json.dumps(case, ensure_ascii=False), flush=True)

    negative_count = tn + fp
    result = {
        "model_directory": load_path.relative_to(project_root).as_posix(),
        "device": str(next(model.parameters()).device),
        "threshold_rate": threshold_rate,
        "source_manifest": source_path.relative_to(project_root).as_posix(),
        "pixel_probability_threshold": 0.5,
        "input_resize": False,
        "total_images": len(cases),
        "negative_images": negative_count,
        "uncertain_images": len(cases) - negative_count,
        "tn": tn,
        "fp": fp,
        "negative_correct_rejection_rate": tn / negative_count if negative_count else None,
        "false_positive_rate": fp / negative_count if negative_count else None,
        "positive_recall": None,
        "elapsed_seconds": perf_counter() - started,
        "cases": cases,
    }
    (report_path / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"图片级判断阈值（预测裂缝像素占比）：{threshold_rate:.2%}")
    print(f"图片总数：{len(cases)}，暂定无裂缝：{negative_count}，标签不确定：{len(cases) - negative_count}")
    print(f"正确排除TN：{tn}，疑似误报FP：{fp}")
    print(f"无裂缝样本正确排除率：{result['negative_correct_rejection_rate']}")
    print(f"无裂缝样本误报率：{result['false_positive_rate']}")
    print("本次仅为少量弱标签外部照片的试用，不能据此推断总体准确率。")


if __name__ == "__main__":
    main()
