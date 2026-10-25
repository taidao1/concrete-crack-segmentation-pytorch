import argparse
import importlib
import json
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def check_environment():
    report = {
        "checked_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {},
        "errors": [],
    }
    modules = {
        "torch": "torch",
        "torchvision": "torchvision",
        "numpy": "numpy",
        "pillow": "PIL",
        "matplotlib": "matplotlib",
        "pyyaml": "yaml",
        "tqdm": "tqdm",
    }
    for package, module_name in modules.items():
        try:
            module = importlib.import_module(module_name)
            report["packages"][package] = module.__version__
        except Exception as exc:
            report["errors"].append(f"{package}: {type(exc).__name__}: {exc}")

    torch = sys.modules.get("torch")
    if torch is None:
        report["passed"] = False
        return report

    report["cuda_available"] = torch.cuda.is_available()
    report["torch_cuda_runtime"] = torch.version.cuda
    if not report["cuda_available"]:
        report["errors"].append("CUDA is unavailable to this interpreter.")
        report["passed"] = False
        return report

    try:
        device = torch.device("cuda:0")
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["gpu_total_memory_mib"] = round(
            torch.cuda.get_device_properties(device).total_memory / 1024**2
        )
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats(device)
        layer = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1).to(device)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        image = torch.randn(2, 3, 64, 64, device=device)
        target = torch.zeros(2, 1, 64, 64, device=device)
        before = layer.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        logits = layer(image)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        gradients_finite = all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in layer.parameters()
        )
        optimizer.step()
        torch.cuda.synchronize(device)
        report["smoke_test"] = {
            "device": str(logits.device),
            "input_shape": list(image.shape),
            "output_shape": list(logits.shape),
            "loss": loss.item(),
            "logits_finite": torch.isfinite(logits).all().item(),
            "loss_finite": torch.isfinite(loss).item(),
            "gradients_finite": gradients_finite,
            "parameters_updated": not torch.equal(before, layer.weight.detach()),
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(device) / 1024**2, 2
            ),
        }
        for key in (
            "logits_finite", "loss_finite", "gradients_finite", "parameters_updated"
        ):
            if not report["smoke_test"][key]:
                report["errors"].append(f"Smoke test failed: {key}")
    except Exception as exc:
        report["errors"].append(f"GPU smoke test: {type(exc).__name__}: {exc}")

    report["passed"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optionally save the JSON report.")
    args = parser.parse_args()
    report = check_environment()
    content = json.dumps(report, ensure_ascii=False, indent=2)
    print(content)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
