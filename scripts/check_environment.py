#!/usr/bin/env python3
"""Print environment capabilities, no training or downloads."""
import importlib
import json
import platform
import sys


def optional_version(mod):
    try:
        obj = importlib.import_module(mod)
        return getattr(obj, "__version__", "installed")
    except Exception as exc:
        return "unavailable: " + str(exc)


def main():
    result = {"platform": platform.platform(), "python": sys.version.split()[0],
              "torch": optional_version("torch"), "mmcv": optional_version("mmcv"),
              "mmengine": optional_version("mmengine"),
              "mmdet": optional_version("mmdet")}
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            result["gpu_total_vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
            # A minimal allocated GPU tensor catches obvious runtime failures.
            tensor = torch.ones((16, 16), device="cuda")
            result["cuda_tensor_smoke"] = float((tensor @ tensor).sum().item()) == 4096.0
    except Exception as exc:
        result["cuda_probe_error"] = str(exc)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
