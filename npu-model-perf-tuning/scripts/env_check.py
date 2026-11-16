#!/usr/bin/env python3
"""环境自检：NPU 硬件 / CANN / torch / torch_npu 版本 / transfer_to_npu 可用性。

用法:
    python env_check.py            # 人类可读
    python env_check.py --json     # 机器可读（进 manifest / 报告 §2）

只依赖标准库 + 已安装的 torch/torch_npu；任何缺失都降级为字符串说明，不抛异常。
"""
import argparse
import json
import platform
import subprocess
import sys


def _try(fn, default=None):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


def collect():
    info = {}
    info["python"] = sys.version.split()[0]
    info["platform"] = platform.platform()

    def _torch():
        import torch
        d = {"torch": torch.__version__}
        return d

    info.update(_try(_torch, {}) or {})

    def _npu():
        import torch_npu  # noqa: F401
        import torch
        d = {"torch_npu": getattr(torch_npu, "__version__", "unknown")}
        try:
            d["npu_available"] = bool(torch.npu.is_available())
            d["device_count"] = int(torch.npu.device_count())
            if d["npu_available"]:
                d["device_name_0"] = torch.npu.get_device_name(0)
        except Exception as e:  # noqa: BLE001
            d["npu_probe_error"] = str(e)
        return d

    info.update(_try(_npu, {"torch_npu": "<not installed>"}) or {})

    def _transfer():
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
        return {"transfer_to_npu": "importable"}

    info.update(_try(_transfer, {"transfer_to_npu": "<import failed>"}) or {})

    def _smi():
        out = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=20
        )
        head = "\n".join((out.stdout or "").splitlines()[:6])
        return {"npu_smi_ok": out.returncode == 0, "npu_smi_head": head}

    info.update(_try(_smi, {"npu_smi_ok": "<npu-smi not available>"}) or {})
    return info


def main():
    ap = argparse.ArgumentParser(description="NPU 环境自检")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    a = ap.parse_args()
    info = collect()
    if a.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return
    print("== NPU 环境自检 ==")
    for k, v in info.items():
        if isinstance(v, str) and "\n" in v:
            print(f"{k}:\n{v}\n")
        else:
            print(f"{k:18}: {v}")


if __name__ == "__main__":
    main()
