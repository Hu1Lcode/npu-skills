#!/usr/bin/env python3
"""NPU 环境自检：硬件、CANN/驱动、torch/torch_npu、设备、环境变量、git 信息。

用法:
    python env_check.py                 # 人类可读
    python env_check.py --json          # 机器可读
    python env_check.py --repo /path/to/model --json

只依赖标准库 + 已安装的 torch/torch_npu；任何探测失败都降级为说明字符串，不抛异常。
"""
import argparse
import glob
import json
import os
import platform
import socket
import subprocess
import sys
import warnings

ENV_KEYS = [
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_HOME_PATH",
    "TASK_QUEUE_ENABLE",
    "CPU_AFFINITY_CONF",
    "PYTORCH_NPU_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
    "MULTI_STREAM_MEMORY_REUSE",
    "HCCL_BUFFSIZE",
    "HCCL_ALGO",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "ASCEND_OPP_CACHE",
    "TORCH_TRANSFER_TO_NPU",
    "NPU_TUNE_PROBE",
]

# 报告第 1 节需要的框架/依赖版本；按目标仓库实际情况补充
PACKAGE_NAMES = [
    "torch",
    "torch_npu",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "mindiesd",
    "mindspeed",
    "megatron-core",
]


def _try(fn, default=None):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return default if default is not None else f"<error: {exc}>"


def _read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _cann_version():
    candidates = []
    home = os.environ.get("ASCEND_HOME_PATH")
    if home:
        candidates.append(home)
    candidates.append("/usr/local/Ascend/ascend-toolkit/latest")
    for root in candidates:
        for pattern in ("*/ascend_toolkit_install.info", "ascend_toolkit_install.info"):
            for path in glob.glob(os.path.join(root, pattern)):
                text = _read_text(path)
                info = {}
                for line in text.splitlines():
                    if "=" in line:
                        key, value = line.split("=", 1)
                        info[key.strip()] = value.strip()
                if info:
                    info["source"] = path
                    return info
    return "<not found>"


def _driver_version():
    path = "/usr/local/Ascend/driver/version.info"
    if os.path.exists(path):
        info = {}
        for line in _read_text(path).splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                info[key.strip()] = value.strip()
        return info
    return "<not found>"


def _npu_smi_head():
    try:
        result = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=20, check=False
        )
        lines = (result.stdout or "").splitlines()
        return {
            "returncode": result.returncode,
            "head": "\n".join(lines[:6]),
            "version_line": next((line.strip("| ").strip() for line in lines if "npu-smi" in line), ""),
        }
    except Exception as exc:  # noqa: BLE001
        return {"returncode": None, "head": f"<error: {exc}>", "version_line": ""}


def _torch_info():
    import torch

    info = {"torch": torch.__version__}
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        info["cuda_available"] = "<error>"
    return info


def _package_versions():
    from importlib import metadata

    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001
            versions[name] = f"<error: {exc}>"
    return versions


def _npu_info():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
        import torch_npu  # noqa: F401

    info = {"torch_npu": getattr(torch_npu, "__version__", "unknown")}
    try:
        info["npu_available"] = bool(torch.npu.is_available())
        info["device_count"] = int(torch.npu.device_count())
        if info["npu_available"]:
            devices = []
            for index in range(info["device_count"]):
                entry = {"index": index}
                try:
                    entry["name"] = torch.npu.get_device_name(index)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    props = torch.npu.get_device_properties(index)
                    entry["total_memory_gib"] = round(props.total_memory / (1024**3), 2)
                    entry["soc_version"] = getattr(props, "name", None) or getattr(props, "soc_version", None)
                except Exception:  # noqa: BLE001
                    pass
                devices.append(entry)
            info["devices"] = devices
    except Exception as exc:  # noqa: BLE001
        info["npu_probe_error"] = str(exc)
    return info


def _transfer_info():
    # transfer_to_npu 在 import 期间会调用 warnings.filterwarnings(action='once') 并打印大段
    # ImportWarning/RuntimeWarning（还会带出 torch.jit 的 DeprecationWarning）。这些是已知噪音，
    # 用 redirect_stderr 隔离；若 import 真的失败，异常本身仍会由 _try 捕获。
    import contextlib
    import io

    buffer = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stderr(buffer):
            from torch_npu.contrib import transfer_to_npu  # noqa: F401

    return {"transfer_to_npu": "importable"}


def _git_info(repo):
    if not repo:
        return "<not provided>"

    def run(*args):
        return subprocess.run(
            ["git", "-C", repo, *args], capture_output=True, text=True, timeout=15, check=False
        ).stdout.strip()

    if not run("rev-parse", "--is-inside-work-tree") == "true":
        return {"is_git": False, "path": repo}
    return {
        "is_git": True,
        "path": repo,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": run("rev-parse", "HEAD"),
        "commit_short": run("rev-parse", "--short", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
        "remote": run("config", "--get", "remote.origin.url"),
    }


def collect(repo=None):
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cann": _try(_cann_version, "<not found>"),
        "driver": _try(_driver_version, "<not found>"),
        "npu_smi": _npu_smi_head(),
        "packages": _try(_package_versions, {}),
        "env": {key: os.environ.get(key, "<unset>") for key in ENV_KEYS},
        "git": _try(lambda: _git_info(repo or os.getcwd()), "<not provided>"),
    }
    info.update(_try(_torch_info, {"torch": "<not installed>"}))
    info.update(_try(_npu_info, {"torch_npu": "<not installed>"}))
    info.update(_try(_transfer_info, {"transfer_to_npu": "<import failed>"}))
    return info


def main():
    parser = argparse.ArgumentParser(description="NPU 环境自检")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--repo", default=None, help="代码仓库路径（用于 git 信息）")
    args = parser.parse_args()
    info = collect(args.repo)
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return
    print("== NPU 环境自检 ==")
    for key, value in info.items():
        if isinstance(value, (dict, list)):
            print(f"{key}:")
            print(json.dumps(value, ensure_ascii=False, indent=2))
        else:
            print(f"{key:18}: {value}")


if __name__ == "__main__":
    main()
