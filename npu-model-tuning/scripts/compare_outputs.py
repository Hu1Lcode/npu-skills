#!/usr/bin/env python3
"""输出级精度对比：张量 / 图像 / 视频（可选）。

用法:
    python compare_outputs.py reference.pt candidate.pt --json out.json
    python compare_outputs.py ref.png cand.png
    python compare_outputs.py ref.mp4 cand.mp4 --frames 10

支持:
    .pt / .pth / .npy        张量（多个张量需 shape 一致；也可传目录）
    .png / .jpg / .bmp       图像
    .mp4 / .avi / .mov       视频（需要 PyAV；比较等间隔采样帧）
"""
import argparse
import json
import os
import sys

import numpy as np


def load_tensor(path):
    if path.endswith(".npy"):
        return np.load(path)
    import torch

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        # 常见 checkpoint/dump 结构：优先取唯一张量或 "output"/"latents" 字段
        tensor_values = [value for value in obj.values() if hasattr(value, "detach")]
        if len(tensor_values) == 1:
            obj = tensor_values[0]
        elif "output" in obj:
            obj = obj["output"]
        else:
            raise ValueError(f"{path} 中有多个张量，无法自动选择；请先提取单个张量")
    if hasattr(obj, "detach"):
        obj = obj.detach().float().cpu().numpy()
    return np.asarray(obj)


def load_image(path):
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_video(path, frames=None):
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("比较视频需要安装 PyAV（pip install av）") from exc
    with av.open(path) as container:
        stream = container.streams.video[0]
        images = [frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0 for frame in container.decode(stream)]
    if not images:
        raise ValueError(f"{path} 没有可解码帧")
    if frames and frames > 0 and len(images) > frames:
        indices = np.linspace(0, len(images) - 1, frames).astype(int)
        images = [images[index] for index in indices]
    return np.stack(images, axis=0)


def load_any(path, frames=None):
    lower = path.lower()
    if lower.endswith((".pt", ".pth", ".npy")):
        return load_tensor(path), "tensor"
    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        return load_image(path), "image"
    if lower.endswith((".mp4", ".avi", ".mov", ".mkv")):
        return load_video(path, frames), "video"
    raise ValueError(f"不支持的格式: {path}")


def metrics(reference, candidate):
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape 不一致: reference={reference.shape}, candidate={candidate.shape}")
    diff = reference - candidate
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    mean_abs = float(abs_diff.mean()) if abs_diff.size else 0.0
    rmse = float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0
    ref_flat = reference.reshape(-1)
    cand_flat = candidate.reshape(-1)
    denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(cand_flat))
    cosine_defined = denom > 0
    cosine = float(np.dot(ref_flat, cand_flat) / denom) if cosine_defined else None
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(ref_flat) + 1e-12))
    mse = float(np.mean(diff**2)) if diff.size else 0.0
    psnr = float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
    result = {
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "rmse": rmse,
        "cosine": cosine,
        "cosine_defined": cosine_defined,
        "rel_l2": rel_l2,
        "psnr": psnr,
        "shape": list(reference.shape),
        "element_count": int(reference.size),
    }
    try:
        from skimage.metrics import structural_similarity as ssim

        if reference.ndim >= 2:
            result["ssim"] = float(ssim(reference, candidate, data_range=1.0, channel_axis=-1 if reference.ndim == 3 else None))
    except Exception:  # noqa: BLE001
        pass
    return result


def main():
    parser = argparse.ArgumentParser(description="输出级精度对比")
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--frames", type=int, default=10, help="视频采样帧数")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    reference, kind = load_any(args.reference, args.frames)
    candidate, _ = load_any(args.candidate, args.frames)
    if kind == "video" and reference.shape[0] != candidate.shape[0]:
        raise SystemExit(f"视频帧数不一致: {reference.shape[0]} vs {candidate.shape[0]}")

    result = metrics(reference, candidate)
    result["type"] = kind
    print(f"type: {kind}")
    print(f"shape: {result['shape']}")
    print(f"max_abs_diff : {result['max_abs_diff']:.6e}")
    print(f"mean_abs_diff: {result['mean_abs_diff']:.6e}")
    print(f"rmse         : {result['rmse']:.6e}")
    cosine_text = f"{result['cosine']:.8f}" if result["cosine"] is not None else "undefined（reference 或 candidate 全零，需人工确认口径）"
    print(f"cosine       : {cosine_text}")
    print(f"rel_l2       : {result['rel_l2']:.6e}")
    print(f"psnr         : {result['psnr']:.4f}")
    if "ssim" in result:
        print(f"ssim         : {result['ssim']:.6f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"[compare_outputs] wrote {args.json}")


if __name__ == "__main__":
    main()
