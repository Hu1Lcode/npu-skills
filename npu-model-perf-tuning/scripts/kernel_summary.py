#!/usr/bin/env python3
"""kernel 级汇总：按 Task Type 统计耗时占比 + Top-N 热点 kernel。

输入：torch_npu.profiler 产出的某个 prof 目录（会递归查找 kernel_details.csv）。
用法:
    python kernel_summary.py <prof_dir> [--top 20] [--csv out.csv]

判读：
    AICPU 高   -> 算子没落 AI Core，找融合算子/float32 化
    AICore/AIV/MIX 高但慢 -> memory-bound / 核利用率 / 切分
    HCCL 高    -> 通信瓶颈
    memcpy/format_cast 多 -> 布局转换开销
"""
import argparse
import csv
import os
import sys


def find_csv(root, name):
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn == name:
                return os.path.join(dirpath, fn)
    return None


def _pick(header, *cands):
    low = {h.lower().strip(): i for i, h in enumerate(header)}
    for c in cands:
        c = c.lower().strip()
        if c in low:
            return low[c]
    # fuzzy
    for c in cands:
        for h, i in low.items():
            if c in h:
                return i
    return None


def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    header, body = rows[0], rows[1:]
    return header, body


def summarize(path, top):
    header, body = load(path)
    di = _pick(header, "duration(us)", "duration", "total time(us)", "time(us)")
    ti = _pick(header, "task type", "tasktype", "op type")
    ni = _pick(header, "op name", "name", "kernel name")
    if di is None:
        print(f"[kernel_summary] 找不到 Duration 列，header={header}", file=sys.stderr)
        return
    parse = lambda s: float(s) if s and s.strip() else 0.0  # noqa: E731
    by_task, by_kernel, total = {}, {}, 0.0
    for r in body:
        try:
            d = parse(r[di])
        except Exception:  # noqa: BLE001
            continue
        total += d
        if ti is not None and ti < len(r):
            by_task[r[ti]] = by_task.get(r[ti], 0.0) + d
        if ni is not None and ni < len(r):
            by_kernel[r[ni]] = by_kernel.get(r[ni], 0.0) + d

    print(f"文件: {path}")
    print(f"总 kernel 耗时: {total:.3f} us\n")
    print("== 按 Task Type ==")
    print(f"{'Task Type':<24}{'time(us)':>16}{'%':>8}")
    for k, v in sorted(by_task.items(), key=lambda kv: -kv[1]):
        print(f"{k:<24}{v:>16.1f}{(v / total * 100 if total else 0):>8.1f}")

    print(f"\n== Top {top} 热点 kernel（按总耗时）==")
    print(f"{'kernel':<60}{'time(us)':>14}{'%':>8}")
    for k, v in sorted(by_kernel.items(), key=lambda kv: -kv[1])[:top]:
        name = k if len(k) <= 58 else k[:55] + "..."
        print(f"{name:<60}{v:>14.1f}{(v / total * 100 if total else 0):>8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prof_dir")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    path = find_csv(a.prof_dir, "kernel_details.csv")
    if not path:
        print(f"[kernel_summary] 在 {a.prof_dir} 下没找到 kernel_details.csv", file=sys.stderr)
        sys.exit(1)
    summarize(path, a.top)


if __name__ == "__main__":
    main()
