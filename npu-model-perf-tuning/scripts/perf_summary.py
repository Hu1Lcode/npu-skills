#!/usr/bin/env python3
"""算子级汇总：读 op_statistics.csv，按算子聚合耗时，给 Top-N 算子。

用法:
    python perf_summary.py <prof_dir> [--top 20]

与 kernel_summary.py 分工：
    perf_summary  -> 按"算子"聚合（哪个 op 最贵）
    kernel_summary-> 按"Task Type"聚合（落在哪种硬件单元）
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
    for c in cands:
        for h, i in low.items():
            if c in h:
                return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prof_dir")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    path = find_csv(a.prof_dir, "op_statistics.csv")
    if not path:
        print(f"[perf_summary] 在 {a.prof_dir} 下没找到 op_statistics.csv", file=sys.stderr)
        print("提示：op_statistics.csv 只在正确设置 Level1+PipeUtilization 时生成。", file=sys.stderr)
        sys.exit(1)

    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        print("空文件", file=sys.stderr)
        sys.exit(1)
    header, body = rows[0], rows[1:]
    ni = _pick(header, "op name", "op", "name")
    ti = _pick(header, "total time(us)", "total time", "avg time(us)", "time(us)")
    ci = _pick(header, "count", "calls")
    if ni is None or ti is None:
        print(f"[perf_summary] 列识别失败 header={header}", file=sys.stderr)
        sys.exit(1)

    agg = {}
    total = 0.0
    for r in body:
        try:
            t = float(r[ti]) if r[ti].strip() else 0.0
        except Exception:  # noqa: BLE001
            continue
        name = r[ni]
        cnt = 0
        if ci is not None and ci < len(r):
            try:
                cnt = int(float(r[ci]))
            except Exception:  # noqa: BLE001
                cnt = 0
        e = agg.setdefault(name, {"time": 0.0, "count": 0})
        e["time"] += t
        e["count"] += cnt
        total += t

    print(f"文件: {path}")
    print(f"总耗时: {total:.3f} us\n")
    print(f"{'op':<55}{'time(us)':>14}{'%':>8}{'count':>10}")
    for name, e in sorted(agg.items(), key=lambda kv: -kv[1]["time"])[: a.top]:
        nm = name if len(name) <= 53 else name[:50] + "..."
        print(f"{nm:<55}{e['time']:>14.1f}{(e['time'] / total * 100 if total else 0):>8.1f}{e['count']:>10}")


if __name__ == "__main__":
    main()
