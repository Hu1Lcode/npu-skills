#!/usr/bin/env python3
"""校验 3 次任务基线抖动是否满足 <= 阈值。

用法:
    # 直接给 3 轮数值
    python baseline_stats.py --values 1234.5 1240.2 1228.7 --label e2e_ms

    # 从 baseline.json 读取（inference: e2e_ms + dit_step_ms_median；training: s_per_step）
    python baseline_stats.py --baseline perf_reports/20260101/baseline.json

    # 自定义阈值并写 JSON
    python baseline_stats.py --values 2.64 2.66 2.63 --threshold 10 --json-out stats.json

退出码: 0=有效, 1=无效/输入错误。
"""
import argparse
import json
import statistics
import sys


def compute(values, threshold):
    if len(values) < 3:
        raise ValueError("至少需要 3 个测量值（基线要求 3 次独立任务）")
    median = statistics.median(values)
    jitter = ((max(values) - min(values)) / median * 100.0) if median else float("inf")
    return {
        "values": values,
        "count": len(values),
        "median": median,
        "min": min(values),
        "max": max(values),
        "jitter_pct": round(jitter, 3),
        "threshold_pct": threshold,
        "valid": jitter <= threshold,
    }


def from_baseline(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    mode = data.get("mode", "inference")
    runs = data.get("runs", [])
    if mode == "training":
        values = [float(run["s_per_step"]) for run in runs]
        return values, "s_per_step", None
    values = [float(run["e2e_ms"]) for run in runs]
    dit_values = []
    for run in runs:
        if run.get("dit_step_ms_median") is None:
            raise ValueError("推理基线缺少 runs[*].dit_step_ms_median（DiT 单步时间必须采集）")
        dit_values.append(float(run["dit_step_ms_median"]))
    return values, "e2e_ms", (dit_values, "dit_step_ms")


def main():
    parser = argparse.ArgumentParser(description="3 次任务基线抖动校验")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--values", nargs="+", type=float, help="3 轮测量值")
    source.add_argument("--baseline", help="baseline.json 路径")
    parser.add_argument("--label", default="metric", help="指标名，默认 metric")
    parser.add_argument("--threshold", type=float, default=10.0, help="抖动阈值%%，默认 10")
    parser.add_argument("--json-out", default=None, help="结果写入 JSON 文件")
    args = parser.parse_args()

    try:
        secondary = None
        if args.baseline:
            values, label, secondary = from_baseline(args.baseline)
        else:
            values, label = args.values, args.label
        result = compute(values, args.threshold)
        result["label"] = label
        if secondary:
            sec_values, sec_label = secondary
            sec_result = compute(sec_values, args.threshold)
            sec_result["label"] = sec_label
            result["secondary"] = sec_result
        result["all_valid"] = result["valid"] and (
            result["secondary"]["valid"] if "secondary" in result else True
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[baseline_stats] ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    return 0 if result["all_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
