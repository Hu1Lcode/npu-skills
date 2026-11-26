#!/usr/bin/env python3
"""对比 baseline / final 两份 torch_npu.profiler 产物，用于 P7.5 最终复核。

用法:
    python profile_compare.py --baseline prof/baseline --final prof/final \
        [--top 20] [--rank 0] [--json profile_compare.json]

脚本复用 profile_summary.py 的解析逻辑，输出 step_trace 占比、Task Type 占比、
Top-N 新增/消失 kernel 与算子，以及结构化 JSON。两份 profile 必须同口径采集，
否则只能做趋势参考。
"""
import argparse
import json
import os
import sys

# 允许从任意目录执行本脚本
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profile_summary import (  # noqa: E402
    find_file,
    summarize_kernels,
    summarize_ops,
    summarize_step_trace,
)


def collect_profile(root, rank, top):
    """收集单份 profile 的 step_trace / kernel / op 汇总。"""
    result = {"dir": root}
    step_path = find_file(root, "step_trace_time.csv", rank)
    kernel_path = find_file(root, "kernel_details.csv", rank)
    op_path = find_file(root, "op_statistics.csv", rank)
    op_kind = "op_statistics"
    if not op_path:
        op_path = find_file(root, "op_summary.csv", rank)
        op_kind = "op_summary"

    if step_path:
        step = summarize_step_trace(step_path)
        if step:
            result["step_trace_time"] = step
    if kernel_path:
        kernels = summarize_kernels(kernel_path, top)
        if kernels:
            result["kernel_details"] = kernels
    if op_path:
        ops = summarize_ops(op_path, top)
        if ops:
            result[op_kind] = ops
    return result


def share_by_task(profile):
    kernel = profile.get("kernel_details")
    if not kernel or not kernel.get("total_us"):
        return {}
    total = kernel["total_us"]
    return {
        task: round(duration / total * 100, 2)
        for task, duration in kernel.get("by_task", [])
    }


def top_kernel_entries(profile, key="top_kernels"):
    kernel = profile.get("kernel_details")
    if not kernel:
        return []
    total = kernel.get("total_us") or 0
    entries = []
    for name, duration in kernel.get(key, []):
        entries.append({
            "name": name,
            "total_us": round(duration, 1),
            "share_pct": round(duration / total * 100, 2) if total else 0.0,
        })
    return entries


def top_op_entries(profile):
    for key in ("op_statistics", "op_summary"):
        ops = profile.get(key)
        if not ops:
            continue
        total = ops.get("total_us") or 0
        entries = []
        for name, entry in ops.get("top_ops", []):
            entries.append({
                "name": name,
                "total_us": round(entry.get("total_us", 0.0), 1),
                "count": entry.get("count", 0),
                "share_pct": round(entry.get("total_us", 0.0) / total * 100, 2) if total else 0.0,
            })
        return entries
    return []


def name_set(entries):
    return {entry["name"] for entry in entries}


def diff_entries(baseline_entries, final_entries):
    baseline_names = name_set(baseline_entries)
    final_names = name_set(final_entries)
    new = [entry for entry in final_entries if entry["name"] not in baseline_names]
    missing = [entry for entry in baseline_entries if entry["name"] not in final_names]
    return new, missing


def fmt(value):
    return "-" if value is None else f"{value:.2f}"


def print_step_table(baseline, final, delta):
    if not baseline and not final:
        return
    print("== step_trace_time ==")
    print(f"{'metric':<28}{'baseline':>12}{'final':>12}{'delta':>12}")
    for key, label in (
        ("computing_pct", "Computing%"),
        ("communication_pct", "Communication%"),
        ("free_pct", "Free%"),
    ):
        left = baseline.get("step_trace_time", {}).get(key) if baseline else None
        right = final.get("step_trace_time", {}).get(key) if final else None
        change = delta.get(key) if delta else None
        print(f"{label:<28}{fmt(left):>12}{fmt(right):>12}{fmt(change):>12}")
    print()


def print_task_table(baseline_shares, final_shares):
    if not baseline_shares and not final_shares:
        return
    tasks = list(dict.fromkeys(list(baseline_shares.keys()) + list(final_shares.keys())))
    print("== Task Type 占比 (%) ==")
    print(f"{'Task Type':<28}{'baseline':>12}{'final':>12}{'delta':>12}")
    for task in tasks:
        left = baseline_shares.get(task)
        right = final_shares.get(task)
        change = None if left is None or right is None else round(right - left, 2)
        print(f"{task:<28}{fmt(left):>12}{fmt(right):>12}{fmt(change):>12}")
    print()


def print_entries(title, entries, limit=20):
    if not entries:
        return
    print(f"== {title} ==")
    print(f"{'name':<64}{'time(us)':>14}{'share%':>10}")
    for entry in entries[:limit]:
        name = entry["name"] if len(entry["name"]) <= 62 else entry["name"][:59] + "..."
        print(f"{name:<64}{entry['total_us']:>14.1f}{entry['share_pct']:>10.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="baseline vs final torch_npu.profiler 对比（P7.5）")
    parser.add_argument("--baseline", required=True, help="prof/baseline 目录")
    parser.add_argument("--final", required=True, help="prof/final 目录")
    parser.add_argument("--top", type=int, default=20, help="每份 profile 取 Top-N（默认 20）")
    parser.add_argument("--rank", type=int, default=None, help="多 rank 产物中指定 rank（默认第一份）")
    parser.add_argument("--json", default=None, help="把结构化对比结果写到 JSON")
    args = parser.parse_args()

    baseline = collect_profile(args.baseline, args.rank, args.top)
    final = collect_profile(args.final, args.rank, args.top)

    baseline_has_data = any((
        baseline.get("step_trace_time"), baseline.get("kernel_details"),
        baseline.get("op_statistics"), baseline.get("op_summary"),
    ))
    final_has_data = any((
        final.get("step_trace_time"), final.get("kernel_details"),
        final.get("op_statistics"), final.get("op_summary"),
    ))
    if not baseline_has_data:
        print(f"[profile_compare] baseline 无可用产物: {args.baseline}", file=sys.stderr)
    if not final_has_data:
        print(f"[profile_compare] final 无可用产物: {args.final}", file=sys.stderr)

    baseline_step = baseline.get("step_trace_time") or {}
    final_step = final.get("step_trace_time") or {}
    delta = {}
    for key in ("computing_pct", "communication_pct", "free_pct"):
        if key in baseline_step and key in final_step:
            delta[key] = round(final_step[key] - baseline_step[key], 2)

    baseline_shares = share_by_task(baseline)
    final_shares = share_by_task(final)

    baseline_kernels = top_kernel_entries(baseline)
    final_kernels = top_kernel_entries(final)
    new_kernels, missing_kernels = diff_entries(baseline_kernels, final_kernels)

    baseline_ops = top_op_entries(baseline)
    final_ops = top_op_entries(final)
    new_ops, missing_ops = diff_entries(baseline_ops, final_ops)

    print(f"baseline: {args.baseline}")
    print(f"final:    {args.final}\n")
    print_step_table(baseline, final, delta)
    print_task_table(baseline_shares, final_shares)
    print_entries("final 中新增/上升的 Top kernel", new_kernels)
    print_entries("baseline Top kernel 中消失的 kernel", missing_kernels)
    print_entries("final 中新增/上升的 Top 算子", new_ops)
    print_entries("baseline Top 算子中消失的算子", missing_ops)

    result = {
        "baseline": baseline,
        "final": final,
        "delta": delta,
        "task_type_share_pct": {
            "baseline": baseline_shares,
            "final": final_shares,
        },
        "new_top_kernels": new_kernels,
        "missing_top_kernels": missing_kernels,
        "new_top_ops": new_ops,
        "missing_top_ops": missing_ops,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"[profile_compare] wrote {args.json}")

    if not baseline_has_data or not final_has_data:
        sys.exit(1)


if __name__ == "__main__":
    main()
