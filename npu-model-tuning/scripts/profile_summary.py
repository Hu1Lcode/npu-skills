#!/usr/bin/env python3
"""汇总 torch_npu.profiler 产物：step_trace 分型、Task Type 占比、Top-N 热点 kernel/算子。

用法:
    python profile_summary.py <prof_dir> [--top 20] [--rank 0] [--json out.json]

会递归查找 step_trace_time.csv、kernel_details.csv 与 op_statistics.csv（缺失时回退
op_summary.csv）。多 rank 产物默认分析找到的第一份；可用 --rank 指定 rank。
不同 torch_npu 版本列名可能不同，脚本使用模糊匹配；找不到关键列时给出明确提示。
"""
import argparse
import csv
import json
import os
import re
import sys


def _matches_rank(path, rank):
    """判断路径中的 rank/worker/device 编号是否精确等于 rank（避免 rank_1 命中 rank_10）。"""
    pattern = re.compile(rf"(?:^|[/_.-])(?:worker|rank|device)[_-]?{rank}(?!\d)", re.IGNORECASE)
    return bool(pattern.search(path.replace("\\", "/")))


def find_file(root, name, rank=None):
    candidates = []
    for dirpath, _dirs, files in os.walk(root):
        for filename in files:
            if filename == name or filename.lower() == name.lower():
                candidates.append(os.path.join(dirpath, filename))
    if rank is not None:
        for candidate in candidates:
            if _matches_rank(candidate, rank):
                return candidate
        # 单 rank 单文件时路径可能不含 rank 标识；只有唯一候选时才可安全回退
        return candidates[0] if len(candidates) == 1 else None
    return candidates[0] if candidates else None


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def pick(header, *candidates):
    lowered = {name.lower().strip(): index for index, name in enumerate(header)}
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in lowered:
            return lowered[key]
    for candidate in candidates:
        for name, index in lowered.items():
            if candidate.lower().strip() in name:
                return index
    return None


def summarize_step_trace(path):
    """汇总 step_trace_time.csv 的 Computing / Communication / Free 占比并做瓶颈分型。"""
    header, body = read_rows(path)
    comp_idx = pick(header, "computing")
    comm_idx = pick(header, "communication(not overlapped)", "communication")
    free_idx = pick(header, "free")
    step_idx = pick(header, "step")
    if comp_idx is None or comm_idx is None or free_idx is None:
        print(f"[profile_summary] {path}: 找不到 Computing/Communication/Free 列，header={header}", file=sys.stderr)
        return None
    total_comp = total_comm = total_free = 0.0
    per_step = []
    for row in body:
        def value(idx):
            try:
                raw = row[idx].strip()
                return float(raw) if raw else 0.0
            except (ValueError, IndexError):
                return 0.0
        comp, comm, free = value(comp_idx), value(comm_idx), value(free_idx)
        total_comp += comp
        total_comm += comm
        total_free += free
        if step_idx is not None and step_idx < len(row):
            step = row[step_idx]
        else:
            step = str(len(per_step) + 1)
        total = comp + comm + free
        if total > 0:
            per_step.append({
                "step": step,
                "computing_pct": round(comp / total * 100, 2),
                "communication_pct": round(comm / total * 100, 2),
                "free_pct": round(free / total * 100, 2),
                "total_us": round(total, 1),
            })
    total = total_comp + total_comm + total_free
    if total <= 0:
        return None
    shares = {
        "computing_pct": round(total_comp / total * 100, 2),
        "communication_pct": round(total_comm / total * 100, 2),
        "free_pct": round(total_free / total * 100, 2),
    }
    # 阈值与 references/profiling-analysis.md 一致，按优先级判定
    if shares["free_pct"] > 20:
        classification = "Hostbound"
    elif shares["computing_pct"] > 85:
        classification = "Computing"
    elif shares["communication_pct"] > 10:
        classification = "Communication"
    else:
        classification = "NoClearBottleneck"
    return {
        "file": path,
        "total_us": round(total, 1),
        "total_computing_us": round(total_comp, 1),
        "total_communication_us": round(total_comm, 1),
        "total_free_us": round(total_free, 1),
        **shares,
        "classification": classification,
        "per_step": per_step,
    }


def print_step_trace(result):
    print(f"step_trace_time: {result['file']}")
    print(f"总耗时: {result['total_us']:.1f} us   "
          f"Computing {result['computing_pct']:.1f}% / "
          f"Communication {result['communication_pct']:.1f}% / "
          f"Free {result['free_pct']:.1f}%")
    print(f"瓶颈分型: {result['classification']}")
    if len(result["per_step"]) > 1:
        print("\n== 分 step 占比 ==")
        print(f"{'step':<12}{'Computing%':>12}{'Communication%':>16}{'Free%':>10}{'total(us)':>14}")
        for row in result["per_step"]:
            print(f"{str(row['step']):<12}{row['computing_pct']:>12.1f}"
                  f"{row['communication_pct']:>16.1f}{row['free_pct']:>10.1f}{row['total_us']:>14.1f}")
    print()


def summarize_kernels(path, top):
    header, body = read_rows(path)
    duration_idx = pick(header, "duration(us)", "duration", "total time(us)", "time(us)")
    task_idx = pick(header, "task type", "tasktype", "op type")
    name_idx = pick(header, "kernel name", "op name", "name")
    if duration_idx is None:
        print(f"[profile_summary] {path}: 找不到 Duration 列，header={header}", file=sys.stderr)
        return None
    by_task, by_kernel, total = {}, {}, 0.0
    for row in body:
        try:
            duration = float(row[duration_idx]) if row[duration_idx].strip() else 0.0
        except (ValueError, IndexError):
            continue
        total += duration
        if task_idx is not None and task_idx < len(row):
            by_task[row[task_idx]] = by_task.get(row[task_idx], 0.0) + duration
        if name_idx is not None and name_idx < len(row):
            by_kernel[row[name_idx]] = by_kernel.get(row[name_idx], 0.0) + duration
    return {
        "file": path,
        "total_us": total,
        "by_task": sorted(by_task.items(), key=lambda kv: -kv[1]),
        "top_kernels": sorted(by_kernel.items(), key=lambda kv: -kv[1])[:top],
    }


def summarize_ops(path, top):
    header, body = read_rows(path)
    name_idx = pick(header, "op type", "op name", "name", "operator")
    total_idx = pick(header, "total time(us)", "total time", "duration(us)", "duration", "time(us)")
    count_idx = pick(header, "count", "calls")
    if name_idx is None or total_idx is None:
        print(f"[profile_summary] {path}: 找不到列，header={header}", file=sys.stderr)
        return None
    by_op = {}
    total = 0.0
    for row in body:
        try:
            duration = float(row[total_idx]) if row[total_idx].strip() else 0.0
        except (ValueError, IndexError):
            continue
        name = row[name_idx]
        entry = by_op.setdefault(name, {"total_us": 0.0, "count": 0})
        entry["total_us"] += duration
        if count_idx is not None and count_idx < len(row):
            try:
                entry["count"] += int(float(row[count_idx]))
            except (ValueError, IndexError):
                pass
        total += duration
    return {
        "file": path,
        "total_us": total,
        "top_ops": sorted(by_op.items(), key=lambda kv: -kv[1]["total_us"])[:top],
    }


def print_kernels(result):
    print(f"kernel_details: {result['file']}")
    print(f"总 kernel 耗时: {result['total_us']:.1f} us\n")
    print("== 按 Task Type ==")
    print(f"{'Task Type':<28}{'time(us)':>16}{'%':>8}")
    total = result["total_us"]
    for task, duration in result["by_task"]:
        share = duration / total * 100 if total else 0
        print(f"{task:<28}{duration:>16.1f}{share:>8.1f}")
    print("\n== Top 热点 kernel ==")
    print(f"{'kernel':<64}{'time(us)':>14}{'%':>8}")
    for name, duration in result["top_kernels"]:
        label = name if len(name) <= 62 else name[:59] + "..."
        share = duration / total * 100 if total else 0
        print(f"{label:<64}{duration:>14.1f}{share:>8.1f}")


def print_ops(result):
    print(f"算子聚合: {result['file']}")
    print(f"总算子耗时: {result['total_us']:.1f} us\n")
    print("== Top 热点算子 ==")
    print(f"{'op':<64}{'count':>10}{'time(us)':>14}{'%':>8}")
    total = result["total_us"]
    for name, entry in result["top_ops"]:
        label = name if len(name) <= 62 else name[:59] + "..."
        share = entry["total_us"] / total * 100 if total else 0
        print(f"{label:<64}{entry['count']:>10}{entry['total_us']:>14.1f}{share:>8.1f}")


def main():
    parser = argparse.ArgumentParser(description="torch_npu.profiler 产物汇总")
    parser.add_argument("prof_dir")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--rank", type=int, default=None, help="多 rank 产物中指定要分析的 rank（默认第一份）")
    parser.add_argument("--json", default=None, help="把结构化结果写到 JSON")
    args = parser.parse_args()

    result = {"prof_dir": args.prof_dir}
    if args.rank is not None:
        result["rank"] = args.rank
    step_path = find_file(args.prof_dir, "step_trace_time.csv", args.rank)
    kernel_path = find_file(args.prof_dir, "kernel_details.csv", args.rank)
    op_path = find_file(args.prof_dir, "op_statistics.csv", args.rank)
    op_kind = "op_statistics"
    if not op_path:
        op_path = find_file(args.prof_dir, "op_summary.csv", args.rank)
        op_kind = "op_summary"

    if step_path:
        step_result = summarize_step_trace(step_path)
        if step_result:
            result["step_trace_time"] = step_result
            print_step_trace(step_result)
    else:
        print("[profile_summary] 未找到 step_trace_time.csv（部分 profiler 配置不生成）", file=sys.stderr)

    if kernel_path:
        kernel_result = summarize_kernels(kernel_path, args.top)
        if kernel_result:
            result["kernel_details"] = kernel_result
            print_kernels(kernel_result)
            print()
    else:
        print("[profile_summary] 未找到 kernel_details.csv", file=sys.stderr)

    if op_path:
        op_result = summarize_ops(op_path, args.top)
        if op_result:
            result[op_kind] = op_result
            print_ops(op_result)
    else:
        print("[profile_summary] 未找到 op_statistics.csv / op_summary.csv（部分 profiler 配置不生成）", file=sys.stderr)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"\n[profile_summary] wrote {args.json}")

    if not step_path and not kernel_path and not op_path:
        sys.exit(1)


if __name__ == "__main__":
    main()
