#!/usr/bin/env python3
"""分段打点计时器（P2 基线 / P7 复测）。

默认关闭：只有环境变量 NPU_TUNE_PROBE=1 时才记录；显式传 StageTimer(enabled=True) 可强制开启。

用法:
    from stage_timer import StageTimer
    t = StageTimer()                          # 读 NPU_TUNE_PROBE，默认 0=关闭
    for i, batch in enumerate(loader):
        if i >= WARMUP + N: break
        t.begin_step(i, measure=(i >= WARMUP))
        with t.stage("dataloader"):
            batch = next(iter(loader))
        with t.stage("forward"):
            out = model(batch)
    t.dump("perf_reports/20260101/artifacts/stages.json")
    t.print_table()

计时点前后都会同步设备：
  - 优先 torch.cuda.synchronize()（在 transfer_to_npu 下等价 NPU 同步）
  - 否则退到 torch_npu.npu.synchronize()
不同步会把“下发耗时”误当成“执行耗时”。

单文件自检:
    NPU_TUNE_PROBE=1 python stage_timer.py
"""
import argparse
import csv
import json
import os
import statistics
import time
from contextlib import contextmanager


def _sync():
    try:
        import torch

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            return
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch_npu

        if hasattr(torch_npu, "npu") and torch_npu.npu.is_available():
            torch_npu.npu.synchronize()
    except Exception:  # noqa: BLE001
        pass


def _env_enabled():
    return os.environ.get("NPU_TUNE_PROBE", "0").strip().lower() in ("1", "true", "yes", "on")


def _percentile(values, q):
    """Simple nearest-rank percentile; q in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


class StageTimer:
    def __init__(self, enabled=None):
        self.enabled = _env_enabled() if enabled is None else bool(enabled)
        self.records = []  # list of (step, name, seconds)
        self._step = 0
        self._measure = True
        self._stack = []

    def begin_step(self, step, measure=True):
        """标记当前 step；measure=False 时该 step 的打点只测不计入统计（warmup 用）。"""
        self._step = step
        self._measure = measure

    def set_step(self, step):
        self._step = step

    def start(self, name):
        if not self.enabled:
            self._stack.append(None)
            return
        _sync()
        self._stack.append((name, time.perf_counter()))

    def stop(self):
        if not self._stack:
            return None
        item = self._stack.pop()
        if item is None:
            return None
        name, started = item
        _sync()
        elapsed = time.perf_counter() - started
        if self._measure:
            self.records.append((self._step, name, elapsed))
        return elapsed

    @contextmanager
    def stage(self, name):
        self.start(name)
        try:
            yield self
        finally:
            self.stop()

    def summary(self):
        grouped = {}
        for _step, name, seconds in self.records:
            grouped.setdefault(name, []).append(seconds)
        grand_total = sum(sum(values) for values in grouped.values())
        summary = {}
        for name, values in grouped.items():
            total = sum(values)
            summary[name] = {
                "count": len(values),
                "total_s": total,
                "mean_s": total / len(values) if values else 0.0,
                "median_s": statistics.median(values) if values else 0.0,
                "p50_s": _percentile(values, 50),
                "p95_s": _percentile(values, 95),
                "min_s": min(values) if values else 0.0,
                "max_s": max(values) if values else 0.0,
                "share_pct": (total / grand_total * 100.0) if grand_total else 0.0,
            }
        return summary

    def print_table(self):
        summary = self.summary()
        grand_total = sum(item["total_s"] for item in summary.values())
        print(f"{'stage':<28}{'n':>4}{'median(s)':>12}{'p50(s)':>12}{'p95(s)':>12}{'mean(s)':>12}{'share%':>8}")
        for name, item in sorted(summary.items(), key=lambda kv: -kv[1]["total_s"]):
            print(
                f"{name:<28}{item['count']:>4}{item['median_s']:>12.6f}{item['p50_s']:>12.6f}"
                f"{item['p95_s']:>12.6f}{item['mean_s']:>12.6f}{item['share_pct']:>8.1f}"
            )
        print(f"{'TOTAL':<28}{'':>4}{'':>12}{'':>12}{'':>12}{grand_total:>12.6f}{100.0:>8.1f}")

    def dump(self, path):
        """path 以 .json 结尾写 JSON（records + summary），否则写 CSV。"""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        payload = {
            "enabled": self.enabled,
            "records": [
                {"step": step, "stage": name, "seconds": seconds}
                for step, name, seconds in self.records
            ],
            "summary": self.summary(),
        }
        if path.lower().endswith(".json"):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        else:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["step", "stage", "seconds"])
                writer.writerows(self.records)
        print(f"[stage_timer] wrote {len(self.records)} records -> {path}")


def main():
    parser = argparse.ArgumentParser(description="StageTimer 自检")
    parser.add_argument("--dump", default=None, help="写出 JSON/CSV 路径")
    args = parser.parse_args()
    timer = StageTimer(enabled=True)
    for step in range(3):
        timer.begin_step(step, measure=(step >= 1))
        with timer.stage("fake_stage_a"):
            time.sleep(0.01)
        with timer.stage("fake_stage_b"):
            time.sleep(0.02)
    timer.print_table()
    if args.dump:
        timer.dump(args.dump)
    print("自检完成：脚本可用。接入真实流水线时把 stage 名换成各阶段名即可。")


if __name__ == "__main__":
    main()
