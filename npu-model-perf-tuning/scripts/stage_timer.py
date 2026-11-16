#!/usr/bin/env python3
"""分段打点计时器（P2 基线）。

用法（脚本内）:
    from stage_timer import StageTimer
    t = StageTimer()
    for i, batch in enumerate(loader):
        if i >= WARMUP + N: break
        t.begin_step(i, measure=(i >= WARMUP))
        with t.stage("dataloader"):
            batch = next(iter(loader))
        with t.stage("forward"):
            out = model(batch)
    t.dump("perf_reports/20260101/baseline.json")   # .json 或 .csv 自动识别
    t.print_table()

用法（命令行自检）:
    python stage_timer.py

关键：每个 stage 的边界都会先 synchronize（NPU 上 torch_npu.npu.synchronize，
回退 torch.cuda.synchronize），否则 NPU 异步执行，time.time() 测到的是
"下发耗时"而不是真实执行耗时。
"""
import time
import csv
import json
from contextlib import contextmanager


def _sync():
    try:
        import torch_npu
        if hasattr(torch_npu, "npu") and torch_npu.npu.is_available():
            torch_npu.npu.synchronize()
            return
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


class StageTimer:
    def __init__(self):
        self.records = []  # (step, name, seconds)
        self._step = 0
        self._measure = True
        self._name = None
        self._t0 = None

    def set_step(self, s):
        self._step = s

    def begin_step(self, step, measure=True):
        """标记当前 step；measure=False 时该 step 的打点只测不计入统计（warmup 用）。"""
        self._step = step
        self._measure = measure

    def start(self, name):
        _sync()
        self._name = name
        self._t0 = time.time()

    def stop(self):
        _sync()
        if self._name is None:
            return None
        dt = time.time() - self._t0
        if self._measure:
            self.records.append((self._step, self._name, dt))
        self._name = None
        self._t0 = None
        return dt

    @contextmanager
    def stage(self, name):
        self.start(name)
        try:
            yield self
        finally:
            self.stop()

    def summary(self):
        agg = {}
        for _step, name, dt in self.records:
            a = agg.setdefault(name, {"count": 0, "total": 0.0, "min": float("inf"), "max": 0.0})
            a["count"] += 1
            a["total"] += dt
            a["min"] = min(a["min"], dt)
            a["max"] = max(a["max"], dt)
        for a in agg.values():
            a["mean"] = a["total"] / a["count"] if a["count"] else 0.0
        return agg

    def print_table(self):
        agg = self.summary()
        grand = sum(a["total"] for a in agg.values())
        print(f"{'stage':<24}{'n':>4}{'mean(s)':>12}{'total(s)':>12}{'%':>8}")
        for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["total"]):
            pct = (a["total"] / grand * 100) if grand else 0
            print(f"{name:<24}{a['count']:>4}{a['mean']:>12.4f}{a['total']:>12.4f}{pct:>8.1f}")
        print(f"{'TOTAL':<24}{'':>4}{'':>12}{grand:>12.4f}{100:>8.1f}")

    def dump(self, path):
        """path 以 .json 结尾写 JSON（含 records + summary），否则写 CSV。"""
        if path.lower().endswith(".json"):
            data = {
                "records": [{"step": s, "stage": n, "seconds": d} for s, n, d in self.records],
                "summary": self.summary(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["step", "stage", "seconds"])
                w.writerows(self.records)
        print(f"[stage_timer] wrote {len(self.records)} records -> {path}")


if __name__ == "__main__":
    t = StageTimer()
    for step in range(3):
        t.begin_step(step, measure=(step >= 1))
        with t.stage("fake_stage_a"):
            time.sleep(0.01)
        with t.stage("fake_stage_b"):
            time.sleep(0.02)
    t.print_table()
    print("自检完成：脚本可用。接入真实流水线时把 stage 名换成各阶段名即可。")
