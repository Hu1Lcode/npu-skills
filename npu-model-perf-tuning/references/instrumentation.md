# P2 · 分段打点基线与计时规范

目标：在 P1 拆解表的基础上，量出**开箱性能**——既含整体 E2E（推理）/ s·step⁻¹ + 吞吐（训练），也含每段耗时。

## 铁律：计时点前必须设备同步

NPU/GPU 是异步执行：`torch` 的算子调用只是"下发"，立即读 `time.time()` 量到的是**下发耗时**，不是执行耗时。所以在**每个计时点之前**都要 `synchronize()`：

```python
def _sync():
    try:
        import torch_npu
        torch_npu.npu.synchronize()
    except Exception:
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass
```
（在 `from torch_npu.contrib import transfer_to_npu` 之后，`torch.cuda.synchronize()` 也会被映射到 NPU。）

## 用 stage_timer 打点（不要散改）

`scripts/stage_timer.py` 提供 `stage(name)` 上下文管理器 + `begin_step(step, measure=)`（warmup 步设 `measure=False` 只测不计）+ `summary()`/`print_table()`/`dump()`（`.json` 或 `.csv`）：

```python
from stage_timer import StageTimer

timer = StageTimer()
for i, batch in enumerate(loader):
    if i >= WARMUP + N:
        break
    timer.begin_step(i, measure=(i >= WARMUP))
    with timer.stage("dataloader"):
        batch = next(iter(loader))
    with timer.stage("forward"):
        out = model(batch)
    with timer.stage("backward"):
        out.loss.backward()
    ...
timer.dump("perf_reports/20260101/baseline.json")
print(timer.summary())
```

## 推理口径
1. **warmup**：先跑 K 次（建议 K≥3）不计时，跳过首次编译/内存分配/JIT。
2. **计时**：再跑 N 次（建议 N≥10，或按 benchmark 常规），**每段取中位数**，避免抖动。
3. 输出：每阶段 中位数(ms) + 占比(%)；整体 E2E 中位数；同时给 P50/P95 看抖动。

## 训练口径
- **权威口径是 log**：训练框架通常在日志里打印 `s/step` 或 `iter_time`（单位常为 s/step），吞吐 = `global_batch_size / s_per_step`。优先从 log / `metrics.jsonl` / `summary.json` 直接解析末尾 N 步的稳定值（按行取 `s/step` 字段，或用框架导出的 summary；不要用分段打点之和代替）。
- **分段仅作指示性**：`synchronize` 会打断异步流水重叠，所以逐段打点只对**少数几步**做，用于看"forward/backward/optim/数据各占多少"，**不要把分段之和当成吞吐基线**，吞吐仍以 log 的 s/step 为准。
- 必须注明：是否含梯度累积、是否 DDP/通信、采样的是哪几步。

## baseline.json 建议结构
```json
{
  "mode": "inference|training",
  "device": {"chip": "...", "cards": 8},
  "command": "...",
  "warmup": 3, "measured_steps": 20,
  "e2e_ms_median": 1234.5,
  "stages": [{"name": "denoise", "ms_median": 900.0, "share_pct": 72.9}],
  "train": {"s_per_step": 2.64, "samples_per_s": 48.43, "source": "log"}
}
```

## 陷阱
- `npu-smi` 在高负载下可能很慢甚至卡住 → 加超时，别在计时循环里调用。
- 首个 step 常含首步编译/预热，务必跳过。
- 末 N 步统计，别把 warmup 或抖动步算进去。
- 逻辑 device 序 ↔ 物理卡号映射要确认（`ASCEND_RT_VISIBLE_DEVICES`）。
