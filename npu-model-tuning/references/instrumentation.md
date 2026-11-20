# P2 · 分段打点基线与计时规范

目标：在 P1 拆解表的基础上，量出**调优前基线**——既含整体 E2E（推理）/ s·step⁻¹ + 吞吐（训练），也含每段耗时。后续所有对比都以它为零点。

## 铁律：计时点前必须设备同步

NPU/GPU 是异步执行：`torch` 的算子调用只是"下发"，立即读 `time.time()` 量到的是**下发耗时**，不是执行耗时。所以在**每个计时点之前和之后**都要同步：

```python
def _sync():
    try:
        import torch
        torch.cuda.synchronize()   # transfer_to_npu 下等价于 NPU 同步
        return
    except Exception:
        pass
    try:
        import torch_npu
        torch_npu.npu.synchronize()
    except Exception:
        pass
```

（如果已启用 `from torch_npu.contrib import transfer_to_npu`，`torch.cuda.synchronize()` 会被映射到 NPU；否则用 `torch_npu.npu.synchronize()`。`scripts/stage_timer.py` 已内置这个逻辑。）

## 用 stage_timer 打点（不要散改）

`scripts/stage_timer.py` 提供：
- `StageTimer(enabled=...)` 或环境变量 `NPU_TUNE_PROBE=1` 控制是否生效，默认关闭
- `stage(name)` 上下文管理器
- `begin_step(step, measure=True)`：warmup 步设 `measure=False` 只测不计
- `summary()` / `print_table()` / `dump(path)`（`.json` 或 `.csv`）
- 统计：count / mean / median / P50 / P95 / min / max / 占比

典型用法：

```python
from stage_timer import StageTimer

timer = StageTimer()                     # 默认读 NPU_TUNE_PROBE，0=关闭
for i, batch in enumerate(loader):
    if i >= WARMUP + N:
        break
    timer.begin_step(i, measure=(i >= WARMUP))
    with timer.stage("dataloader"):
        batch = next(iter(loader))
    with timer.stage("forward"):
        out = model(batch)
    ...
timer.dump("perf_reports/20260101/artifacts/stages.json")
timer.print_table()
```

## 打点代码如何注入（probes.patch）

打点代码只能以独立 patch 的形式存在，绝不进入最终交付的 `all.patch`。推荐流程：

```bash
# 1. 确认工作区干净（在 P0 的调优分支上）
git status --porcelain

# 2. 在 P1 标注的位置临时插入打点（import StageTimer + with timer.stage(...)）
#    可复用的代码片段建议放到 artifacts/probe_snippets.py 备查

# 3. 生成 probes patch（新建文件先 git add -N，否则 diff 不含 untracked）
git add -N path/to/probe_file.py 2>/dev/null || true
git diff > perf_reports/<date>/artifacts/probes.patch

# 4. 还原代码，保持基线 commit 干净
git checkout -- .

# 5. 测量时应用 / 测完还原
git apply perf_reports/<date>/artifacts/probes.patch
NPU_TUNE_PROBE=1 bash run.sh ...
git apply -R perf_reports/<date>/artifacts/probes.patch
```

注意：
- **非 git 仓库**：先按 P6 备份将改动的文件，用 `cp` 保留原文件后在副本上打点；`diff -uN --label a/<rel> --label b/<rel>` 生成 `probes.patch`，测量时用 `patch -p1` 应用、`patch -R -p1` 还原，并在 `changes.log` 记录。
- 如果后续有优化 commit，`probes.patch` 可能因上下文变化失效；每次正式复测前在**当前固化状态**上重新生成一次，或在 `artifacts/` 里按版本号保存（`probes-v1.patch`、`probes-v2.patch`）。
- 最终校验：`grep -n "StageTimer\|NPU_TUNE_PROBE" <all.patch>` 必须无输出。

## 扩散模型计时口径

按 P1 的阶段分别计时，至少包含：

| 阶段 | 计时边界 |
|---|---|
| `prompt_encode` | tokenizer 开始 → text_encoder 输出拿到 |
| `image_encode` | 参考图/源视频预处理开始 → VAE encoder 输出拿到（如无此阶段则跳过） |
| `dit_denoise` | 第一个 denoise step 前 → 最后一个 denoise step 后；同时记录单步平均 = 总时间 / `num_inference_steps` |
| `vae_decode` | VAE decoder 调用前 → 输出张量拿到 |
| `postprocess` | 后处理/格式转换开始 → 结束（如计入 E2E 则必须明确） |

报告里每个阶段的耗时用**中位数 + P50 + P95 + 占比**。E2E 计时必须写清楚包含哪些阶段、是否含模型加载/输入读取/MP4 编码等。

## 测量口径（所有场景以本节为准）

用户已确认两档口径：**筛选/实验可以缩短，定稿必须完整**。

| 阶段 | 筛选 / 实验 / 采 profiling | 定稿 / 上报架构师 / 最终报告 |
|---|---|---|
| 推理 | warmup 1 轮；扩散模型可把 denoise steps 降到 5 步（或少量步）来节省时间，只用于筛选与定位 | warmup 1 轮；使用原始 denoise 步数端到端跑完整 pipeline；连续测量 ≥ 3 轮（按单次成本调整），报中位数/P50/P95 |
| 训练 | warmup 20 steps；实验测量 10 steps 即可 | warmup 20 steps；使用原始启动脚本完整训练流程，按日志稳定段统计 `s/step`；分段打点只做少量 step 的指示性拆分 |

补充：

1. 短步数实验（推理 5 step / 训练 10 step）只能用于候选筛选、profiling 定位和快速对比，不能作为最终收益上报。
2. 任何"优化点真实收益"必须用定稿口径的端到端完整流程重测；报告里必须注明是筛选口径还是定稿口径。
3. 如果模型有随机采样，固定 seed；如果多次输入不同，固定输入集合（如 3 个样本 × 2 个尺寸）并分别报告，不只报最好样本。
4. **每个优化点复测必须同时记录 E2E 时间与所有阶段耗时**（中位数/P50/P95/占比），不能只报 E2E 或只报阶段。

## 训练口径

- **权威口径是日志**：训练框架通常在日志里打印 `s/step` 或 `iter_time`。吞吐 = `global_batch_size / s_per_step`。优先从 log / `metrics.jsonl` / `summary.json` 解析末尾 N 步的稳定值。
- **分段仅作指示性**：`synchronize` 会打断异步流水重叠，逐段打点只对少数 step 做，用于看 forward/backward/optim/数据各占多少；**不要用分段之和代替吞吐基线**，吞吐仍以 log 的 s/step 为准。
- 必须注明：是否含梯度累积、是否 DDP/通信、采样的是哪几步、batch size 和全局 batch。

## baseline.json 建议结构

`baseline.json` 必须同时包含 **E2E 时间** 与 **每一阶段的耗时**；`StageTimer` 的 `dump()` 只产出阶段明细，E2E 由 A 单独测量后与阶段结果一起写入本文件。每个优化点的测试结果与台账也必须保存同口径的 E2E + 阶段耗时。

```json
{
  "mode": "inference|training",
  "device": {"chip": "Ascend910", "cards": 8},
  "command": "bash run.sh ...",
  "protocol": "final",
  "warmup": 1,
  "measured_steps": 10,
  "e2e_ms_median": 1234.5,
  "e2e_ms_p50": 1230.0,
  "e2e_ms_p95": 1290.0,
  "stages": [
    {"name": "prompt_encode", "ms_median": 45.2, "ms_p50": 44.9, "ms_p95": 48.1, "share_pct": 3.7},
    {"name": "image_encode", "ms_median": 120.0, "ms_p50": 119.2, "ms_p95": 126.0, "share_pct": 9.7},
    {"name": "dit_denoise", "ms_median": 900.0, "ms_p50": 897.5, "ms_p95": 945.0, "share_pct": 72.9},
    {"name": "vae_decode", "ms_median": 169.3, "ms_p50": 168.0, "ms_p95": 175.0, "share_pct": 13.7}
  ],
  "train": {"s_per_step": 2.64, "samples_per_s": 48.43, "source": "log"}
}
```

## 陷阱

- `npu-smi` 在高负载下可能很慢甚至卡住 → 加超时，别在计时循环里调用。
- 首个 step 常含首步编译/预热，务必跳过。
- 同一组测量要连续完成，避免跨时间段拼数据。
- 逻辑 device 序 ↔ 物理卡号映射要确认（`ASCEND_RT_VISIBLE_DEVICES`）。
- 打点开启时会有同步开销；**正式性能数字必须用打点关闭的数据**，阶段耗时对比才用打点开启的数据。
