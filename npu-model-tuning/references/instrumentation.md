# P2 · 分段打点基线与计时规范

目标：在 P1 拆解表的基础上，量出**调优前基线**。推理含整体 E2E、各阶段耗时和 DiT 单步时间；训练不含 E2E，只有单步时间及单步内拆解，P50/P99/吞吐取自完整训练日志。后续所有对比都以它为零点。

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
- 统计：count / mean / median / min / max / 占比 / 抖动（`(max-min)/median`）；推理报告只报时间与抖动，不报 P50/P99

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

## 基线有效性：3 次任务 + 抖动门禁（先执行）

无论是推理还是训练，基线都必须**独立跑 3 次任务**，并计算 3 轮之间的抖动：

```text
jitter_pct = (max - min) / median × 100%
```

- 判据：`jitter_pct ≤ 10%` 才算有效基线；> 10% 时基线不可用，先排查抢卡、频率/温度、数据缓存、随机性、编译缓存等，再重新采集。
- 工具：`python scripts/baseline_stats.py --baseline perf_reports/<date>/baseline.json --json-out perf_reports/<date>/artifacts/baseline_stats.json` 直接读取 `runs` 并校验；推理会同时校验 `e2e_ms` 与 `dit_step_ms_median`，训练校验 `s_per_step`。也可用 `--values <v1> <v2> <v3>` 快速计算。退出码 0 表示全部有效。
- 推理：用 3 轮的 **E2E 时间**计算抖动。
- 训练：用 3 轮日志中的 **s/step** 计算抖动（训练没有 E2E 任务时间指标）。
- 3 轮必须是相同代码、相同机器/卡、相同 batch/分辨率/步数/dtype/seed、相同环境变量；不能跨时段拼数据。

## 扩散模型计时口径

按 P1 的阶段分别计时，至少包含：

| 阶段 | 计时边界 |
|---|---|
| `prompt_encode` | tokenizer 开始 → text_encoder 输出拿到 |
| `image_encode` | 参考图/源视频预处理开始 → VAE encoder 输出拿到（如无此阶段则跳过） |
| `dit_denoise` | 第一个 denoise step 前 → 最后一个 denoise step 后（**总时间**） |
| `dit_step` | **单个 denoise step** 的耗时；**跳过前 5 个预热 step**，从第 6 步开始逐 step 记录 |
| `vae_decode` | VAE decoder 调用前 → 输出张量拿到 |
| `postprocess` | 后处理/格式转换开始 → 结束（如计入 E2E 则必须明确） |

- **DiT 单步时间**是扩散模型必须单独上报的最小单位：既要知道整个去噪循环的 `dit_denoise` 总时间，也要知道单步 `dit_step` 的时间。单步时间同样要求 3 轮任务间抖动 ≤ 10%。
- 单步时间建议用**独立的 `StageTimer` 实例**采集，避免与 `dit_denoise` 总时间在同一份 summary 里重复计：

```python
dit_timer = StageTimer()
for i in range(num_steps):
    dit_timer.begin_step(i, measure=(i >= 5))   # 前 5 步预热
    with dit_timer.stage("dit_step"):
        latent = transformer(...)
dit_timer.dump(".../artifacts/dit_step.json")
```

- 报告里每个阶段的耗时用**中位数 + 占比**；推理只看时间，不报 P50/P99/吞吐。E2E 计时必须写清楚包含哪些阶段、是否含模型加载/输入读取/MP4 编码等。

## 测量口径（所有场景以本节为准）

| 场景 | 筛选 / 实验 / 采 profiling | 基线采集 | 定稿 / 上报架构师 / 最终报告 |
|---|---|---|---|
| 推理 | warmup 1 轮；denoise steps 可降到 5 步，仅用于筛选与定位 | warmup 1 轮；denoise steps 固定为 **10 步左右**（不必跑完整的 40–50 步）；**独立跑 3 次任务**，E2E 抖动 ≤ 10%；采各阶段时间 + DiT 单步时间（5 步预热后） | 与基线同一步数、同一协议，跑完整 pipeline / 原始启动脚本；3 次任务取中位数，报 E2E + 阶段 + DiT 单步 |
| 训练 | warmup 20 steps；实验测量 10 steps 即可 | 独立跑 3 次任务，用日志 `s/step` 算抖动，要求 ≤ 10%；P50/P99/吞吐必须来自**完整训练任务日志**；单步内阶段用打点少量 step 拆解 | 原始启动脚本完整训练流程，从日志稳定段取 `s/step`、P50、P99、吞吐；阶段拆解仍用打点 |

补充：

1. 短步数实验（推理 5 step / 训练 10 step）只能用于候选筛选、profiling 定位和快速对比，不能作为最终收益上报。
2. 基线的推理步数固定为 10 步左右并写入 `baseline.json`；所有优化点对比必须使用与基线完全相同的步数。若用户要求真实部署步数（如 40–50 步），单独再跑一轮并注明口径。
3. 如果模型有随机采样，固定 seed；如果多次输入不同，固定输入集合（如 3 个样本 × 2 个尺寸）并分别报告，不只报最好样本。
4. **每个优化点复测必须同时记录**：推理为 E2E 时间 + 所有阶段耗时 + DiT 单步时间；训练为日志 `s/step` + 单步内各阶段耗时（P50/P99/吞吐从日志取，不能只报阶段拆解）。
5. **推理没有 P50/P99 指标**；训练 P50/P99 必须来自完整训练任务日志，不能拿打点拆解的少量 step 冒充。

## 训练口径

- **训练没有 E2E 任务时间指标，只有单步时间**；时间拆解也在单步内部进行。默认拆 `dataloader` / `forward` / `backward` / `optimizer` / `zero_grad`；若 step 内实际包含 text_encoder / VAE / DiT 等模型组件，按 P1 的实际调用链继续拆分子阶段。
- **权威口径是日志**：训练框架通常在日志里打印 `s/step`、`iter_time`、P50/P99 和吞吐；吞吐 = `global_batch_size / s_per_step`。优先从 log / `metrics.jsonl` / `summary.json` 解析末尾稳定段。
- **P50/P99 必须来自完整训练任务日志**，不能在少量 step 的打点数据上算，也不能用打点拆解替代。
- **分段仅作指示性**：`synchronize` 会打断异步流水重叠，逐段打点只对少数 step 做，用于看 dataloader/forward/backward/optimizer/zero_grad 各占多少；**不要用分段之和代替 s/step 或吞吐**。
- 必须注明：是否含梯度累积、是否 DDP/通信、采样的是哪几步、batch size 和全局 batch、P50/P99 的统计窗口。

## baseline.json 建议结构

`baseline.json` 必须包含 3 次任务的原始值、中位数和抖动是否有效，以及各阶段明细。`StageTimer` 的 `dump()` 只产出阶段明细，跨任务 E2E（推理）/ s·step（训练）由 A 汇总写入本文件。

推理示例：

```json
{
  "mode": "inference",
  "device": {"chip": "Ascend910", "cards": 1},
  "command": "bash run.sh ...",
  "protocol": {"runs": 3, "jitter_threshold_pct": 10, "denoise_steps": 10, "dit_step_warmup": 5},
  "runs": [
    {"run": 1, "e2e_ms": 1234.5, "dit_step_ms_median": 90.1},
    {"run": 2, "e2e_ms": 1240.2, "dit_step_ms_median": 91.0},
    {"run": 3, "e2e_ms": 1228.7, "dit_step_ms_median": 89.8}
  ],
  "e2e": {"median_ms": 1234.5, "min_ms": 1228.7, "max_ms": 1240.2, "jitter_pct": 0.9, "valid": true},
  "stages": [
    {"name": "prompt_encode", "ms_median": 45.2, "share_pct": 3.7},
    {"name": "image_encode", "ms_median": 120.0, "share_pct": 9.7},
    {"name": "dit_denoise", "ms_median": 900.0, "share_pct": 72.9},
    {"name": "vae_decode", "ms_median": 169.3, "share_pct": 13.7}
  ],
  "dit_step": {"ms_median": 90.1, "jitter_pct": 1.3, "warmup_steps": 5, "valid": true,
               "note": "dit_denoise 的子项，不参与阶段占比求和"}
}
```

> `dit_step` 是 `dit_denoise` 内部的最小计时单位，不能与 `dit_denoise` 同时计入阶段占比总和；需要占比时用 `dit_step / dit_denoise` 或单独说明。

训练示例：

```json
{
  "mode": "training",
  "device": {"chip": "Ascend910", "cards": 8},
  "command": "bash train.sh ...",
  "protocol": {"runs": 3, "jitter_threshold_pct": 10, "warmup_steps": 20, "profile_steps": 10},
  "runs": [
    {"run": 1, "s_per_step": 2.64, "p50_s_per_step": 2.60, "p99_s_per_step": 2.95, "samples_per_s": 48.43},
    {"run": 2, "s_per_step": 2.66, "p50_s_per_step": 2.62, "p99_s_per_step": 3.01, "samples_per_s": 48.12},
    {"run": 3, "s_per_step": 2.63, "p50_s_per_step": 2.59, "p99_s_per_step": 2.98, "samples_per_s": 48.66}
  ],
  "step_time": {"median_s_per_step": 2.64, "jitter_pct": 1.1, "valid": true},
  "log_metrics": {"source": "train.log", "p50_s_per_step": 2.60, "p99_s_per_step": 2.95, "throughput_samples_per_s": 48.43, "full_run": true},
  "stages": [
    {"name": "dataloader", "ms_median": 120.0, "share_pct": 4.5},
    {"name": "forward", "ms_median": 1500.0, "share_pct": 56.8},
    {"name": "backward", "ms_median": 900.0, "share_pct": 34.1},
    {"name": "optimizer", "ms_median": 100.0, "share_pct": 3.8},
    {"name": "zero_grad", "ms_median": 20.0, "share_pct": 0.8}
  ]
}
```

`stages` 的占比是单步内拆分（打点开启），与日志的 `s/step` 口径不同，报告中必须注明。

## 陷阱

- `npu-smi` 在高负载下可能很慢甚至卡住 → 加超时，别在计时循环里调用。
- 首个 step 常含首步编译/预热，务必跳过；DiT 单步至少跳过前 5 个 denoise step。
- 同一组测量要连续完成，避免跨时间段拼数据；基线 3 次任务要连续同环境执行。
- 逻辑 device 序 ↔ 物理卡号映射要确认（`ASCEND_RT_VISIBLE_DEVICES`）。
- 打点开启时会有同步开销；**正式性能数字必须用打点关闭的数据**，阶段耗时对比才用打点开启的数据。
- 训练 P50/P99 只能从完整训练任务的日志/指标文件算；少量 step 的打点结果只用于阶段占比，不能当作 P50/P99。
