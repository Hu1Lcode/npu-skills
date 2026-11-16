# P4 · Profiling 采集与分析配方

目标：拿到"时间到底花在哪种 task 上"的证据（AICore/AIV/MIX/AICPU/HCCL/memcpy），与 P2 分段耗时互相印证，锁定瓶颈类型。

## 用 torch_npu.profiler 包住整条流水线

推理包住**一次完整推理流程**；训练包住**一个完整 step**。多步迭代时**只采 3~5 步 demo**——步数一多，trace 文件会到几个 GB、解析极慢。

```python
import torch, torch_npu
from torch_npu.profiler import profile, ProfilerActivity, _ExperimentalConfig, schedule

exp = _ExperimentalConfig(
    profiler_level=6,          # Level1 的等价枚举值因版本而异，优先用字符串/常量；见下
    aic_metrics=1,             # PipeUtilization 的等价枚举值
)
# 更稳妥：用官方推荐写法
exp = _ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                          aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    schedule=schedule(wait=0, warmup=2, active=3, repeat=1),  # 采 3 步
    experimental_config=exp,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./perf_reports/20260101/prof"),
) as prof:
    for i in range(WARMUP + 5):
        run_one_step()          # 推理：run_pipeline()；训练：train_step()
        prof.step()
```

> **关键**：`profiler_level` 不设到 Level1、`aic_metrics` 不设 PipeUtilization 时，`kernel_details.csv` 只有 9 列、`op_statistics.csv` 不生成——一定要带上 `_ExperimentalConfig`。
> 枚举常量在不同 torch_npu 版本命名不同：先 `python -c "import torch_npu.profiler as p; print(dir(p))"` 确认，或直接传已知整数值。

## 产物结构与持久化（硬要求）
```
perf_reports/<日期>/prof/
├── <hostname>_<pid>_<ts>_ascend_pt/      # profiler 原始目录
│   ├── ASCEND_PROFILER_OUTPUT/
│   │   ├── kernel_details.csv            # 逐 kernel：Name/Task Type/Duration(us)/...
│   │   ├── op_statistics.csv             # 按算子聚合
│   │   └── ...
│   └── ...
└── manifest.json                         # 见下
```
`manifest.json` 至少记：采集命令、warmup/active 步数、profiler_level/aic_metrics、torch/torch_npu/CANN 版本、模型与 batch/seq/dtype、采集时间。**prof/ 目录不得随临时目录清理，路径要交给用户查看。**

## 分析：先看 Task Type 占比
用 `scripts/kernel_summary.py <prof 目录> --top 20`：
- **AICPU%** 高 → 有算子没落到 AI Core（典型：自定义 python/numpy 逻辑、float64、不支持的算子），优先找融合算子或改写（见 `diagnosis-playbook.md` "AICPU 主导"）。
- **AICore/AIV/MIX%** 高但整体仍慢 → 看是否 memory-bound（带宽）、同步等待、或核利用率低（切分/tiling）。
- **HCCL%** 高 → 通信瓶颈（多卡）：查并行策略、通信/计算重叠、`HCCL_BUFFSIZE`。
- **memcpy/format_cast 多** → 数据布局转换开销，考虑 `npu_format_cast`/消除多余 cast。
- **Top-N 热点 kernel** 与 P2 分段对应，确认哪个阶段/模型最重。

## 备选/补充采集工具
| 工具 | 用途 |
|---|---|
| `torch_npu.profiler` | 本 skill 默认，Task Type + kernel 级 |
| `msprof`（`ops-profiling` skill） | 更细的 kernel/pipe 级、指令级 |
| `npu-smi` / `ascend-dmi` | 硬件健康、功耗/温度/利用率粗看 |
| `npu-compute` | 指标集采集、pipe 流水 |
| `ops-simulator` | 无硬件时的性能仿真 |
| `ops-perf-profile-feature-extractor` | 算子级诊断特征（pipe 图、逐指令时序） |

## 陷阱
- trace/JSON 过大导致解析失败或截断 → 降到 3 步、只采必要 activity。
- 首步含编译/预热 → 用 `wait=0,warmup=2` 跳过。
- 采集本身有开销 → 只用于定位，不用它当最终性能数字（最终数字回到 P2/P7 口径）。
- DDP/多进程时只在一个 rank 采、或用 `ASCEND_PROFILER` 相关开关避免所有 rank 同时写爆盘。
