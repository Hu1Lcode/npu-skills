# Profiling 采集配方（P2 基线 / P4 热点 / P7.5 最终复核）

目标：拿到"时间到底花在哪种 task 上"的证据（AICore / AIV / MIX / AICPU / HCCL / memcpy），与 P2 分段耗时互相印证，锁定瓶颈类型；同时保证最终 profile 能与基线同口径对照。

> 采集完成后的**深度分析**（step_trace_time 分型、算子透视、Roofline、Hostbound、慢卡检测）见 `profiling-analysis.md`。本文件只负责"怎么采、怎么保证数据可用"。

## 0. 采集红线与脚本改造方式

来自社区 `pytorch-profiling-collection` 的实践要求：

- **禁止修改用户的原始脚本**。基于原脚本创建带时间戳的副本（如 `train_profiling_20261119_120000.py`），放在原脚本同级目录；原脚本保持不动。
- **`.py` 脚本**：复制后在副本中注入 profiler 初始化与 `prof.start()/stop()`。`prof.start()` 放在训练/推理循环**外面**，`prof.step()` 放在每步末尾，`prof.stop()` 放在循环外面。
- **`.sh` 脚本**：先复制 `.sh` 为 `*_profiling_<timestamp>.sh`，从 `python` / `torchrun` / `deepspeed` 命令或 `--module` 参数中提取真正执行的 Python 入口；复制该 `.py` 并注入 profiler 代码；修改副本 `.sh` 中的入口路径指向新 `.py`。原环境变量、分布式启动参数、其余参数保持不变。
- **采样步数**：`active` 默认 3（1–5 按需），不要采几百步——产物会到 GB 级且解析极慢。
- **默认只采主 rank（如 rank 0）**：单卡/单进程场景可用 `ASCEND_RT_VISIBLE_DEVICES` 限定单卡；多进程 DDP/FSDP 只在主 rank 打开 profiler，**不要**把全局 `ASCEND_RT_VISIBLE_DEVICES` 限制为单卡（否则其他 rank 不可见）。多 rank 同时采集会让 trace 体积、磁盘占用和解析时间成倍放大。仅当怀疑快慢卡或通信异常等特殊情况，**由用户确认后**再补采全 rank。
- **采集后验证产物**：确认输出目录下存在 `*.pt.json` 和 `worker_0_*/trace_*.json`（多进程按 rank 分别检查）；文件为空或缺失视为采集失败，必须先修复再分析。
- **采集期间关闭 P2 打点**（`NPU_TUNE_PROBE=0`）：`synchronize()` 会制造空泡，污染 Free 占比。

### 基线 / 最终 profile 对照（P2 / P7.5）

- `prof/baseline/`：P2 计时基线完成后、P3 O2 环境变量组**之前**采集；原始代码状态，`NPU_TUNE_PROBE=0`。它是最终对照的固定基线，后续 P4 的 profile 不能替代它。
- `prof/warmstart/`：P3 环境变量组之后、P4 热点分析用的 profile（可选）；它包含 P3 优化，不能当作原始基线。
- `prof/final/`：P7.5 在 P7 的干净集成分支/工作区（`baseline_commit` + 全部已采纳 patch，内容与最终 `all.patch` 等价）、加载最终环境变量（`artifacts/env.applied.sh`，若有）后采集；`NPU_TUNE_PROBE=0`。
- baseline 与 final 必须使用**相同**的 profiler 配置（Level1 + PipeUtilization）、主 rank、warmup、active/denoise steps、输入 shape/dtype/batch；否则只能做趋势参考，不能得出“优化生效”结论。
- 每份 profile 的 `manifest.json` 必须记录 `purpose`（baseline / warmstart / final）、命令、环境变量、warmup/active 步数、profiler 配置、版本、模型配置、rank、输入规格和时间。
- profiling run 只用于机制证据，**不作为最终性能数字**。

### 采集级别选择

| 级别 | 内容 | 开销 | 适用 |
|---|---|---|---|
| `Level0` | 基础算子耗时，无 AICore 细节 | 最低 | 快速看 top 算子 |
| `Level1` | 含 AICore 利用率 / pipe 指标 | 中等 | **默认推荐**（本技能 P4 标准） |
| `Level2` | 全量采集（含更细的 runtime / 指令信息） | 最高 | 需要定位 host 侧或极端细节时，且只采 1–2 步 |

## 1. torch_npu.profiler 包住完整流水线

- **推理**：包住一次完整推理流程；采 1–3 次完整流程即可；扩散模型 profiling 可把 denoise steps 降到 5 步节省时间。
- **训练**：包住一个完整 step；采 3–5 个 step demo（实验性 profiling 可到 10 step 级别）。
- 步数一多，trace 会到几个 GB，解析极慢。

```python
import torch
import torch_npu
from torch_npu.profiler import profile, ProfilerActivity, _ExperimentalConfig, schedule

exp = _ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    schedule=schedule(wait=0, warmup=2, active=3, repeat=1),
    experimental_config=exp,
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./perf_reports/20260101/prof/warmstart"),
) as prof:
    for i in range(WARMUP + ACTIVE):
        run_one_step()      # 推理：run_pipeline()；训练：train_step()
        prof.step()
```

**关键**：
- `profiler_level=Level1`、`aic_metrics=PipeUtilization` 不设时，`kernel_details.csv` 可能只有 9 列、`op_statistics.csv` 不生成。
- 枚举常量在不同 torch_npu 版本命名不同：先 `python -c "import torch_npu.profiler as p; print(dir(p))"` 确认，或直接传已知整数值。
- `schedule` 的 `warmup`/`active` 按**循环 step** 计，不是自动按“轮”；推理用 warmup 1 轮、active 1–3 轮，训练用 warmup 20 steps 后在稳定段采集 3–5 steps（示例 `warmup=2, active=3` 需按实际循环替换）。
- `with_stack=True` 默认关闭（会增加开销、放大算子时间）；需要确认调用链时再开，且只采 1 步。

## 2. 产物结构与持久化（硬要求）

```text
perf_reports/<日期>/prof/
├── baseline/                     # P2 基线（最终对照用）
│   ├── <hostname>_<pid>_<ts>_ascend_pt/
│   │   ├── ASCEND_PROFILER_OUTPUT/
│   │   │   ├── kernel_details.csv      # 逐 kernel：Name / Task Type / Duration(us) / ...
│   │   │   ├── op_statistics.csv      # 按算子聚合
│   │   │   └── ...
│   │   └── ...
│   └── manifest.json
├── warmstart/                    # P3 后、P4 分析用（可选）
│   └── ...
├── final/                        # P7.5 最终全栈
│   └── ...
└── manifest.json                 # 如需记录本轮所有 profile 的索引
```

每份 profile 自己的 `manifest.json` 记录：purpose、采集命令、warmup/active 步数、profiler 配置、版本、模型与 batch/seq/dtype、rank、时间。`prof/` 目录不得随临时目录清理，报告里给出路径供用户查看。

## 3. npu-smi 遥测采样

`scripts/npu_monitor.py`：
- 每 1–2 秒轮询 `npu-smi info`，输出 CSV：`timestamp,index,memory_used_mib,memory_total_mib,utilization_pct,power_w,temperature_c`
- `index` 为 torch_npu 逻辑设备号，会按 `ASCEND_RT_VISIBLE_DEVICES` 做物理→逻辑重映射
- 可在 benchmark 运行期间后台启动，`--duration` 控制时长

用于判断：AICore 是否长期低利用、HBM 是否打满、功耗/温度是否异常、卡间是否不均衡。

## 4. 分析：先看 Task Type 占比

`scripts/profile_summary.py <prof_dir> --top 20`：
- **AICPU% 高**（典型 30%–80%）→ 算子没落到 AI Core。优先查 float64、numpy/Python 逻辑、未实现算子，再找融合算子（见 diagnosis-playbook A）
- **AICore / AIV / MIX% 高但整体仍慢** → 可能 memory-bound、核利用率低、切分/tiling 问题
- **HCCL% 高** → 通信瓶颈（多卡）：查并行策略、通信/计算重叠、`HCCL_BUFFSIZE`
- **memcpy / SDMA / PCIe / format_cast 多** → 数据布局转换或搬运开销，考虑减少 `.to()` / `.contiguous()` / transpose，统一 layout（NZ/ND），`npu_format_cast` 一次到位
- **Top-N 热点 kernel / 算子** → 与 P2 分段对应，确认哪个阶段/模型最重

## 5. torch_npu trace schema 与 CUDA 的差异（分析 trace JSON 时注意）

| 差异点 | CUDA trace | torch_npu 2.x trace |
|---|---|---|
| 顶层结构 | `{"traceEvents": [...]}` | 顶层直接是事件列表 |
| `ts` / `dur` | 数字 | 字符串，需转 float |
| 设备任务 category | `kernel`、`gpu_memcpy`、`gpu_memset` | 大量设备任务 `cat` 为 `None`，需读 `args["Task Type"]` |
| 计算 task type | — | `KERNEL_AIVEC`、`KERNEL_AICORE`、`KERNEL_AICPU`、`KERNEL_MIX_AIV`、`KERNEL_MIX_AIC` |
| 内存/搬运 | `gpu_memcpy` | `SDMA_SQE`、`PCIE_DMA_SQE`、`WRITE_VALUE_SQE`、`MEMCPY` |
| 集合通信 | `nccl_*` | `hcom_*`（HCCL） |
| 运行时事件 | `cuda_runtime`、`cuda_driver` | `privateuse1_runtime`、`HostToDevice`、`async_npu`、`async_task_queue`、`dequeue`、`enqueue`、`fwdbwd` |

如果自己解析 trace：先归一化 `ts`/`dur` 字符串，再按 `args["Task Type"]` 分类设备任务；`hcom_*` 归为通信；`HostToDevice` 归为 H2D 搬运。

## 6. 备选/补充工具

| 工具 | 用途 |
|---|---|
| `torch_npu.profiler` | 本技能默认，Task Type + kernel 级 |
| `msprof` | 更细的 kernel/pipe 级、指令级 |
| `npu-smi` / `ascend-dmi` | 硬件健康、功耗/温度/利用率粗看 |
| `npu-compute` | 指标集采集、pipe 流水 |
| 算子级工具 | 单个 AscendC 算子极限优化时使用 |

## 7. 陷阱

- trace 过大导致解析失败或截断 → 降到 1–3 步、只采必要 activity。
- 首步含编译/预热 → `warmup=2` 跳过。
- 采集本身有开销 → 只用于定位，不用它当最终性能数字；最终数字回到 P2/P7 口径。
- DDP/多进程时默认只在一个 rank（主 rank，如 rank 0）采，避免所有 rank 同时写爆盘；需要全 rank 时必须先经用户确认。
- 多卡 profiler 产物可能每个 rank 一份；默认只分析主 rank 那一份，全 rank 对比仅用于慢卡/通信异常专项。`profile_summary.py --rank N` 可指定要分析的 rank 产物。
