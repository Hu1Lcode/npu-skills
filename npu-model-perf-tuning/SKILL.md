---
name: npu-model-perf-tuning
description: Systematically tune the end-to-end performance of PyTorch models on Huawei Ascend NPU for both training and inference. Trace the call chain from the launch script, decompose the inference pipeline or one training step, capture a per-stage baseline with synchronized timers, apply low-intrusion warm-start optimizations (torch_npu transfer_to_npu, proven environment variables, torch_npu fusion operators), collect and analyze a 3–5 step torch_npu.profiler trace, iterate one optimization at a time with freeze/conflict handling, validate precision, and deliver a structured report comparing out-of-box vs final performance. Use when the user asks to profile, diagnose, or speed up a model on Ascend NPU; reduce step time or latency or improve throughput; find AICPU fallbacks or kernel hotspots; try HCCL/allocator/affinity environment variables or npu_* fusion operators; or produce a performance-tuning report.
---

# NPU Model Perf Tuning (Ascend NPU 模型端到端性能调优)

对 Ascend NPU 上的 PyTorch 模型（训练或推理）做系统化性能调优。目标不是"跑一遍 profiler"，而是**从启动脚本出发，先拆清流水线、量出分段基线，再用低侵入手段逐点优化、每次固化、冲突交人工，最后交付一份可复现的调优报告**。

## 何时使用 / 不适用

**适用**：模型已在 NPU 跑通（精度无问题）后，要降低推理时延/训练 step time、提升吞吐；怀疑 AICPU 回退/热点算子/通信/数据加载瓶颈；想试经验环境变量或 `npu_*` 融合算子；需要一份"开箱 vs 调优后"的正式对比报告。

**不适用**：
- 模型还没在 NPU 上跑通 → 先做「模型适配」（另属 ascend-model 范畴），再回来调优。
- 纯精度问题（结果不对但性能正常）→ 转精度类 skill（`ops-precision-standard`、`model-infer-precision-debug`）。
- 单个 AscendC 算子的极限优化 → 转 `ops-perf-*` 工作流。

## 输入 / 输出契约

**输入**
- 模型代码目录（必须）
- 启动/benchmark 命令：一个 smoke 命令（快速跑通）+ 一个 full 命令（正式测量），需说明 warmup 步数与测量步数、batch/seq/dtype 等
- 可选：已有 baseline（`metrics.jsonl` / `summary.json` / 日志中的 s·step⁻¹）
- 可选：目标指标（如 step time ≤ X ms、吞吐 ≥ Y samples/s）

**输出**（全部写入目标模型的 `perf_reports/<YYYYMMDD>/`，不要写到技能目录里）
- `pipeline_breakdown.md` —— P1 的调用链/流水线拆解表
- `baseline_report.md` + `baseline.json` —— P2 的开箱 E2E + 分段基线
- `prof/` —— **持久化**的 profiling 产物（trace + `kernel_details.csv` + `op_statistics.csv` + `manifest.json`），路径要交给用户查看
- `optimization_ledger.md` + `optimizations/{applied.patch, env.applied.sh}` —— 优化台账与累积固化状态（含「待决策」清单）
- `final_report.md` —— 按 `references/report-template.md` 的固定结构，含开箱 vs 最终对比

## 工作流（9 阶段）

逐步推进；**每进入一个阶段，先读对应的 reference**（渐进披露，见文末文件索引）。

### P0 接入 · 环境 · 口径
1. 确认硬件（芯片型号/卡数）、软件栈版本、复现命令、warmup 与测量步数、精度要求、目标指标。
2. 跑 `scripts/env_check.py --json`：输出 NPU 硬件、CANN/torch/torch_npu 版本与 `transfer_to_npu` 可用性；再用 `git -C <模型目录> rev-parse HEAD` 与 `git -C <模型目录> rev-parse --abbrev-ref HEAD` 记录 commit hash 与分支。一并存为报告 §2 素材。
3. smoke 命令跑通（exit 0）。记录「口径」：warmup 次数、统计步数、精度容差。

### P1 静态分析 · 调用链拆解  → 读 `references/pipeline-analysis.md`
从启动脚本静态追踪到真正执行计算的位置（入口 → 构造 → `__call__`/`train_step` → 各子模块），判定推理还是训练，产出一张 **pipeline 拆解表**：`阶段序号 | 阶段名 | 使用的模型/模块 | 源码 file:line | 输入→输出 | 打点位置`。
- 推理：拆出整条流水线的每一部分以及它用的是哪个模型（如 AIGC：`prepare_data`=text_encoder+VAE 编码 → `denoise`=DiT → `vae_decode`=VAE 解码）。
- 训练：拆出一个 step 的各部分（dataloader→forward→backward→optimizer→zero_grad）。
- 结果写入 `<model>/perf_reports/<日期>/pipeline_breakdown.md`。示例见 `references/pipeline-analysis.md` 与 `examples/pipeline-breakdown-examples.md`。

### P2 分段打点基线  → 读 `references/instrumentation.md`
在 P1 表格的每个阶段边界之前/之后打点：`time.time()`，**每个计时点之前先 `torch.cuda.synchronize()`**（在 `transfer_to_npu` 下等价 NPU 同步；无 CUDA 映射时用 `torch_npu.npu.synchronize()`）。统一用 `scripts/stage_timer.py` 的 `stage()` 上下文管理器包裹，避免散改。
- 推理：K 次 warmup（不计时）后 N 次计时，报每阶段**中位数 + 占比%** 与 **E2E**。
- 训练：s/step 与吞吐以 log 为准（权威口径）；分段只做少量 step 的**指示性**拆分，并注明 `synchronize` 会打断异步重叠。
- 产出 `baseline.json` + `baseline_report.md`。**这就是「开箱性能」**，后续所有对比以它为零点。

### P3 优化阶段启动 · Warm-start  → 读 `references/warmstart-defaults.md`
按"经验先上"的顺序落地低侵入基础手段（**这一步就是优化阶段的开始**）：
1. **torch_npu 接入（地基，最重要）**：入口最前面 `import torch_npu` + `from torch_npu.contrib import transfer_to_npu`（或等价 `TORCH_TRANSFER_TO_NPU=1`）。
2. **环境变量起步集**：用 `assets/env.default.sh` 里的 5 个经验值先上（`HCCL_BUFFSIZE`、`CPU_AFFINITY_CONF`、`TASK_QUEUE_ENABLE`、`MULTI_STREAM_MEMORY_REUSE`、`PYTORCH_NPU_ALLOC_CONF`）。
3. **融合算子首轮**：用 `scripts/scan_fusion_ops.py` 扫描 `torch_npu` 可用的 `npu_*`，按 `references/fusion-op-catalog.md` 的映射把热点原语替换为 `npu_fusion_attention` / `npu_rms_norm` / `npu_rotary_mul` 等。详细替换配方复用 `model-infer-fusion` skill。
> Warm-start 落地后**先复测一次**，量化这一批基础的收益，再决定是否进入 P4 深挖。

### P4 采集 profiling + 分析  → 读 `references/profiling-recipes.md`
- 用 `torch_npu.profiler` 包住**整条流水线**（推理=一次完整流程；训练=一个完整 step）；若多步迭代，**只采 3~5 步 demo**（步数多 → 文件巨大、分析极慢）。
- 配置：`schedule(wait=0, warmup=W, active=3~5, repeat=1)`；`_ExperimentalConfig(profiler_level=Level1, aic_metrics=PipeUtilization)`（否则 `kernel_details.csv` 只有 9 列）。
- **产物必须持久化**到 `perf_reports/<日期>/prof/` 并生成 `manifest.json`；在对话和报告里给出该路径供用户查看，不得随临时目录清理。
- 分析：`scripts/kernel_summary.py` 解析 `kernel_details.csv` → Task Type 占比(AICore/AIV/AICPU/MIX/HCCL/memcpy) + AICPU% + Top-N 热点；与 P2 分段耗时互相印证，锁定瓶颈类型后查 `references/diagnosis-playbook.md`。
- 可参考既有分析 skill：`ops-profiling`、`model-infer-perf-breakdown`、`model-infer-profiling`、`ascendc-performance-review`、`code-performance-advisor`、`ops-simulator`；算子级深挖走 `ops-perf-profile-feature-extractor`。

### P5 优化迭代  → 读 `references/optimization-loop.md`
**优先级：低代码侵入优先**。每次只改一个变量，复测后按三态处理：
- ✅ **有效** → **固化**：写入 `optimizations/applied.patch` + `env.applied.sh`，台账标"已固化"，更新累积基线；**后续优化都在此固化状态之上继续**，不丢已得收益。
- ❌ **无效/精度回归** → 否决并回滚。
- ⚠️ **冲突或高侵入** → 记入台账"待决策"，**汇总交用户决定，不自行取舍**（例：`PYTORCH_ALLOC_CONF` vs `PYTORCH_NPU_ALLOC_CONF` 互斥；融合算子与 dtype/精度冲突；需要大规模改代码的优化只列清单）。

然后回到 P4，在**新的固化状态**上重新 profile，找下一个瓶颈，循环直到达标或无更多低侵入收益。

### P6 安全实施（贯穿 P3/P5 每次改动）
改动前备份原文件；给出可回退的 patch；`python -m py_compile` / import 校验通过再继续。

### P7 复测 + 精度校验
同口径重跑出前后对比表；数值/loss 与开箱一致（在容差内）；若精度回归 → 回滚该优化点并二分定位。

### P8 报告与复现  → 读 `references/report-template.md` 与 `references/reproduce-guide-template.md`
按固定结构产出 `final_report.md`（§见 report-template）：摘要 → 环境与代码信息(含 commit hash) → 模型与任务信息(含调用栈) → 开箱基线 → 优化点汇总表(改动/提升%/原因) → 开箱 vs 最终对比 → 精度结论 → 高侵入候选清单(待决策) → 附录(prof 路径/台账/patch/复现)。

## 关键原则（务必遵守）

- **基线两层**：既要有整体 E2E / s·step⁻¹，也要有每个流水线阶段/step 子部分的耗时。
- **打点必须同步**：计时点前 `torch.cuda.synchronize()`，否则测到的是下发时间不是执行时间。
- **profiler 只采 3~5 步且必须保存**：文件要留给用户查看，不得自动删除。
- **优化一次只改一个变量**，有效即固化，冲突/高侵入交人工。
- **低代码侵入优先**：环境变量、`torch_npu` 接入、融合算子替换、config 级优先；大改代码的先列清单。
- **精度是硬门**：任何优化后都要过精度校验，回归即回滚。
- **文档面向用户**：profiling 路径、优化台账、最终报告都要让用户能直接打开查看。

## 快速开始清单

```
1. 跑 env_check.py --json（环境/版本/transfer_to_npu）+ git rev-parse（commit hash）
2. 按启动脚本画调用链 → pipeline_breakdown.md
3. 各阶段前后 time.time()+synchronize 打点 → baseline_report.md
4. 上 transfer_to_npu + env.default.sh 5 个变量 → 复测
5. torch_npu.profiler 采 3~5 步 → prof/（保存！）→ kernel_summary.py
6. 查 diagnosis-playbook 定位 → 每轮改一个点 → 复测 → 固化/否决/记冲突
7. 精度校验 → final_report.md
```

## 文件索引（渐进披露）

| 文件 | 何时读 |
|---|---|
| `references/pipeline-analysis.md` | P1 拆调用链 / 流水线拆解表模板与三例 |
| `references/instrumentation.md` | P2 打点模板、sync 语义、warmup/中位数、训练口径 |
| `references/warmstart-defaults.md` | P3 torch_npu 接入 / 环境变量 / 融合算子首轮 |
| `references/fusion-op-catalog.md` | P3/P5 选融合算子（族分类 + 映射） |
| `references/profiling-recipes.md` | P4 torch_npu.profiler 采集/导出/保存，msprof/npu-smi 备选 |
| `references/optimization-loop.md` | P5 单变量/固化/侵入性/冲突交人工、台账规范 |
| `references/diagnosis-playbook.md` | P4/P5 从瓶颈证据到修复方案 |
| `references/report-template.md` | P8 最终报告固定结构 |
| `references/reproduce-guide-template.md` | P8 可复现调优指南模板 |
| `scripts/env_check.py` | P0 环境+版本+git hash |
| `scripts/stage_timer.py` | P2 分段打点上下文管理器 |
| `scripts/perf_summary.py` | P4 解析 op_statistics.csv → 按算子聚合 Top-N（谁最贵） |
| `scripts/kernel_summary.py` | P4 解析 kernel_details.csv → Task Type/AICPU%/Top-N |
| `scripts/scan_fusion_ops.py` | P3 扫描 torch_npu 可用 npu_* 融合算子 |
| `assets/env.default.sh` | P3 环境变量起步模板 |
| `assets/optimization-ledger.template.md` | P5 优化台账模板 |
| `examples/openwam-case.md` | 真实案例：AICPU→融合算子 15.79→3.02 s/step |
| `examples/pipeline-breakdown-examples.md` | AIGC/分离/训练 三份拆解表样例 |

## 与其它 skill 的边界
- 融合算子替换细节：`model-infer-fusion`
- profiling 采集/瓶颈分析：`ops-profiling`、`model-infer-profiling`、`model-infer-perf-breakdown`
- 静态性能坏味道：`ascendc-performance-review`；profiling 专家规则：`code-performance-advisor`
- 仿真器：`ops-simulator`；硬件健康：`npu-smi`、`ascend-dmi`；环境：`ascendc-env-check`
- 算子级极限优化：`ops-perf-optimize-workflow` 及 `ops-perf-*`
