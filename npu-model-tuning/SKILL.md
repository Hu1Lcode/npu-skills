---
name: npu-model-tuning
description: >-
  Systematically profile and tune the end-to-end performance of DiT-based generative models and world models (WM, e.g. cosmos-framework) on Huawei Ascend NPU, for both inference and training. Use this skill whenever the user provides a model code directory plus a launch script (or exact launch command) and asks to optimize performance — lower inference latency, improve throughput, shorten training step time, diagnose NPU bottlenecks, analyze AICPU/HCCL/memory issues, replace operators with torch_npu fusion ops, or produce a performance tuning report. Covers video/image generation pipelines (text encoder + DiT + VAE) and cosmos-like world models. Do not use for VLA / robot policy models, pure precision debugging, single-operator development, or deep adaptation of models that do not yet run on NPU. Also trigger on 中文 keywords: 优化一下性能、加速、跑得慢、昇腾、NPU、性能调优、推理加速、训练吞吐、step time、算子融合、profiling、打点、显存优化、AICPU、HCCL、DiT、world model、WM、cosmos。
---

# NPU Model Tuning（Ascend NPU DiT / World Model 性能调优）

对 Ascend NPU 上的 DiT 类生成模型与 cosmos 类 world model（WM）做系统化性能调优，覆盖训练与推理。目标不是"跑一遍 profiler"，而是：**从启动脚本出发，拆清调用链，量出分段基线，先用无侵入手段（环境变量 / 融合算子）逐点优化，再把侵入性改动按预期收益排序交用户决策，每个优化点单独测性能与精度，最终交付可直接应用的 patch 和一份可复现的报告。**

本技能**默认以多 Agent 模式运行**（架构师 + A 调用链/基线 + B Profiling/优化/实施 + C 独立测试），详见下方“多 Agent 协作模式（默认开启）”。

## 触发条件

当用户同时给出以下信息时，触发本技能：

1. **一个代码目录**（模型代码 / 项目仓库）；
2. **一个启动脚本或明确启动命令**（训练或推理）；
3. **性能优化意图**：如“优化一下性能”“跑得太慢”“降低时延”“提升吞吐”“缩短 step time”“做一下 profiling 调优”。

当前版本的模型范围：

- ✅ **DiT 类生成模型**：视频/图像生成，典型链路 `text_encoder` → `DiT denoise` → `VAE decode`，包括 diffusers 实现与自研实现（含 VAE encode / 参考图编码等前置阶段）。
- ✅ **cosmos 类 World Model（WM）**：cosmos-framework 等世界模型的推理与训练性能优化。
- ❌ **VLA / 机器人策略模型**：当前版本不覆盖。
- ❌ 纯精度排查（结果不对但性能正常）、单个 AscendC 算子开发、模型尚未在 NPU 上跑通的深度适配。

如果只有代码目录、没有启动脚本，先按 P0 要求用户补充启动方式；不要凭目录结构猜执行路径。

## 何时使用 / 不适用

**适用**：
- 模型已在 Ascend NPU 上跑通，需要降低推理时延 / 训练 step time / 提升吞吐
- 怀疑 AICPU 回退、热点算子、内存搬运、HCCL 通信、数据加载瓶颈
- 想系统尝试环境变量、`torch_npu` 融合算子等无侵入优化
- 需要一份"调优前 vs 调优后"的正式报告和可回放 patch

**不适用**：
- VLA / 机器人策略模型 → 当前版本不覆盖
- 纯精度问题（结果不对但性能正常）→ 先做精度排查，再回来调优
- 单个 AscendC 算子的极限优化 → 走算子级优化流程
- 模型完全未在 NPU 上跑通 → 只做 P0 的最小跑通检查；深度适配不在本技能范围

## 输入 / 输出契约

### 输入
| 输入 | 必需 | 说明 |
|---|---|---|
| 代码目录 | 是 | 模型代码 / 项目仓库 |
| 启动脚本 | 是 | 训练或推理的原始启动命令或 `.sh` 脚本，是调用链分析和基线的起点 |
| 权重 / 数据 / 输入样本 | 视脚本 | 优先从启动脚本解析；缺失时向用户确认 |
| 调优约束（可选） | 否 | 目标指标、精度红线、禁用手法、可改动范围 |

### 输出
所有产物写入目标代码目录下的 `perf_reports/<YYYYMMDD>/`（不要写进技能目录）：

```text
perf_reports/<YYYYMMDD>/
├── final_report.md              # 正式交付报告（固定 8 节结构，见 references/report-template.md）
├── patches/
│   ├── 001-<优化点名>.patch     # 每个已固化优化点一个 patch（串行叠加）
│   ├── 002-<优化点名>.patch
│   ├── all.patch                # 全部优化点的合并 diff
│   ├── index.json               # 优化点编号、commit、patch 文件映射
│   └── stack_verify.json        # P7 集成验证结果（patch 顺序、可应用性、smoke/full）
├── artifacts/                   # 原始测量数据、probe patch、日志、精度对比结果
│   ├── env_info.json            # P0 环境自检结果（报告第 1 节来源）
│   ├── model_info.json          # P0 记录的模型信息（报告第 2 节来源）
│   ├── fusion_ops.json          # P3/P5 torch_npu 融合算子扫描结果（含签名/说明）
│   ├── changes.log              # 非 git 仓库：逐个优化点的改动与回退记录
│   ├── applied.patch            # 运行态备份（可选）
│   ├── env.applied.sh           # 已固化环境变量（可选，可 source）
│   ├── probes.patch             # P2 打点补丁（默认关闭，不进入最终 patch）
│   ├── baseline_stats.json      # P2 三次基线抖动校验结果（可选）
│   ├── profile_compare.json     # P7.5 baseline vs final 结构化对比（profile_compare.py 输出）
│   ├── precision/               # P7 精度对比结果（cosine / PSNR / SSIM / 逐层 dump）
│   ├── tests/                   # 多 Agent 单点测试：<opt_id>/result.json 与原始日志
│   └── backups/                 # 非 git 仓库：每个优化点的文件级备份与回退清单
├── backup_src/                  # 非 git 仓库：P0 完整代码备份（最终兜底）
├── prof/
│   ├── baseline/                # P2 基线 profile（P3 环境变量组之前，probe 关闭，最终对照用）
│   ├── warmstart/               # P3 环境变量后的 profile（P4 热点分析用，可选）
│   └── final/                   # P7.5 最终全栈 profile（probe 关闭）
├── profiling_analysis.md        # P4 profiling 分析结论与候选方向
├── final_profiling_analysis.md  # P7.5 baseline vs final profile 对比
├── optimization_verification.md # P7.5 每个已采纳优化点的生效矩阵
├── pipeline_breakdown.md        # P1 调用链、模型定义映射与打点位置说明
├── baseline.json                # P2 基线数据
├── optimization_ledger.md       # 每个优化点的实验台账
├── invasive_candidates.md       # 侵入性优化候选清单（按预期收益降序，待用户决策）
└── agent/                       # 多 Agent 默认模式下的计划、状态与交接记录
```

**路径约定**：本文件与 references 中出现的 `scripts/...`、`assets/...`、`references/...` 均指**本技能目录**下的文件，执行时使用技能目录的绝对路径（或先 `cd` 到技能目录）；`perf_reports/...` 则指**目标代码目录**下的产物。不要把 `scripts/` 当成目标仓库里的目录。

**patch 生成方式**：
- git 仓库：P0 从当前分支新建 `npu-tuning/<YYYYMMDD>` 分支；每个被采纳的优化点提交一次 commit（消息前缀 `opt(N):`）；P8 用 `scripts/make_patches.py` 按 commit 生成独立 patch 和 `all.patch`。
- 非 git 仓库：P0 完整备份原代码到 `perf_reports/<date>/backup_src/`，同时维护 `artifacts/changes.log` 记录每次改动；P8 用 `diff -uN` 生成 patch（新增文件也能覆盖）。
- 最终 patch **不包含**计时打点代码（见 P2）。

## 多 Agent 协作模式（默认开启）

本技能**默认以多 Agent 模式运行**：技能触发后，主 Agent 先担任总体架构师，创建 `perf_reports/<date>/agent/plan.md` 和 `state.json`，再按角色卡启动 A/B/C。只有以下情况才降级为单 Agent，并在 `plan.md` 记录原因：

- 用户明确要求单 Agent；
- 运行环境不支持子代理/`Agent` 工具；
- 任务只是单点验证（如只跑一次基线），且用户同意简化。

**主 Agent 担任总体架构师**，负责定目标与验收标准、派活、验收、维护状态台账和写最终报告；三个子 Agent 分工：

| 角色 | 职责 | 关键产出 |
|---|---|---|
| **架构师**（主 Agent） | 项目管理 + 验收 + 用户接口 + 最终报告；P7.5 硬门禁验收；不是单纯转发 | `agent/state.json`、`optimization_ledger.md`、`final_report.md`、`patches/` |
| **小弟 A** | P1 调用链拆解 + P2 分段打点基线 + P2 末 `prof/baseline/` | `pipeline_breakdown.md`、`baseline.json`、`prof/baseline/`、`artifacts/probes.patch` |
| **小弟 B** | P3/P4 Profiling 分析 + P5 优化点分析**与无侵入实现（环境变量/融合算子）** + P7.5 final 分析 | `profiling_analysis.md`、`optimization_candidates.json`、候选 patch、`final_profiling_analysis.md`、`optimization_verification.md` |
| **小弟 C** | 独立单点测试：同轮 control vs candidate，性能 + 精度；P7.5 `prof/final/` 采集 | `artifacts/tests/<opt_id>/result.json`、`prof/final/`、原始日志 |

用 `Agent` 工具（general-purpose 子代理）启动 A/B/C，prompt 按 `references/multi-agent-workflow.md` 的角色卡组织。交接必须 **artifact-first**：架构师派活时传“关键摘要 + 文件路径 + 输出 schema”，不能只靠自然语言转述。完整角色卡、Context Pack 模板、`state.json` 结构、并行/隔离规则见 `references/multi-agent-workflow.md`。

> 注意：B 如果只分析不实现，必须增加“小弟 D · 优化实施”，否则没人写 patch。默认 B 兼任实施。

## 经验库与自学习（默认开启）

每次调优结束都要沉淀经验；每次新任务开始都要先召回经验。经验库是**候选假设库**，不是结论库——命中后仍要用当前模型的 profiling 和测量结果验证。

**三类经验**（存放在 `experience/`，可用 `NPU_TUNE_EXPERIENCE_DIR` 指向外部目录）：

| 文件 | 内容 |
|---|---|
| `experience/optimization-methods.jsonl` | 常用有效的优化手段（现象、动作、收益、精度 caveat、适用模型/框架） |
| `experience/device-issues.jsonl` | 设备/驱动/CANN/torch_npu 的常见问题与对策 |
| `experience/code-reading.jsonl` | 找入口、追调用链、识别框架结构的经验 |

**P0 经验召回（必须）**：

```bash
python scripts/experience.py search \
  --query "<模型类型/症状关键词>" \
  --model-type <DiT/VAE/world-model/...> --framework <diffusers/cosmos-framework/...> \
  --json perf_reports/<date>/agent/experience_recall.json
```

把命中的经验摘要写进给 A/B 的 Context Pack；A 重点看 `code-reading`，B 重点看 `optimization-methods` 和 `device-issues`。候选优化点必须标注 `experience_id`（如果来自经验库），并在台账记录最终是否有效。

**P8 经验沉淀（必须）**：

```bash
python scripts/experience.py add --category optimization \
  --title "..." --symptom "..." --action "..." --confidence observed \
  --tags "fusion,AICPU" --model-types "DiT,VAE" --frameworks "diffusers" \
  --evidence '{"gain":"4.1%","precision":"cosine 0.9999"}' \
  --source-run perf_reports/<date>

python scripts/experience.py update --id <经验id> --result success|failed --note "..."
```

**质量红线**：
- 只有实测过、有 `source_run` 和性能/精度数据的经验才能标 `verified`；推测标 `hypothesis`，观察但未成对测量标 `observed`。
- 必须写清适用模型/框架/硬件/软件版本；不能写成“所有模型都有效”。
- 先去重再新增：先 `search`，相似条目用 `update` 补充证据。
- 不记录权重路径、客户数据、密钥或私有代码片段。

## 工作流（9 阶段）

每进入一个阶段，先读对应 reference（渐进披露，见文末文件索引）。

### P0 接入 · 环境 · 口径

1. **默认多 Agent 启动**：架构师先执行 `mkdir -p perf_reports/<date>/{agent,artifacts,patches,prof}`，再写 `agent/plan.md`、`agent/state.json`；确认 `Agent` 工具可用，否则按降级模式执行并在 `plan.md` 记录原因。
2. 确认：训练还是推理、模型类型、启动脚本、目标指标（时延 / 吞吐 / step time）、精度约束、可改动范围。同时按“经验库与自学习”执行 P0 经验召回，写入 `agent/experience_recall.json`，随任务包传给 A/B。
3. **记录模型信息**到 `artifacts/model_info.json`（模板见 `assets/model-info-template.json`），报告第 2 节直接引用。至少包含：`model_name`、`model_type`（DiT / VAE / world-model / ...）、`structure`（text_encoder → DiT → VAE 等）、`params`（参数量及统计口径）、`dtype`、`input_spec`（分辨率 / 帧数 / 序列长度 / batch）、`parallel_strategy`、`framework`、`code_version`（branch + commit / 备份标识）。从启动脚本、模型 config 与用户确认中获取；拿不到的字段写 `"unknown"` 并注明待确认，不要编造。
4. 记录代码状态：git 仓库记录 `git rev-parse HEAD`、当前分支、是否有未提交改动，并创建调优分支；非 git 仓库按 P6 做完整备份并写 `artifacts/changes.log`。
5. 跑 `python scripts/env_check.py --repo <repo> --json > perf_reports/<date>/artifacts/env_info.json`：硬件、CANN、驱动、torch、torch_npu、Python、框架/依赖版本、设备数、可见卡映射、关键环境变量、目标仓库的 git 信息。结果直接作为报告第 1 节素材。
6. **最小跑通检查**：确认 `torch_npu` 可导入、`torch.npu.is_available()` 为真、设备可分配；若模型尚未接入 NPU，只做最小修正（入口导入 `torch_npu`、设备映射、HCCL 初始化），不展开深度适配。
7. 确认并记录两类命令：**smoke 命令**（快速跑通，如推理 1 次、训练 2–3 step）和 **full 命令**（正式测量）。从启动脚本/参数中解析，缺失时向用户确认。跑通 smoke（exit 0），记录口径：warmup 次数、测量步数、batch/seq/dtype、精度容差。

### P1 静态分析 · 调用链拆解  → 读 `references/pipeline-analysis.md`

1. 读启动脚本：环境变量、`python -m` / `torchrun` 入口、参数、真正执行的 Python 文件。
2. **先找到 pipeline**：推理通常是 `DiffusionPipeline` / `WanPipeline` / 自研 pipeline 的 `__call__` / `forward` / `generate`；训练是训练循环里的 `train_step` 或完整 step 函数。记录 pipeline 的类/函数名、定义位置 `file:line`、启动脚本中的调用点。
3. **再找 pipeline 对每个模型的调用**：按执行顺序列出 `text_encoder` / `image_encoder` / `DiT` / `VAE` / 条件编码器等；每个组件记录构造/加载点（`from_pretrained` / `build_model` / `load_checkpoint`）和 pipeline 内的调用点 `file:line`。
4. **进入模型定义定位算子**：对每个模型组件记录模型类/函数定义 `file:line`、forward 入口 `file:line`，并继续向内定位融合算子替换涉及的关键实现：Attention、RMSNorm/LayerNorm、RoPE、FFN/激活、Add+Norm、VAE 卷积/逐元素等。模型来自外部包无法读源码时，记录包名/版本与 `external:<package>`，不能编造 `file:line`。
5. 判定推理还是训练，切分可观测阶段，记录每阶段的 `file:line`、输入输出、使用的模型/模块。
6. 产出 **Mermaid 流程图**并标注**计划插入计时打点的位置**（`T1`、`T2`…），同时输出到 `pipeline_breakdown.md`；其中必须包含 **模型定义与调用映射表**（组件 → 模型类 → 定义 file:line → 构造点 → pipeline 调用点 → forward 入口 → 关键算子/融合候选）。
7. 扩散模型默认拆法：
   - `prompt_encode`：tokenizer + `text_encoder` 文本/prompt 编码
   - `image_encode`：VAE encoder / 参考图或源视频编码（如有）
   - `dit_denoise`：DiT 去噪循环（标注总耗时 + DiT 单步时间，单步在 5 步预热后采集）
   - `vae_decode`：VAE 解码
   - 可选：`preprocess`、`postprocess`
8. 训练模型默认拆法：`dataloader`（含 H2D 拷贝）→ `forward` → `backward` → `optimizer` → `zero_grad`；多卡时标注含 HCCL 的阶段。

**为什么必须找到模型定义**：融合算子替换发生在模型实现内部（Attention / Norm / RoPE / FFN / VAE 等），P1 不定位到模型类和 forward，P5 就无法把 profiling 热点映射成可执行的替换点，也无法判断舍入误差会影响哪些层和哪个阶段。

### P2 分段打点基线  → 读 `references/instrumentation.md`

1. 用 `scripts/stage_timer.py` 的 `stage()` 上下文管理器在 P1 标注的位置打点，不要散改代码。
2. **每个计时点开始前和结束后都同步**：`torch.cuda.synchronize()`（启用 `transfer_to_npu` 时等价 NPU 同步）；若不可用则用 `torch_npu.npu.synchronize()`。不同步会把"下发耗时"误当成"执行耗时"。
3. 打点默认关闭：通过独立 `artifacts/probes.patch` 注入，并检查环境变量 `NPU_TUNE_PROBE=1` 才生效。测量阶段按需 `git apply`（非 git 用 `patch -p1`），测完反向还原；最终 `all.patch` 不包含 probes。
4. **扩散模型至少拆三个大段计时**：prompt 编码与图像编码（两者可各自单独计时）→ DiT 去噪 → VAE decode；如果有额外编码器（音频/视觉条件）再单独加阶段。每段都要有明确的计时边界。
5. **基线有效性门禁（推理/训练通用）**：基线必须独立跑 **3 次任务**；按 `抖动 = (max - min) / median × 100%` 计算 3 轮差异，要求 **≤ 10%**（可用 `python scripts/baseline_stats.py --baseline perf_reports/<date>/baseline.json` 校验，脚本会同时校验推理 E2E + DiT 单步 / 训练 `s/step`，退出码非 0 即无效）。超限说明机器噪声、抢卡、降频等异常，基线不可用，必须排查后重新采集。推理看 E2E 时间；训练看日志 `s/step`（训练没有 E2E 时间指标）。
6. **推理基线口径**：采集阶段 `prompt_encode`（text_encoder）、`image_encode`（VAE encoder，如有）、`dit_denoise` 总时间、`vae_decode`；另外必须单独记录 **DiT 单步时间**——去噪是多步迭代，需要既有总时间也有最小单位，单步时间在**预热 5 个 denoise step 后**逐 step 采集，同样要求抖动 ≤ 10%。采集基线时 denoise steps 固定为 **10 步左右**即可（不必跑完整的 40–50 步），基线与后续所有对比必须使用同一步数。
7. **训练基线口径**：训练**不报 E2E 任务时间**，只有单步时间；时间拆解也在单步内部完成，默认阶段为 `dataloader` / `forward` / `backward` / `optimizer` / `zero_grad`，若 step 内实际还包含 text_encoder / VAE / DiT 等模型组件，按 P1 的实际调用链继续拆分子阶段。`s/step`、**P50/P99** 和吞吐从**完整训练任务的日志**读取，不需要打点；打点只用于单步内各阶段拆解。
8. 产出 `baseline.json` + 基线表：
   - 推理：3 轮 E2E 时间与抖动、各阶段时间/占比、DiT 单步时间/抖动（5 步预热后）；**推理只有时间指标，不报 P50/P99/吞吐**。
   - 训练：3 轮日志 `s/step` 与抖动、完整训练任务的 P50/P99/吞吐、单步内各阶段时间/占比；不报 E2E。
   每个优化点的复测结果也要按同样结构记录。
9. **基线 Profiling（最终对照用，必须）**：P2 计时基线完成后、P3 应用 O2 环境变量组之前，用与 P7.5 完全相同的 profiler 配置、主 rank、warmup、active/denoise steps 和输入，采集 `prof/baseline/` 并写 `manifest.json`；采集时 `NPU_TUNE_PROBE=0`。这份 profile 只作为 P7.5 的对照证据，不作为性能数字。P3 O1 若只是最小跑通接入，可视为基线状态的一部分，但要在 manifest 说明。

### P3 Warm-start · 基础接入与环境变量  → 读 `references/warmstart-defaults.md`

1. **O1 · torch_npu 接入检查**：若入口尚未导入 `torch_npu` / `transfer_to_npu`，在入口最前面补上（基础接入）。已接入则跳过并记录。
2. **O2 · 环境变量组**：按场景从 `assets/env.default.sh` 的候选集中选择相关变量，作为一个优化点成组应用、成组测量。不要无脑全开；逐项确认当前 CANN/torch_npu 版本支持。应用前必须确认 P2 的 `prof/baseline/` 已采集；O2 是第一个性能优化点，P7.5 的 `prof/final/` 需要与这份基线对照。
3. 运行 `python scripts/scan_fusion_ops.py --detail --json perf_reports/<date>/artifacts/fusion_ops.json`，建立当前机器实际可用的 `npu_*` 算子清单（含 torch_npu 版本、分组、函数签名与首行说明），供 P4/P5 使用。扫描结果只是候选池，不直接全量替换。
4. 应用后先按 P2 同口径复测一次，记录本组的性能收益（作为一个优化点）。
5. **融合算子替换不在 P3 盲目执行**：必须等 P4 profiling 定位到热点、并基于 P1 的模型定义映射找到对应 `file:line` 后，再逐个替换、逐个测量性能和精度。融合算子通常能减少下发和访存，但收益大小取决于热点占比；同时可能因 dtype 降级、累加顺序或近似激活引入舍入误差，精度不通过时先尝试精度保持变体，仍不通过就回滚，不能默认全量启用。

### P4 采集 Profiling + 分析  → 读 `references/profiling-recipes.md`、`references/profiling-analysis.md`

1. `torch_npu.profiler` 包住完整流水线：推理默认只采主 rank（如 rank 0）；扩散模型 profiling 可把 denoise steps 降到 5 步节省时间；训练采 3–5 个 step（实验可用 10 step 级别）。多 rank 仅在有快慢卡/通信异常等特殊情况时，由用户确认后补采全 rank。步数过多会产生 GB 级 trace、解析极慢。P4 的 profile 保存到 `prof/warmstart/`（若已应用 P3 环境变量）或 `prof/round-<N>/`，它是热点分析用，**不是原始基线**；最终对照统一使用 P2 的 `prof/baseline/`。
2. 配置 `ProfilerActivity.CPU + NPU`；按版本设置 `_ExperimentalConfig(profiler_level=Level1, aic_metrics=PipeUtilization)`，否则 `kernel_details.csv` 可能缺少关键列。
3. 产物必须持久化到 `perf_reports/<date>/prof/warmstart/` 或 `prof/round-<N>/` 并生成 `manifest.json`（命令、步数、版本、模型配置、purpose），路径写进报告。
4. 用 `scripts/npu_monitor.py` 在关键运行期间采样 `npu-smi`（HBM / AICore% / 功耗 / 温度），与 profiler 互证。
5. 用 `scripts/profile_summary.py` 分析（默认取找到的第一份产物；多 rank 可用 `--rank N` 指定）：
   - `step_trace_time.csv` 的 Computing / Communication / Free 占比与瓶颈分型
   - Task Type 占比：AICore / AIV / AICPU / MIX / HCCL / memcpy
   - Top-N 热点算子 / kernel
   - 与 P2 分段耗时对照，判断瓶颈属于：计算、搬运、通信、调度还是数据加载
6. 按 `references/profiling-analysis.md` 做分型与下钻：`step_trace_time.csv` 的 Computing / Communication / Free 占比 → 瓶颈类型 → 对应深入分析（算子透视 / Roofline / Hostbound / 慢卡检测）。
7. 按 `references/diagnosis-playbook.md` 把证据映射为候选优化方向：先分"无侵入（环境变量/融合算子）/ 有侵入"两组，有侵入组按预期收益从高到低排序，作为 P5 输入。

### P5 优化迭代 · 先无侵入，后有侵入清单  → 读 `references/optimization-loop.md`、`references/inference-optimization-playbook.md`、`references/training-optimization-playbook.md`

优化按两阶段推进，**先穷尽无侵入项，再列侵入性清单**：

**P5a 无侵入优化（自动实施、逐点固化）** —— 只做两类：

- **环境变量组**：P3 基础组 + 按 profiling 证据追加的组，成组应用、成组测量；
- **融合算子 / 算子后端替换**：先 `python scripts/scan_fusion_ops.py --detail --json perf_reports/<date>/artifacts/fusion_ops.json` 扫描本机 `torch_npu` 实际导出接口（含签名/说明；不同版本有增删，不能照抄目录），再对照 `references/fusion-op-catalog.md`、P4 热点和 P1 的模型定义映射（定位到具体 `file:line`）逐个替换；mindiesd 等框架自带的融合接口、attention backend 切换同属此类。融合算子一般能减少下发/访存，但收益取决于热点占比；可能引入舍入误差（fp16/bf16 输入、累加顺序、近似激活），必须逐个测性能 + 精度，不能默认全量启用。

1. 从**当前固化状态**出发，一次只应用一个优化点（一个环境变量组或一个算子替换）。推理场景优先按 `references/inference-optimization-playbook.md` 的路线图与"症状 → 候选"表选择；训练场景按 `references/training-optimization-playbook.md` 的路线图与并行/显存/通信候选选择。
2. 每个替换必须可开关、可回退，保留原实现；优先以独立补丁/包装层/monkey patch 落地，尽量不改模型主体结构。先做 `py_compile` / import 校验再测量。
3. 按 P2/P7 同口径复测性能 + 精度：推理记录 **E2E + 各阶段 + DiT 单步（5 步预热后）**；训练记录 **日志 `s/step` + P50/P99 + 吞吐 + 单步内阶段拆解**；三态处理：有效且精度通过 → 固化（git commit；非 git 更新备份与 `changes.log`）；无效/负收益/精度回归 → 回滚；冲突 → 待决策。
4. 回到 P4 重新 profile，继续下一个无侵入项，直到达标或无更多无侵入收益。

**P5b 侵入性改动清单（只列不改，交用户拍板）** —— 无侵入项穷尽后：

5. 把环境变量与融合算子以外的改动全部列为候选（局部代码改写、模块/接口重构、并行范式调整、图模式、量化、Attention Cache / 稀疏注意力、多流 / 权重预取、VAE patch parallel 等），**按预期性能收益从高到低排序**，按 `assets/invasive-candidates-template.md` 写入独立文件 `invasive_candidates.md`（同时同步到 `optimization_ledger.md` 与报告第 6 节）。
6. 每项写清：优先级、优化点、P4 证据、预期收益及依据、侵入程度、改动位置（`file:line`）、工作量、风险/副作用、依赖关系、回退方式。等用户选择后，再按 P5a 的单变量流程逐项实施。

### P6 安全实施（贯穿 P3/P5）

- **git 仓库**：从 P0 commit 新建 `npu-tuning/<date>` 分支；每个固化优化点一个 commit（`opt(N): ...`），回退用 `git revert` / `git reset --hard <上一状态>`。
- **非 git 仓库：每个优化点都必须可独立回退**（用户已确认）：
  1. P0 先做完整备份 `perf_reports/<date>/backup_src/`，作为最终兜底。备份目录位于目标仓库内的 `perf_reports/` 下，**必须排除 `perf_reports/` 自身**，否则会递归拷贝输出目录：`rsync -a --exclude 'perf_reports' <repo>/ perf_reports/<date>/backup_src/`（无 rsync 时用 `tar --exclude='./perf_reports' -cf - . | tar -xf - -C <backup_dir>`）。
  2. 每个优化点动手前，把将被修改的文件按原相对路径复制到 `artifacts/backups/<opt_id>/`，并在 `artifacts/backups/backup_manifest.json` 记录：`opt_id`、原路径、备份路径、`sha256`、备份时间、将改动的范围。
  3. 改动后立即生成该点 diff（用 `a/<相对路径>`、`b/<相对路径>` 标签，保证 `patch -p1` 可应用；`-N` 让新增文件也能生成创建 diff）：`diff -uN --label a/<rel> --label b/<rel> artifacts/backups/<opt_id>/<rel> <repo>/<rel> > patches/NNN-<name>.patch`；新增文件在 `backup_manifest.json` 记 `change_type: add`；并把时间、优化点、文件列表、改动原因、回退命令追加到 `artifacts/changes.log`。
  4. 回退时优先 `patch -R -p1 < patches/NNN-*.patch`，或 `cp -a artifacts/backups/<opt_id>/. <repo>/`；回退后重新核对 `sha256`，确认与备份一致。
  5. 验证未通过/被否决的优化点也必须保留备份与记录，便于复盘。
- 优先非侵入式：新增独立文件/包装层/配置开关，尽量不改原文件结构；必须改时保持接口不变。
- 每次改动后跑 `python -m py_compile` 或 import 校验；再进入测量。
- 任何新开关默认值应为"保持原行为"。

### P7 复测 + 精度校验 + 最终 Profiling 复核  → 读 `references/precision-validation.md`、`references/profiling-recipes.md`、`references/profiling-analysis.md`

1. **集成验证（必须）**：把已采纳的优化点按顺序应用到从 `baseline_commit` 新建的集成分支/工作区；每个 patch 先 `git apply --check`，有冲突退回 B 基于当前栈重做 patch。可用 `scripts/verify_stack.py --repo <repo> --base <commit> --patches <dir>` 在临时 worktree 自动验证，并可带 `--command` 跑 smoke/full。**非 git 仓库**：把 `backup_src/` 复制到临时目录，按顺序 `patch -p1` 应用 `patches/NNN-*.patch`，再跑 smoke/full，结果同样记录到 `stack_verify.json`。逐点集成复测：control = 当前栈（首次为原始基线），candidate = 当前栈 + 本优化点，记录 `standalone_gain`（并行筛选值）与 `stacked_gain`（集成增量）。
2. 所有已固化优化叠加后，按 P2 的**同一基线口径**复测最终性能：推理独立跑 3 次、固定 10 步左右，报 E2E 时间 + 各阶段时间 + DiT 单步时间（5 步预热后），不报 P50/P99；训练从完整训练日志取 `s/step`、P50、P99、吞吐，单步内各阶段耗时另用打点拆解。权威性能数字用**打点关闭**的数据；阶段对比用**打点开启**的数据，并注明口径差异。
3. 精度校验（每个优化点都要有记录）：
   - 默认：与基线输出做**输出级对比**——主判据统一为 `cosine ≥ 0.999`；同时记录 `max_abs_diff`、`mean_abs_diff`、相对 L2 用于定位差异来源（不作为第二门槛）；图像/视频另算 PSNR/SSIM 并保存关键帧对照。
   - 若 `cosine < 0.999` 或肉眼可见差异：用 forward hook 做**逐层 dump 对比**，定位首个异常层。
   - 训练：固定数据批次，主判据用固定 batch 前向输出的 `cosine ≥ 0.999`；同时记录若干步 loss / gradient norm 作为辅助监控，持续偏离或发散 → 回滚。
4. 精度回归 → 回滚对应优化点，并在台账记录。
5. 回归检查：关闭所有新开关后，结果应与原始基线一致。
6. 若使用分卡并行筛选：并行结果只能作为候选筛选，**最终报告数字必须来自串行集成后的全栈复测**；测试前用 `scripts/check_cards.py` 确认卡空闲，同一候选的 control/candidate 必须在同一卡上成对跑。

**P7.5 最终全栈 Profiling 复核（P8 前硬门禁）**

P7.5 只证明“优化机制在最终代码中真实生效、瓶颈发生迁移”，不重新分配单项收益；单项收益仍以 P7 的 control vs candidate / `stacked_gain` 为准。

1. **干净路径**：从 `baseline_commit` 新建干净集成分支/工作区，按顺序应用所有已采纳 patch（即 `all.patch`）并加载最终环境变量（`artifacts/env.applied.sh`，若有），确认 `NPU_TUNE_PROBE=0`、无探针/调试代码、无未提交改动。最终 profile 必须来自这条最终交付代码路径，不能来自 B 的开发工作区。
2. **同口径采集**：按 `prof/baseline/manifest.json` 的 profiler 配置（Level1 + PipeUtilization）、主 rank、warmup、active steps/denoise steps、输入形状/dtype/batch，采集 `prof/final/` 并写自己的 `manifest.json`（同时记录最终环境变量）。带 profiler 的 run **不作为最终性能数字**；最终性能数字仍来自第 2 步的非 profiling 复测。
3. **对比分析**：用 `scripts/profile_summary.py` 分别汇总 baseline/final，或用 `scripts/profile_compare.py --baseline prof/baseline --final prof/final`，生成 `final_profiling_analysis.md`：step_trace 的 Computing / Communication / Free 占比、Task Type 占比、Top-N kernel/算子、AICPU/CPU fallback、HCCL/memcpy 与关键算子数量变化。
4. **填写生效矩阵**：按 `assets/optimization-verification-template.md` 写 `optimization_verification.md`。每个已采纳优化点占一行，记录预期 profiling 签名、baseline 证据、final 证据、逐点性能收益、状态（`verified` / `mechanism_verified` / `time_only` / `failed`）和备注。
5. **判定门禁**：`failed` 的优化点不得写入“有效优化”，必须回查 patch 是否真正命中、是否被 fallback、是否有环境变量未生效，确认无效后回滚；`time_only` 必须附 P7 同卡成对性能证据（profile 上看不出机制时，不能伪造签名）。所有复核结论同步到 `optimization_ledger.md` 和报告第 7 节。

### P8 报告 · Patch · 复现  → 读 `references/report-template.md`

1. 生成 patch：git 仓库运行 `scripts/make_patches.py --repo <dir> --base <P0 commit> --output perf_reports/<date>/patches`；脚本按 `opt(N):` commit 生成 `NNN-<名称>.patch`，并生成 `all.patch`。非 git 仓库按 P6 的每点 `diff -uN` 产物收集为 `patches/NNN-<名称>.patch`，`all.patch` 用 P0 完整备份与最终代码的 `diff -ruN` 生成（排除 `perf_reports/`）。生成前确认所有 probe/打点改动已还原，最终代码与 patch 中不得残留 `StageTimer` / `NPU_TUNE_PROBE`。
2. 写 `final_report.md`，严格按固定 8 节顺序：
   1. 环境信息
   2. 模型信息
   3. 调优前基线
   4. 阶段打点耗时
   5. 调用链分析（pipeline 定义 → 各模型定义/调用映射 + Mermaid 流程图，标注打点位置）
   6. 优化点逐一说明（现象 / 为什么做 / 改动 / 性能收益（推理 E2E + 各阶段 + DiT 单步；训练 s/step + P50/P99 + 吞吐 + 单步内阶段）/ 精度影响 / **Profiling 生效证据** / 风险与回退）
   7. 优化后性能（含 P7.5 最终全栈 Profiling 复核：baseline vs final 总览 + 优化点生效矩阵摘要）
   8. 优化后 vs 基线对比
3. **经验沉淀（默认必做）**：从 `optimization_ledger.md` 和最终报告提取新经验，用 `scripts/experience.py add` 写入对应类别；本次复用过的经验用 `experience.py update --result success|failed` 回写结果。只写有实测证据、可复用、已脱敏的条目。
4. 附产物索引：patch 路径、`prof/baseline/`、`prof/final/`、`profiling_analysis.md`、`final_profiling_analysis.md`、`optimization_verification.md`、baseline.json、artifacts/profile_compare.json、artifacts/env_info.json、artifacts/model_info.json、artifacts/fusion_ops.json、optimization_ledger.md、invasive_candidates.md、复现命令。
5. 说明分支位置 / 备份位置，以及如何回退所有改动。

## 交付前自检

- [ ] `perf_reports/<date>/` 下 `final_report.md`、`patches/`、`baseline.json`、`prof/`、`profiling_analysis.md`、`final_profiling_analysis.md`、`optimization_verification.md`、`optimization_ledger.md`、`invasive_candidates.md`、`artifacts/env_info.json`、`artifacts/model_info.json`、`artifacts/fusion_ops.json` 齐全
- [ ] 报告严格按 8 节顺序，所有性能数字都有基线 vs 优化后的成对数据与口径
- [ ] `prof/baseline/` 与 `prof/final/` 均存在，两个 `manifest.json` 的 profiler 配置、rank、warmup、active/denoise steps 完全一致，且 profiling run 未用作性能数字
- [ ] `final_profiling_analysis.md` 有 baseline vs final 的 step_trace、Task Type、Top-N kernel/算子对比
- [ ] `optimization_verification.md` 覆盖全部已采纳优化点，无 `failed`；`time_only` 均附 P7 同卡成对性能证据
- [ ] `baseline.json` 含 3 次任务原始值，推理 E2E / 训练日志 `s/step` 抖动 ≤ 10%（见 `artifacts/baseline_stats.json`）
- [ ] 非 git 仓库：`backup_src/`、`artifacts/backups/`、`artifacts/changes.log` 齐全，每个优化点都能单独回退
- [ ] 每个优化点有独立 patch、独立性能收益、独立精度结论、独立 Profiling 复核状态
- [ ] 侵入性优化清单已按预期收益降序排列，未实施项明确标注"待用户决策"
- [ ] `all.patch` 不含 `StageTimer` / `NPU_TUNE_PROBE` / 打点代码
- [ ] `pipeline_breakdown.md` 的 Mermaid 图标注了 T1/T2… 打点位置
- [ ] `pipeline_breakdown.md` 含**模型定义与调用映射表**，融合算子/侵入性候选能回指到模型实现的 `file:line`
- [ ] 复现命令、分支/备份位置、回退方式已写明

## 关键原则

- **先测量，后改动**：任何优化前必须有基线和 profiling 证据。
- **基线三次有效**：推理/训练都独立跑 3 次任务，推理看 E2E 时间、训练看日志 `s/step`，抖动 ≤ 10% 才算有效基线；超限必须重采。
- **推理基线**：E2E 时间 + 各阶段耗时 + **DiT 单步时间（5 步预热后）**；denoise steps 固定 10 步左右，且只有时间指标，不报 P50/P99/吞吐。
- **训练基线**：只有单步时间，不报 E2E；阶段拆解在单步内完成；P50/P99/吞吐从完整训练日志读取，不用打点。
- **计时必同步**：每个打点前后都要 `synchronize()`，否则测到的是下发时间。
- **收益成对对比**：基线必须当轮同卡重测；不拿历史数据当分母。
- **卡必须干净**：测试前用 `scripts/check_cards.py` 确认目标卡无其他进程；不能只看 HBM（空闲也可能有预留）。
- **并行只筛选，集成才定稿**：分卡并行只用于候选筛选；最终数字必须来自串行集成后的全栈复测。
- **逐点归因**：每个优化点单独测量、单独记录性能收益和精度影响；环境变量组作为一组，融合算子逐个。
- **最终 Profile 必复核**：所有已采纳优化点叠加后，必须在最终代码路径上采 `prof/final/`，与 P2 的 `prof/baseline/` 同口径对比；它证明机制生效和瓶颈迁移，不替代 P7 的逐点收益测试。`failed` 的优化点不能算有效。
- **无侵入优先**：只自动做环境变量与融合算子两类；其余改动只列清单，按预期收益从高到低交用户决策。
- **融合算子不是默认全开**：通常有收益但幅度取决于热点占比，且可能引入舍入误差；必须基于 P1 模型定义定位到 `file:line`，逐点测量性能与精度，精度不通过先尝试精度保持变体、仍不通过即回滚。
- **精度是硬门**：任何优化都要通过精度校验，回归即回滚。
- **预期不写成结果**：没实测的收益只能写"预期/待验证"。
- **失败也记录**：负收益、回滚的实验同样进台账，避免重复踩坑。
- **打点不入最终 patch**：probes 只用于测量，最终交付的 patch 不含打点代码。

## 文件索引（渐进披露）

| 文件 | 何时读 |
|---|---|
| `references/pipeline-analysis.md` | P1 调用链拆解、模型定义与调用映射、Mermaid 模板、扩散模型拆解示例 |
| `references/instrumentation.md` | P2 打点规范、sync 语义、扩散/训练拆解与统计口径 |
| `references/warmstart-defaults.md` | P3 torch_npu 接入、环境变量候选集 |
| `references/fusion-op-catalog.md` | P3 扫描 / P5 选择融合算子、收益与精度权衡决策 |
| `references/profiling-recipes.md` | P4 采集规范：profiler 配置、脚本注入方式、产物验证、npu-smi |
| `references/profiling-analysis.md` | P4 深度分析：step_trace 分型、算子透视、Roofline、Hostbound、慢卡、训练 profiling/MFU |
| `references/diagnosis-playbook.md` | P4/P5 瓶颈证据 → 优化方向 |
| `references/optimization-loop.md` | P5 单变量、固化、回滚、候选看板、重采时机、最终验收 |
| `references/inference-optimization-playbook.md` | P5 推理候选池：attention 后端、融合算子、显存/VAE、多卡并行、量化、图模式、评测协议 |
| `references/training-optimization-playbook.md` | P5 训练候选池：数据加载、并行策略、显存/重算、通信、图模式/量化、MFU 与训练口径 |
| `references/precision-validation.md` | P7 输出级对比与逐层 dump |
| `references/report-template.md` | P8 最终报告固定模板 |
| `references/multi-agent-workflow.md` | 默认多 Agent 协作：角色卡、交接规范、并行/隔离、集成验证 |
| `scripts/env_check.py` | P0 环境、版本、设备、git 信息 |
| `scripts/stage_timer.py` | P2 分段打点（默认关闭，`NPU_TUNE_PROBE=1` 开启） |
| `scripts/npu_monitor.py` | P4 npu-smi 遥测采样 |
| `scripts/check_cards.py` | P7/多 Agent 测试前卡空闲门禁与卡分配校验 |
| `scripts/profile_summary.py` | P4 step_trace / kernel_details / op_statistics（缺失回退 op_summary）汇总与瓶颈分型；`--rank` 指定 rank |
| `scripts/profile_compare.py` | P7.5 baseline vs final profile 对比：step_trace 占比、Task Type、Top-N 新增/消失 kernel 与算子 |
| `scripts/scan_fusion_ops.py` | P3 扫描可用 `npu_*` 算子（`--detail` 查看签名与说明） |
| `scripts/compare_outputs.py` | P7 输出级精度对比 |
| `scripts/make_patches.py` | P8 按优化 commit 生成 patch |
| `scripts/verify_stack.py` | P7 集成验证：在临时 worktree 按顺序应用 patch，可选跑 smoke/full 命令 |
| `scripts/experience.py` | P0 经验召回 / P8 经验沉淀与更新 |
| `experience/README.md` | 经验库 schema、置信度、使用命令、质量要求 |
| `assets/env.default.sh` | P3 环境变量候选模板 |
| `assets/invasive-candidates-template.md` | P5b 侵入性优化候选清单模板（按预期收益降序） |
| `assets/model-info-template.json` | P0 模型信息记录模板 |
| `assets/optimization-verification-template.md` | P7.5 优化点生效矩阵模板（预期签名 / baseline 证据 / final 证据 / 状态） |
