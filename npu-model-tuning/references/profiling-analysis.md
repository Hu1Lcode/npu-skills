# Profiling 深度分析：从 step_trace 到瓶颈结论（P4 / P7.5）

本文件回答"拿到 profiling 数据后怎么分析"。采集与产物落盘见 `profiling-recipes.md`；P4 负责**分型 → 下钻 → 证据 → 候选方向**，P7.5 负责**baseline vs final 最终全栈复核**（见第 9 节）。

分析顺序固定如下，不要跳步：

```text
数据完整性检查
  → step_trace_time.csv 总览分型（Computing / Communication / Free）
  → 按瓶颈类型下钻（Computing / Hostbound / Communication / 慢卡）
  → 与 P2 分段耗时交叉验证
  → 输出 profiling_analysis.md（证据 + 候选方向 + 置信度）
```

> 采集 profiling 时先关闭 P2 打点（`NPU_TUNE_PROBE=0`）。打点里的 `synchronize()` 会人为制造空泡，让 Free 占比虚高、误判成 Hostbound。

## 1. 数据完整性检查（先做，缺数据不要猜）

| 检查项 | 要求 | 缺失处理 |
|---|---|---|
| `step_trace_time.csv` | 存在且至少覆盖 1 个完整 step/推理 | 训练场景缺失则重采；推理无训练循环时可能不生成，改用 kernel_details/op_statistics + P2 分段统计 |
| `kernel_details.csv` | 含 `Task Type`、`Duration(us)`；Level1 + PipeUtilization 下应有 pipe 比率列 | 检查 profiler 级别与 `aic_metrics` 配置后重采 |
| `op_statistics.csv` / `op_summary.csv` | 至少一个存在，用于算子聚合 | 只靠 kernel_details 也可，但聚合口径要说明 |
| 多卡场景 | 默认只采/只分析主 rank（如 rank 0）一份；全 rank 仅限用户确认的慢卡/通信专项 | 记录采集范围，避免把单卡结论套到集群；需要集群结论时先向用户申请补采全 rank |
| 采集开销 | 记录 profiler 配置与 active 步数 | 写进 `manifest.json`，报告可复现 |

## 2. 第一步：step_trace_time.csv 总览分型

`step_trace_time.csv` 通常包含 `Step`、`Computing`、`Communication(Not Overlapped)`、`Free` 等列（不同 CANN 版本列名略有差异）。可直接运行 `python scripts/profile_summary.py <prof_dir>`：它会解析 `step_trace_time.csv`、打印总占比与分 step 占比，并按下面的阈值给出分型。手工计算口径：

```text
Computing%     = Σ Computing                    / Σ (Computing + Communication(Not Overlapped) + Free)
Communication% = Σ Communication(Not Overlapped) / Σ 总时间
Free%          = Σ Free                          / Σ 总时间
```

**分型阈值（社区实践口径，按优先级从高到低判定）：**

| 优先级 | 判定 | 阈值 | 含义 |
|---|---|---|---|
| 1 | **Hostbound（下发/空泡）** | `Free > 20%` | 设备等 CPU 下发、Python 侧开销、同步点、数据加载 |
| 2 | **Computing（计算）** | `Computing > 85%` | 设备计算时间占主导 |
| 3 | **Communication（通信）** | `Communication > 10%` | HCCL 集合通信占比高 |
| 4 | 无明显瓶颈 | 以上都不满足 | 看具体阶段与算子，可能被平均掩盖 |

注意：
- 这是**启发式阈值**，不是物理定律。先看它指向哪类问题，再用下钻证据确认；平均占比会掩盖局部热点，必要时按 step / 阶段拆分统计。
- 扩散推理没有训练循环时，把一次完整 pipeline（或一次 denoise 循环）当作一个 step 来统计。
- 多卡默认只分析主 rank；如果主 rank 显示 Communication 占比异常或扩卡不线性，**向用户申请补采全 rank** 后再做逐 rank 对比——单 rank 数据无法判断慢卡。

## 3. Computing 高：算子级下钻

### 3.1 先筛 Top-N 高耗时算子

- 优先读 `op_statistics.csv`：按 `OP Type` + `Ratio(%)` 排序取前 3–5 个。
- 没有 op_statistics 时读 `op_summary.csv` 的 `OP Type` / `Task Duration(us)`，或 `kernel_details.csv` 的 `Type` / `Duration(us)` 聚合。
- 把 Top-N 算子与 P2 的分段（text_encoder / DiT denoise / VAE decode）对上，确认热点归属；再通过 P1 的模型定义与调用映射回查到具体模型类/forward 和算子实现 `file:line`，作为 P5 融合算子替换候选的输入。

### 3.2 看指令占比透视表（判断"算什么、卡在哪"）

对每个 Top 算子按 `Input Shapes` 分组，统计以下列的平均值：

| 指标 | 反映的流水 | 高值含义 |
|---|---|---|
| `aic_mac_ratio` | Cube MAC 利用率 | 算力打满，属真计算瓶颈（进入 Roofline 核对） |
| `aic_saclar_ratio` | Scalar 单元占比 | 小 shape / 标量逻辑多，Cube 吃不饱 |
| `aic_mte1_ratio` | L1 → L0A/L0B 搬运 | Cube 输入供给压力 |
| `aic_mte2_ratio` | GM → L1 搬运 | 带宽受限信号，先查是否符合预期搬运量 |
| `aic_fixpipe_ratio` | L0C → GM 写出 | 输出写回瓶颈 |
| `aiv_vec_ratio` | Vector 单元占比 | 逐元素/归约类计算为主 |
| `aiv_saclar_ratio` | Vector 侧 Scalar 占比 | 小算子、循环、地址计算多 |
| `aiv_mte2_ratio` | GM → UB 搬运 | Vector 侧带宽压力 |
| `aiv_mte3_ratio` | UB → GM 搬运 | 写回压力 |

判读要点：
- `aic_mac_ratio` 高 → 真正算力瓶颈，用 Roofline 判断是否值得优化单 kernel。
- `aic_mte2_ratio` / `aiv_mte2_ratio` 高而 MAC/Vec 不高 → 先怀疑 memory-bound、layout 转换、冗余搬运，而不是继续堆融合。
- `aic_saclar_ratio` / `aiv_saclar_ratio` 高 → 算子切得太碎或 shape 太小，优先合批、融合、改 tiling。
- `aic_fixpipe_ratio` 高 → 输出写回占主导，检查是否有大量中间结果写回 GM。

### 3.3 MatMul 类算子提取形状

对 MatMul / MatMulV2 / BatchMatMul 等，把 `Input Shapes` 解析成 `M, K, N`：
- M 或 N 很小（如 decode 单 token、小 batch）→ Cube 利用率天然低，单 kernel 优化空间有限，优先合批/融合/减少调用。
- K 或 N 未对齐（16/32/64/128 边界）→ 检查 padding、head_dim、hidden_size 对齐。
- Batch 很大但单批 shape 小 → 看能否合并 batch 维。

### 3.4 Roofline 核对（只对 `aic_mac_ratio >= 0.80` 的算子）

目的：判断"这个算子理论上该多快、现在差多少、值不值得优化"。

```text
理论耗时 = max(FLOPs / NPU峰值算力, 访存量 / NPU峰值带宽)
效率     = 理论耗时 / 实际耗时
```

前置条件（缺一不可）：
- `kernel_details.csv` 含 `aic_mac_ratio` 与算子形状/FLOPs 信息；
- 提供**真实的** FP16/BF16 Cube 峰值算力（TFLOPS）与 HBM 带宽（GB/s 或 TB/s）。

不要猜硬件峰值。查 `npu-smi info` / `ascend-dmi` / 官方规格；查不到时向用户确认，并把来源写进报告。

| 效率（理论/实际） | 判断 | 下一步 |
|---|---|---|
| 80%–120% | 已接近峰值，**不优先优化单 kernel** | 转向图级融合、合批、减少调用、算法降算量 |
| 70%–80% | 可选优化 | 核对合批与精度路径 |
| 40%–70% | 建议优化 | 查 M/N/K、合批、库路径、tiling |
| < 40% | 需要优化 | 同上，优先处理浪费最大的算子类型 |
| > 150% | 模型高估，**不据此排序** | 检查 FLOPs 估算是否把访问量算多了 |

补充：单次 `Duration < 10us` 的低效率通常是 launch 开销，不是 kernel 本身的问题。

## 4. Hostbound（Free > 20%）：找"谁在等"

### 4.1 设备侧证据

- 小 kernel 数量极多、`Duration` 很短，但相邻 kernel 之间有空隙；
- `enqueue` / `dequeue` / `async_npu` 等 runtime 事件占比高；
- CPU 侧单线程忙，NPU 利用率周期性掉 0；
- 训练场景还要排除 dataloader：step 之间空泡呈周期性、worker 处理时间低于等待间隔。

### 4.2 Host 侧 trace（ftrace）

MindStudio Insight 采集的系统 trace 可分析：
- 进程/线程运行时间；
- 上下文切换次数（频繁切换 → 调度抖动）；
- CPU 迁移次数（跨核迁移 → cache 失效）；
- 中断时间（中断风暴 → 主机侧阻塞）。

对应工具：`profiling-analysis-hostbound` 的 `trace_analyzer.py --input <trace文件> --output <报告>`。没有 ftrace 时，至少用 `npu_monitor.py` 看 CPU 利用率与 NPU 利用率的反相关系。

### 4.3 常见修复方向（低 → 中侵入）

| 方向 | 手段 | 备注 |
|---|---|---|
| 减少算子数 | 融合算子（RMSNorm/RoPE/Add+Norm 等） | 见 `fusion-op-catalog.md` |
| 下发优化 | `TASK_QUEUE_ENABLE=1/2`（按版本实测） | 环境变量，无侵入 |
| 绑核 | `CPU_AFFINITY_CONF=2` | 避免跨 NUMA |
| 图模式 | torchair / npugraph_ex | 中侵入，列清单 |
| 数据加载 | `num_workers`、`pin_memory`、预取、`/dev/shm` 缓存 | 训练场景优先 |
| Python 开销 | 去掉逐帧 PIL/numpy 转换、向量化、缓存 tokenizer | 扩散推理常见 |

## 5. Communication（> 10%）：通信与慢卡

### 5.1 先分清"通信慢"还是"有人拖后腿"

- 默认只有主 rank 数据时，先根据主 rank 的 `Communication(Not Overlapped)` 占比与扩展性判断是否存在通信瓶颈；怀疑慢卡/通信异常时，向用户说明理由并申请补采全 rank（用户决定，不自动全采）。
- 拿到全 rank 数据后，看所有 rank 的 `Communication`、`Computing`、`Free`：如果个别 rank 的 Computing 明显偏低、等待时间偏高，先怀疑慢卡。
- 用 `msprof-analyze cluster` 做慢卡检测：

```bash
msprof-analyze cluster -d ./profiling_data -m slow_rank -o ./result
```

判读（以工具输出为准）：
- 各 rank 的 `slowAffectCount`：最大值 `> 20` 时进入 Z-score 判定；
- `Z = (slowAffectCount - μ) / σ`：`Z > 0.5` 提示存在慢卡影响，`Z > 2` 通常判定为明显慢卡；
- 若 `slowAffectCount` 最大值 `≤ 20` 或 Z-score 不高 → 转通信算子异常分支，检查 plog 是否有 HCCL 超时/重传/建链异常。

### 5.2 通信本身的排查

| 现象 | 可能原因 | 检查 |
|---|---|---|
| 扩卡不线性、HCCL 占比随卡数上升 | 通信量/并行策略不匹配 | 并行切分、TP/SP/CP 度、每卡通信量估算 |
| 通信与计算串行 | 未重叠 | 开启通算融合/多流、检查 stream 依赖 |
| 个别 rank 常年等待 | 慢卡 / 负载不均 | 慢卡检测、数据分片、绑核 |
| 小消息多、时延高 | 通信粒度太细 | 增大 batch、梯度累积、合并通信 |

无侵入手段：`HCCL_BUFFSIZE` 调整（按版本与显存权衡）；有侵入：并行策略、通信组重构等，只列清单按预期收益降序交用户。

### 5.3 多卡 DiT 专项

- CP（Context Parallel）/ VAE patch parallel 场景，检查每个 rank 的序列长度切分是否均匀、边界交换是否成为同步点；
- CFG parallel 检查两个分支的耗时是否对称；
- 先用最简 data parallel 验证 HCCL 拉起正常，再上复杂并行。

## 6. 慢卡 / 卡间不均衡（仅在用户确认补采全 rank 后执行）

单 rank 采集无法做卡间对比；本节流程仅在用户同意补采全 rank 后使用。

除 `msprof-analyze cluster` 外，MindStudio Insight 的 `analysis.db` 可按 rank 统计：
- 计算时间（Computing）与通信时间（Communication）；
- 卡间差异百分比 = `(单卡指标 - 卡间中位数) / 卡间中位数`。

判定建议：
- 单卡 Computing 或 Communication 偏离中位数超过 10%–15%，且重复出现 → 标记可疑慢卡；
- 结合 `npu-smi info` 检查 HBM、功耗、温度、降频；
- 排除硬件后，再查绑核、NUMA、数据分片、进程共卡。

## 7. 训练场景专用

### 7.1 MindSpeed-MM（tools.json / YAML）

Megatron 引擎：在 `MM_TOOL` 指向的 `tools.json` 中加 `profile` 段：

```json
{
  "profile": {
    "enable": true,
    "profile_type": "static",
    "ranks": [0],
    "static_param": {
      "level": "level1",
      "with_stack": false,
      "with_memory": false,
      "record_shapes": false,
      "with_cpu": true,
      "save_path": "./npu_profiling",
      "start_step": 10,
      "end_step": 12,
      "data_simplification": false,
      "aic_metrics_type": "PipeUtilization",
      "analyse_flag": true
    }
  }
}
```

独立 FSDP2 引擎：在 YAML 末尾加同样的 `tools.profile` 结构。`profile_type: dynamic` 时通过 `dynamic_param.config_path` 下的 `profiler_config.json` 在运行中改 `start_step` / `end_step`。

### 7.2 MindSpeed-LLM

```bash
--profile --profile-step-start 2 --profile-step-end 4 \
--profile-ranks 0 --profile-level level1 --profile-with-cpu \
--profile-save-path ./profiling_output
```

### 7.3 MFU（训练效率）

```text
MFU = 模型每步理论 FLOPs / (step_time × NPU 峰值 FLOPS × 卡数)
```

- 峰值 FLOPS 必须来自实际硬件规格（如 Ascend910B2 的 FP16/BF16 Cube 峰值约 353 TFLOPS，仅作示例，不是所有机型通用值）。
- DiT / World Model 的 FLOPs 按 backbone（Transformer block 数、hidden、seq_len、patch 数）估算；VAE 单独统计 decode 占比。
- MFU 低不等于一定要优化计算：先排除数据加载、通信、Hostbound。

## 8. 输出：`profiling_analysis.md` 模板

分析结论写到 `perf_reports/<date>/profiling_analysis.md`，至少包含：

```markdown
# Profiling 分析（<日期> / <模型> / <推理|训练>）

## 采集信息
- profiler 配置、active 步数、采集 rank、prof 路径

## 总览
| 指标 | 值 |
|---|---|
| Computing% | |
| Communication% | |
| Free% | |
| 分型结论 | Computing / Hostbound / Communication / 无明显瓶颈 |

## 证据
- Top-N 高耗时算子（名称、耗时、占比、指令占比关键列）
- Roofline 结果（如有）
- 多卡均衡性 / 慢卡结论（如有）
- 与 P2 分段耗时的对应关系

## 候选优化方向（按优先级）
| 方向 | 依据 | 预期收益 | 侵入性 | 风险 |
|---|---|---|---|---|

## 限制与置信度
- 采集范围、样本量、profiler 开销、未覆盖模块
```

候选方向必须能回指到本节的具体证据（哪个算子、哪个指标、哪张表），不能只写"感觉 attention 慢"。

## 9. P7.5 最终全栈 Profile 复核

P7.5 与 P4 的目标不同：P4 找“当前最慢的点”，P7.5 证明“最终交付代码里，已采纳的优化机制确实存在，且瓶颈从原来的位置迁移走了”。它不做单项归因——单项收益仍看 P7 的 control vs candidate / stacked_gain。

### 9.1 采集前置检查

- [ ] 工作区来自 `baseline_commit` + `all.patch`，并已加载最终环境变量（`artifacts/env.applied.sh` 或 manifest 记录）；无未提交改动、无调试代码；
- [ ] `NPU_TUNE_PROBE=0`；
- [ ] `prof/baseline/manifest.json` 与 `prof/final/manifest.json` 的 profiler level、`aic_metrics`、rank、warmup、active/denoise steps、输入 shape/dtype/batch 一致；
- [ ] 两份 profile 的 `ASCEND_PROFILER_OUTPUT` 都包含必要的 `kernel_details.csv` / `op_statistics.csv`（推理场景按实际产物检查）。

### 9.2 baseline vs final 总览模板

先运行：

```bash
python scripts/profile_compare.py --baseline prof/baseline --final prof/final --json artifacts/profile_compare.json
```

按输出填写 `final_profiling_analysis.md`：

```markdown
# 最终 Profiling 复核（<日期> / <模型> / <推理|训练>）

## 口径
- baseline：`prof/baseline/`，采集于 P2、P3 O2 之前
- final：`prof/final/`，采集于 P7.5，`baseline_commit + all.patch` + 最终环境变量
- profiler：Level1 + PipeUtilization；rank=<N>；warmup=<N>；active=<N>；输入=<shape/dtype/batch>
- 说明：以下为机制证据，不作为最终性能数字

## 总览
| 指标 | baseline | final | 变化 | 解释 |
|---|---:|---:|---:|---|
| Computing% | | | | |
| Communication(Not Overlapped)% | | | | |
| Free% | | | | |
| AICPU Task Type 占比 | | | | |
| AICore / AIV / MIX 占比 | | | | |
| Top-1 热点 kernel | | | | |
| kernel 数 / 算子种类数 | | | | |

## 关键变化
- 新增/增强的融合 kernel：<名称、出现位置、耗时占比>
- 消失/下降的 kernel：<名称、原耗时占比、对应优化点>
- Host/Free 变化及解释：
- 通信/HCCL 变化及解释：
- AICPU/CPU fallback 变化及解释：

## 优化点生效矩阵摘要
见 `optimization_verification.md`。摘要：<通过 N 项 / mechanism_verified N 项 / time_only N 项 / failed N 项>
```

### 9.3 优化点生效矩阵

`optimization_verification.md` 每个已采纳优化点一行，模板见 `assets/optimization-verification-template.md`：

| 优化点 ID | 优化点 | 类别 | 预期 profiling 签名 | baseline 证据 | final 证据 | 逐点性能收益 | 状态 | 备注 |
|---|---|---|---|---|---|---|---|---|
| O2 | 环境变量组 | env | Free 下降、kernel 间隔缩短 | Free 28% | Free 18% | E2E -6.2% | verified | |
| O5 | npu_rms_norm | fusion | `RmsNorm` kernel 消失、`npu_rms_norm` 出现 | 有 `RmsNorm` 12% | 出现 `npu_rms_norm` 9% | DiT 单步 -3.1% | verified | |

### 9.4 状态判定与失败处理

| 状态 | 判定 | 是否可计入“有效优化” |
|---|---|---|
| `verified` | 预期 profiling 签名出现，且对应指标改善 | 是 |
| `mechanism_verified` | 预期签名出现，但整体指标被其他变化掩盖；需结合 P7 成对性能证据 | 是（须注明） |
| `time_only` | profile 看不出机制差异，但有 P7 同卡 control/candidate 成对性能收益 | 是（仅限有性能证据） |
| `failed` | 预期签名未出现且无 P7 性能证据；或 patch 未命中/被 fallback | 否，必须回查并回滚 |

失败回查顺序：patch 是否真正应用（`git apply --check` / diff 对照）→ 环境变量是否进入进程（`/proc/<pid>/environ` 或启动日志）→ 是否走了 fallback/其他后端 → 是否被后续优化覆盖 → 仍无效则从最终栈移除并重新采 final profile。

## 参考来源

- `awesome-ascend-skills/skills/profiling/profiling-analysis`（含 computing / roofline / hostbound / communication 子技能）
- `awesome-ascend-skills/skills/profiling/pytorch-profiling-collection`
- `awesome-ascend-skills/skills/profiling/mindspeed-mm-train-profiler`、`mindspeed-llm-train-profiler`、`training-mfu-calculator`

阈值与判据来自上述社区实践，使用时以目标机 CANN / torch_npu / msprof-analyze 版本的实际输出为准。
