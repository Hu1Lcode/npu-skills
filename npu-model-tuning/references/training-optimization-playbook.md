# P5 · 训练优化手册：DiT / World Model（Ascend NPU）

本文件是训练场景的**候选池与决策规则**：先按 P4 的 profiling 证据选方向，再按“无侵入优先、侵入性列清单交用户拍板”的原则推进。训练优化与推理优化的最大区别是：**step time 由数据、前反向、优化器和通信共同决定，任何优化都必须以日志里的 `s/step` / 吞吐为权威口径**。

## 目录

1. 优化路线图
2. 训练链路与基线口径
3. 数据加载与 Host 侧
4. 并行策略（中/高侵入）
5. 显存与重计算
6. 计算与融合算子
7. 通信与重叠
8. 图模式、量化与缓存
9. 训练专用 profiling 指标
10. 候选排序与上报

## 1. 优化路线图

```text
基线（P2/P4，固定数据与口径）
  → 1. 数据加载 / H2D / Host 下发
  → 2. 单卡热点算子与融合（RMSNorm / RoPE / Attention / FFN）
  → 3. 显存（BF16 / 激活重算 / 优化器状态）
  → 4. 并行策略（DDP → FSDP/ZeRO → TP/PP/SP/EP；会改代码结构，尽早定型）
  → 5. 通信重叠 / 梯度累积 / 通算融合
  → 6. 图模式、量化、缓存（最后做，避免图按错误 dtype/layout 捕获）
```

顺序原因：数据/下发问题会掩盖真实计算热点；并行策略会改变通信组与代码结构；图模式和量化应建立在最终 dtype/layout/kernel 路径之上。每一步仍按 P5 的单变量流程验证。

## 2. 训练链路与基线口径

- **训练没有 E2E 任务时间指标，只有单步时间**；所有时间拆解都在单步内部完成。
- 阶段拆分：`dataloader`（含预处理与 H2D）→ `forward`（含 loss）→ `backward` → `optimizer.step()` → `zero_grad()`；多卡时标注 HCCL 所在的阶段。若 step 内实际还包含 text_encoder / VAE / DiT 等模型组件，按 P1 的实际调用链继续拆分子阶段（与推理一样要能看到具体组件，而不是只有一个 `forward` 总数）。
- **权威指标是日志**：训练框架通常打印 `s/step`、`iter_time`、P50/P99 和吞吐；吞吐 = `global_batch_size / s_per_step`。分段打点只做少量 step 的指示性拆分，不能把分段之和当作 `s/step` 或吞吐。
- 基线必须独立跑 **3 次任务**，用日志 `s/step` 计算抖动 `(max-min)/median`，要求 **≤ 10%**，否则重采。
- **P50/P99 必须来自完整训练任务日志**，不能用少量 step 的打点数据计算；打点只用于 `dataloader` / `forward` / `backward` / `optimizer` / `zero_grad` 的单步内占比拆解。
- 口径：**warmup 20 steps**；筛选/实验测 **10 steps**；定稿/上报用原始启动脚本完整流程，按日志稳定段统计 `s/step`、P50、P99、吞吐。固定数据、seed、batch、梯度累积和卡数。
- 每个实验记录：日志 `s/step`（3 轮中位数与抖动）、P50、P99、吞吐、显存峰值，以及单步内各阶段耗时（中位数/占比）。

## 3. 数据加载与 Host 侧

先排除数据空泡，再谈计算优化。注意：**除环境变量外，数据加载相关改动（`num_workers`、`pin_memory`、prefetch、DataLoader 代码等）一律按 P5b 列清单交用户决策**；只有 `OMP_NUM_THREADS` / `MKL_NUM_THREADS`、`CPU_AFFINITY_CONF` 这类环境变量可归入 P5a 环境变量组。

| 现象 | 候选 | 处理 |
|---|---|---|
| dataloader 段高、设备周期性掉 0 | `num_workers`↑、`pin_memory`、`prefetch_factor`、`persistent_workers` | 均属 P5b 配置/代码改动，交用户决策；先测 CPU/内存是否成为新瓶颈 |
| 每 step 有固定 H2D 等待 | 预取、批量 H2D、数据常驻 device、`/dev/shm` 缓存 | 代码/配置改动，列 P5b |
| CPU 预处理重、线程抢占 | `OMP_NUM_THREADS` / `MKL_NUM_THREADS`、`CPU_AFFINITY_CONF` | 环境变量组，成组测量 |
| 小 batch 导致调度开销大 | 增大 batch / 梯度累积、融合小算子 | 受显存限制；记录全局 batch，启动参数/代码改动按 P5b 列清单 |

## 4. 并行策略（中/高侵入）

| 策略 | 适用 | 主要收益 | 主要风险 |
|---|---|---|---|
| DDP | 单机多卡，模型可单卡放下 | 数据并行、实现简单 | HCCL 通信占比、梯度 bucket |
| FSDP / ZeRO | 参数量/优化器状态超出单卡 | 分片参数、优化器状态 | 通信量增加、实现复杂 |
| TP | 单层参数量大、单卡算力不足 | 层内切分 | 通信频繁、layout/切分对齐 |
| PP | 层数多、可流水 | 跨层切分 | 气泡、micro-batch 调度 |
| SP / CP | 长序列视频/世界模型 | 序列切分、降低单卡激活 | attention 需配套实现 |
| EP / MoE | 稀疏专家模型 | 专家并行 | 路由/负载均衡、all-to-all |

- 并行策略属于**有侵入**，只列 `invasive_candidates.md`，按预期收益从高到低交用户决策；用户批准后再按单变量流程实施。
- 先用小规模 smoke 验证并行正确性，再测性能；并行改造后重新建立 baseline，不能拿旧单卡数字对比。
- 多卡测试前用 `scripts/check_cards.py` 确认卡空闲；慢卡/通信异常需要全 rank profiling 时**先向用户申请**。

## 5. 显存与重计算

- BF16/FP16 混合精度（先确认模型数值稳定性）；优化器状态分片/ZeRO；激活重算（gradient checkpointing）以时间换显存。
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 缓解碎片；`MULTI_STREAM_MEMORY_REUSE` 视版本支持。
- 减小 micro-batch、增大梯度累积；检查是否有未释放的缓存/中间张量常驻。
- 显存优化后必须重新测 step time；重计算可能让计算变慢，收益要用日志 `s/step` 判断。

## 6. 计算与融合算子

- 先按 P4 找 AICPU / 小 kernel / memcpy 热点，再通过 P1 的模型定义映射定位到模型实现的具体 `file:line`，替换 `torch_npu` 融合算子：RMSNorm、RoPE、SwiGLU/FFN、Attention、Add+Norm；一次只换一个并做精度对比。
- 融合算子通常能减少下发/访存并带来正收益，但收益大小由热点占比决定；可能因 fp16/bf16 输入、累加顺序或近似激活产生舍入误差。训练中以固定 batch 前向输出 `cosine ≥ 0.999` 为主判据，loss / gradient norm 辅助监控；不通过先尝试精度保持变体或只替换非敏感层，仍不通过就回滚。
- Attention 训练优先 `npu_fusion_attention`（按 dtype/layout/mask 选 v2/v3）；推理后端不直接套到训练。
- 通算融合：`npu_all_gather_base_mm`、`npu_mm_all_reduce_base`、`npu_mm_reduce_scatter_base`，需确认并行策略与 shape 匹配。
- 融合算子替换属于无侵入自动项；需要改模型结构/接口的归入 P5b。

## 7. 通信与重叠

- 先看 `step_trace_time.csv` 的 `Communication(Not Overlapped)` 与 HCCL task 占比；再看并行策略是否产生过多小消息。
- 候选：`HCCL_BUFFSIZE`↑、梯度累积/更大 batch 提升计算通信比、通信与计算重叠、通算融合、减少不必要的 all-gather/reduce-scatter。
- 并行策略调整属于有侵入；通信库参数属于环境变量组。
- 扩卡不线性时，先按用户确认的流程补采全 rank，再判断慢卡；单 rank 数据不能下慢卡结论。

## 8. 图模式、量化与缓存

- 图模式（torchair / npugraph_ex / torch.compile）可减少下发开销，但会改变执行图；放在并行、dtype、layout 稳定之后，作为侵入性候选列清单。
- 量化（W8A8 等）会改变 dtype 与 kernel 路径；先做精度评估，再测性能；训练量化要与优化器/梯度 scale 一起验证。
- 算子编译缓存（`ASCEND_OPP_CACHE`）用于减少首次编译，不改变稳态 step time；不要把它当成性能优化点上报。

## 9. 训练专用 profiling 指标

- `step_trace_time.csv`：Computing / Communication(Not Overlapped) / Free；训练还要排除 dataloader 空泡。
- `kernel_details.csv`：Task Type（AICore/AIV/MIX/AICPU/HCCL/memcpy）与 Top kernel。
- MindSpeed-MM（`tools.json` / YAML `tools.profile`）与 MindSpeed-LLM（`--profile` 系列参数）的采集配置见 `profiling-analysis.md` §7；训练吞吐仍以框架日志的 `s/step` 为准。
- MFU：`模型每步理论 FLOPs / (step_time × NPU 峰值 FLOPS × 卡数)`；峰值必须用真实硬件规格；MFU 低先排除数据/通信/Hostbound。
- 记录 global batch、梯度累积、并行配置、卡数，否则不同实验的 `s/step` 不可比。

## 10. 候选排序与上报

- 排序统一折算为 `s/step` 或吞吐的百分比（训练没有 E2E 任务时间指标）：优先 P4 直接证据（热点阶段/算子占比 × 预计消除比例），其次经验库/同类模型实测，最后理论估算并标“待验证”。
- 收益接近时，侵入低、工作量小、依赖少的排前面。
- 每项写清：优先级、优化点、P4 证据、预期收益及依据、侵入程度、改动位置 `file:line`、工作量、风险/副作用、依赖、回退方式；模板见 `assets/invasive-candidates-template.md`。
- 只有用户批准后才实施侵入性训练改动；无侵入项（环境变量组、融合算子）仍按 P5a 逐点测量并固化。
- 训练精度校验：固定 batch 前向输出的 `cosine ≥ 0.999` 为主判据；loss / gradient norm 作为辅助监控，需在基线波动范围内。若没有可对比的前向输出，测试前向用户确认替代口径。
