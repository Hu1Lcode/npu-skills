# P5 · 推理优化手册：DiT / World Model（Ascend NPU）

本文件是 P5 的**候选池与决策规则**：给定 P4 的瓶颈证据，按图索骥挑优化点。来源为社区 `awesome-ascend-skills` 的 inference 系列与 cannbot `model-infer-optimize` 方法论，已按本技能"无侵入优先（环境变量 / 融合算子自动做，其余列清单）"原则重新分级。

cosmos 类 World Model 的推理链路通常也是 `text/condition encoder → world model denoise/transform → decoder`，本手册的阶段顺序同样适用；具体算子与并行 API 以 cosmos-framework 的实际实现为准。

配套文件：融合算子清单见 `fusion-op-catalog.md`；实验协议见 `optimization-loop.md`；精度口径见 `precision-validation.md`。

## 目录

1. 优化路线图（按依赖顺序，不是按收益排序）
2. Attention 后端与 Flash Attention
3. 融合算子替换（Attention 子链路优先）
4. 显存与 VAE decode
5. 多卡并行（中/高侵入，列清单）
6. 计算削减（高级，中/高侵入）
7. 图模式与调度（中侵入，列清单）
8. 统一评测协议（所有优化点共用）
9. 候选看板与状态机（探索式优化时使用）
10. 常见症状 → 候选方向速查

## 1. 优化路线图（按依赖顺序，不是按收益排序）

```text
基线（P2/P4，固定场景与精度口径）
  → 1. 并行化 / 多卡部署（代码结构会变，必须最先定型）
  → 2. Attention 后端 / Flash Attention 替换
  → 3. 融合算子（Attention 子链路、Norm、FFN）
  → 4. 量化（可选；会改变 dtype 与 kernel 路径）
  → 5. 图模式（尽量捕获最终 dtype/layout/kernel/cache 路径）
  → 6. 多流 / 权重预取 / SuperKernel（探索式，按 profiling 证据追加）
```

顺序原因：
- 并行化会改通信组、切分方式与代码结构，后面所有阶段都建立在它之上；
- 量化改变 dtype 与算子路径，图模式应在其后，避免图按错误 dtype 捕获；
- 每阶段遵循 **分析 → 方案确认 → 实施 → 验证 → 阶段总结**，不可跳过验证直接进入下一阶段。

**本技能的处理方式**：只自动实施**无侵入**项——环境变量、框架已暴露的 attention backend / 融合算子接口开关、融合算子替换（含 mindiesd 内置融合接口）；并行化、量化、图模式、多流、Attention Cache、稀疏注意力、VAE 改写等一律只进"待决策清单"，**按预期收益从高到低排序**后交用户确认。

## 2. Attention 后端与 Flash Attention

Attention 是 DiT 的第一热点。先确认当前用的后端与 shape 特征（`q.shape[1] == k.shape[1]` 为 self-attention，否则 cross-attention），再逐个切换测量。

### 2.1 diffusers 路径

```python
# Ascend NPU 优先尝试的原生后端
pipe.transformer.set_attention_backend("_native_npu")
```

- 必须在启用 context parallel **之前**设置 attention backend。
- 若报 `Native attention does not support return_lse=True` 等内核限制，退化为多进程 data parallel 验证通路。

### 2.2 自研 / Wan 类路径（mindiesd 多后端）

```python
from mindiesd import attention_forward

# ALGO=0（默认）
attention_forward(q, k, v, op_type="fused_attn_score")
# ALGO=1
attention_forward(q, k, v, op_type="ascend_laser_attention")
# ALGO=3
torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads=..., input_layout="BSND")
```

| 环境变量 `ALGO` | 后端 | 适用场景 | 限制 |
|---|---|---|---|
| `0` | `fused_attn_score` | 默认首选 | — |
| `1` | `ascend_laser_attention` | 特定长序列 / 大 head 场景 | 仅 self-attention |
| `3` | `npu_fused_infer_attention_score` | 支持量化 attention 路径 | 仅 self-attention |

通用约束：
- `ALGO=1/3` 与 RainFusion 都只作用于 **self-attention**，cross-attention 始终走默认后端；
- NPU attention 路径通常强制 **BF16** 的 Q/K/V；FP32 精度要求会与之冲突；
- head 数无法被 `USE_SUB_HEAD` 整除时不允许开启 sub-head splitting。

切换后端属于配置级无侵入，但**每个后端都要独立测量**：不同 CANN/torch_npu 版本、不同 seq_len 下最优后端可能不同，不能照搬结论。

## 3. 融合算子替换（Attention 子链路优先）

按 P4 定位到的热点逐个替换，不要整网盲改。融合算子通常有性能收益，但幅度取决于热点占比；同时可能因 fp16/bf16 输入、累加顺序或近似激活引入舍入误差。先通过 P1 的模型定义映射定位到模型内部的具体 `file:line`，一次只换一个，并按 `fusion-op-catalog.md` 的收益/精度决策流程确认是否保留。核心替换：

| 原实现 | Ascend 替换 | 关键点 |
|---|---|---|
| 手写 RMSNorm（`.float()` 后计算再 `type_as`） | `torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]` | 返回 tuple，取 `[0]` |
| `nn.LayerNorm` + `.float()` | `F.layer_norm(x, normalized_shape, weight, bias, eps)` 直接用原生 dtype | 去掉 `.float()`，否则多一次转换 kernel |
| 手写复数乘法 RoPE（float64/complex128） | `mindiesd.rotary_position_embedding(x, cos, sin, rotated_mode="rotated_interleaved", fused=True)` | 预先算好 cos/sin，`freqs_list` 按分辨率缓存并及时清理 |
| `flash_attention()` 单函数 | 2.1/2.2 的多后端 dispatch | 先保证 self/cross 分流正确 |
| 逐元素 add + norm | `torch_npu.npu_add_rms_norm` 等 | 见 `fusion-op-catalog.md` |

精度注意事项：
- 融合算子通常要求 FP16/BF16 输入，`.float()` 保留会抵消收益；替换后必须做输出级精度对比。
- 位置编码里的 float64/complex128 在 NPU 上支持差、且会落 AICPU，优先降为 float32/complex64。
- 需要跨平台复现随机数时，按框架语义设置 `PRECISION`（本技能模板取 `PRECISION=0`），让随机数在 CPU 生成后再搬回 NPU。

## 4. 显存与 VAE decode

### 4.1 显存估算

| 组件 | 估算 |
|---|---|
| 模型参数 | 参数量 × dtype 字节数（BF16 = 2 bytes） |
| 推理激活 | 约 1–3× 模型参数大小 |
| VAE decode | 与输出分辨率相关，1024×1024 约 1–2 GB |
| 安全余量 | 预留 2–4 GB |

分辨率对显存近似**平方增长**：512→1024 约 4×，512→2048 约 16×。

### 4.2 显存优化手段（按侵入性从低到高）

| 手段 | 显存收益 | 性能代价 | 备注 |
|---|---|---|---|
| `torch_dtype=torch.bfloat16` | 参数减半 | 通常更快 | 首选，前提是精度可接受 |
| `pipe.enable_attention_slicing()` | 降注意力峰值 | 略慢 | 切片越小越省显存 |
| `pipe.enable_vae_slicing()` | 降 VAE 批量峰值 | 略慢 | 批量生成时有效 |
| `pipe.enable_vae_tiling()` | 降高分辨率 decode 峰值 | 略慢 | 1024×1024 以上优先 |
| `pipe.enable_model_cpu_offload(device="npu:0")` | 显著降显存 | 受 PCIe 带宽限制，常明显变慢 | NPU 上实验性，先验证兼容性 |
| `enable_sequential_cpu_offload` | 最省显存 | 最慢 | 仅其他手段装不下时用 |

注意：使用 CPU offload 时**不要**先 `pipe.to("npu")`；Generator 的设备要与模型一致（offload 时用 CPU generator）。VAE 大分辨率 decode 往往是 OOM 的第一来源，先上 VAE tiling 再考虑 offload。

### 4.3 VAE patch parallel（多卡，中/高侵入）

- 将 VAE decode 的 H×W 空间切成多块分给多个 NPU；
- 相邻 patch 通过 P2P 交换边界 halo 数据；
- 需要 monkey-patch `conv2d/conv3d/interpolate/pad` 的边界逻辑，并调整 CausalConv3d 的 padding 策略；
- 收益在小显存卡 + 高分辨率场景明显，但实现复杂、边界误差需重点验证。

## 5. 多卡并行（中/高侵入，列清单）

### 5.1 diffusers（>= 0.36）

```python
from diffusers import ContextParallelConfig

pipe.transformer.set_attention_backend("_native_npu")
pipe.transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=2))
```

- 默认优先 **Ulysses**；复杂场景可用 `ContextParallelConfig(ring_degree=1, ulysses_degree=2)` 组合；
- `torchrun --nproc_per_node=N` 启动，`dist.init_process_group(backend="hccl")`；
- 版本低于 0.36 没有该 API，退化到多进程 data parallel（每卡一个进程，按请求切分）。

### 5.2 Wan 类自研并行框架

`ParallelConfig` 维度：

```text
world_size = tp_degree × sp_degree × cfg_degree
sp_degree  = ulysses_size × ring_size
cfg_degree ∈ {1, 2}   # CFG parallel 开启时为 2
```

- **CFG parallel**：每个 denoising step 本来要跑 cond + uncond 两次前向，CFG parallel 让两个 rank 各算一个分支再 all_gather，单卡计算量约减半。是最值得优先评估的并行手段。
- **Ulysses**：按 head 维切分，all-to-all 转置；**Ring**：按序列维切分，环形传 KV。
- **TP**：Attention 的 QKV/O 与 FFN 的 Column/Row ParallelLinear；RowParallel 可接 `torch_npu.npu_mm_all_reduce_base` 做通算融合。
- 约束：各并行度乘积必须精确等于 world_size；TP+FSDP 时先 TP 再 FSDP 包装。

### 5.3 验证要求

并行化不是"能跑就行"：要验证每个 rank 输出一致/合理、精度达标、扩卡效率（不是线性至少不能倒退）、通信占比（P4 复测 < 10% 为目标）。最终报告中的性能必须是固定部署口径下的数据。

## 6. 计算削减（高级，中/高侵入）

### 6.1 Attention Cache（`mindiesd.CacheAgent`）

去噪循环相邻 step 的 attention 输出高度相似，可跨 step 复用：

```python
from mindiesd import CacheConfig, CacheAgent

config = CacheConfig(
    method="attention_cache",
    ratio=1.2,        # 缓存复用比例
    interval=4,       # 跳步间隔
    start_step=12,    # 从第 12 步开始
    end_step=37,      # 到第 37 步结束
)
cache = CacheAgent(config)
```

需要给每个 Transformer block 注入 `cache` 与 `args`，并在 block forward 中用 `cache.apply(...)` 包住 attention。属于算法级近似，必须做画质/指标对比（PSNR/SSIM/CLIP 等），并在报告中明确"近似优化"。

### 6.2 RainFusion 稀疏注意力（v1/v2）

利用视频 token 的时空局部性跳过部分 attention 计算：

```python
rainfusion_config = {
    "type": "v1",          # 或 "v2"（blockwise Top-K）
    "sparsity": 0.64,      # 推荐起点：质量与速度平衡
    "skip_timesteps": 15,  # 前 15 步用全量 attention
    "atten_mask_all": None,
    "grid_size": None,
}
```

- 只作用于 **self-attention**；cross-attention 不动；
- 首帧与文本 token 始终保留全量 attention；
- `sparsity=0` 等于关闭；`0.9` 速度最快但画质风险大；`skip_timesteps` 越大越保守；
- mask 与分辨率/grid 相关，换分辨率必须重算。

### 6.3 W8A8 量化（msmodelslim）

- 离线：`msmodelslim` W8A8 **dynamic** 量化（无需校准数据），`disable_level='L0'` 让不兼容层自动回退；
- 运行时：`mindiesd.quantize(model, quant_des_path, use_nz)` 加载量化描述文件；
- 优先找 `quant_model_description_w8a8_dynamic.json`，没有则找 mxfp8 版本；
- MoE / 双专家（high-noise + low-noise）模型要分别量化两个子模型；
- 量化后 dtype 与算子路径都变，**必须重新 profiling**，并逐模块验证精度与回退清单。

## 7. 图模式与调度（中侵入，列清单）

- 后端：`torchair` / `npugraph_ex`（按模型与 CANN 版本选型）；
- 图模式适合**固定 shape 的 denoise 循环 / decode**；prefill 或动态 shape 段通常不开；
- 先分析图中断点：dynamic shape、数据依赖控制流、Python 副作用、`.item()` / `.cpu()` 同步；
- 首次迭代包含图编译成本，单独记录 `first_iter`，不能混入稳态中位数；
- 图模式与多流可能互斥或相互放大，需在同一代码路径上做组合验证。

多流 / 权重预取 / SuperKernel 属于探索式优化：只在 profiling 明确指出"单流串行、权重加载卡顿、kernel launch 开销大"时作为候选，必须按 `optimization-loop.md` 的候选看板管理，不能默认全开。

## 8. 统一评测协议（所有优化点共用）

### 8.1 性能（口径以 `instrumentation.md` 为准）

| 场景 | 筛选 / 实验 / profiling | 基线采集 | 定稿 / 上报架构师 / 最终报告 |
|---|---|---|---|
| 推理 | warmup 1 轮；denoise steps 可降到 5 步节省时间 | warmup 1 轮；denoise steps 固定 10 步左右；独立跑 3 次任务，E2E 抖动 ≤ 10%；采各阶段 + DiT 单步（5 步预热后） | 与基线同一步数跑完整 pipeline / 原始启动脚本；3 次任务取中位数，报 E2E + 各阶段 + DiT 单步 |
| 训练 | warmup 20 steps；实验测 10 steps 即可 | 独立跑 3 次任务，日志 `s/step` 抖动 ≤ 10%；P50/P99/吞吐取自完整训练任务日志；单步内阶段用打点拆解 | 原始启动脚本完整训练流程，从日志稳定段取 `s/step`、P50、P99、吞吐 |

补充：
- 同步：停止计时前必须 `torch.npu.synchronize()`（或 `torch.cuda.synchronize()` 经 `transfer_to_npu` 等价）。
- **推理只有时间指标**：E2E、各阶段、DiT 单步时间；**不报 P50/P99/吞吐**。P50/P99 只在训练场景使用，且必须来自完整训练日志。
- 环境：同卡同机、卡干净（`check_cards.py`）、同 batch/分辨率/步数/dtype；基线当轮重测。
- **每个优化点无论筛选还是定稿，都必须同时记录**：推理的 E2E + 各阶段 + DiT 单步；训练的日志 `s/step` + 单步内各阶段拆解。短步数结果只能筛选，不能作为最终收益。
- 基线必须独立跑 3 次；E2E（推理）/ `s/step`（训练）抖动 > 10% 时基线无效，重采。

### 8.2 精度

- 参考实现：CPU FP32 输出（或固定的原始 NPU eager 输出）作为 golden；
- 指标：`cosine` 为主判据；同时记录 `max_abs_diff`、`rel_l2`、PSNR/SSIM（图像/视频）用于定位差异来源；
- 判定（用户已确认统一用 cosine）：
  - `PASS`：`cosine ≥ 0.999`；
  - `FAIL`：`cosine < 0.999` → 回滚或逐层 dump 定位；
  - `max_abs_diff` 等指标不作为第二门槛，只写入报告辅助说明。
- 图模式/量化等改动必须加 **torchair-vs-eager** 或 **quant-vs-fp** 的同设备对照，隔离"图/量化差异"与"精度舍入差异"。

### 8.3 报告口径

- 每个优化点：现象（P4 证据）→ 为什么（机制）→ 改动 → 性能收益（standalone 与 stacked）→ 精度影响 → 风险与回退；
- 最终性能以串行集成后的全栈复测为准，分卡并行结果只用于筛选；
- 没实测的收益只能写"预期/待验证"。

## 9. 候选看板与状态机（探索式优化时使用）

当 P5 同时管理多个候选、或用户要求"在 baseline 之上继续榨性能"时，用以下状态机管理，避免候选丢失或重复（与 `optimization-loop.md` 一致，若有出入以 `optimization-loop.md` 为准）：

| 状态 | 含义 | 进入条件 |
|---|---|---|
| `待决策` | 有侵入（环境变量/融合算子以外）或存在冲突，需用户拍板 | 方案超出无侵入范围 |
| `待实现` | 用户已批准（或属无侵入项），尚未实施验证 | 候选初次登记且允许实施 |
| `通过` | 有明确收益，功能/精度/回退均验证通过 | reviewer 建议通过或主 Agent 有更强证据 |
| `淘汰` | 无收益、风险过高、实现失败、破坏精度，或被同互斥组更优方案替代 | 任一淘汰条件成立 |

判定规则：
- **判通过**：代码路径在目标场景真实执行到；精度/功能满足口径；目标指标达标或保留价值明确；开关与回退可用；不破坏已通过项；互斥组内无更优已通过项。
- **判淘汰**：明确失败且不可补验；功能/精度/稳定性/编译问题；收益低于噪声且无调优空间；明显副作用（内存暴涨、layout 转换、图退化）；与已通过项冲突且更差。淘汰必须记录：证据、代码是否回退/开关是否关闭、是否派生新 Plan。
- **保持待实现**：实现未完成但卡点可修复；只是证据不足需补验；方向成立但参数/强度待调。

互斥与叠加：
- 同一互斥组最终只保留一个或一组明确兼容的 `通过` 方案（如"融合算子改写 attention"与"多流切分 attention"可能互斥）；
- 可叠加方案必须在最终验收前**实测组合效果**，不能假设收益相加；
- 组合后退化 → 追加组合验证轮，必要时淘汰收益更差/风险更高一方。

**决策清单排序规则**：按预期性能收益从高到低（统一折算为 E2E 时延的百分比）——优先用 P4 直接证据（`热点阶段/算子耗时占比 × 预计可消除比例`），其次用经验库/同类模型实测收益，最后是理论估算并标注"待验证"；收益接近时，侵入低、工作量小、依赖少的优先。

**重采 profiling 的触发条件**（决定是否需要回到 P4）：
- 改动改变了算子下发时序、计算图结构、dtype 或 layout（多流、融合、图模式、量化基本都算）；
- 现有 profile 已不能反映当前代码；
- 切换到新方向/新热点模块；
- 准备终裁或最终验收前，必须对最终代码路径重采一次。

**最终验收清单**：
- [ ] 看板无 `待实现` 候选；
- [ ] 所有 `通过` 项已在最终代码路径真实生效；
- [ ] 所有 `淘汰` 项已回退或开关关闭；
- [ ] 互斥组状态自洽；
- [ ] 可叠加组合已实测；
- [ ] 精度/功能仍满足初始口径；
- [ ] 最终性能与 baseline 同场景、同口径对照；
- [ ] 无法再派生出仍有潜在收益的新候选。

## 10. 常见症状 → 候选方向速查

| P4 证据 | 优先候选 | 侵入性 |
|---|---|---|
| attention 算子占比最高、`aic_mac_ratio` 高但效率 < 70% | 换 attention 后端 / FA 版本，检查 head_dim 对齐与合批 | 低 |
| RMSNorm/RoPE 等小算子多、AICPU 占比高 | 融合算子替换、float64→float32 | 低 |
| VAE decode 占比高、逐元素操作多 | VAE 融合、减少 transpose/contiguous、VAE tiling | 低–中 |
| 显存不足 / OOM 边缘 | BF16、attention slicing、VAE tiling、CPU offload | 低 |
| 多卡扩展不线性、HCCL 占比 > 10% | CFG parallel、Ulysses/Ring、HCCL_BUFFSIZE、通算融合 | 中–高 |
| denoise 循环 step 间相似度高、attention 是热点 | Attention Cache | 中–高 |
| 视频 token 长、attention 计算量巨大 | RainFusion 稀疏注意力 | 中–高 |
| 计算图固定、launch 开销占比高 | torchair / npugraph_ex 图模式 | 中 |
| 单流串行、权重加载与计算不重叠 | 多流 / 权重预取 / SuperKernel | 高 |
| dtype 为 FP32、显存与算力都吃紧 | W8A8 量化 | 高 |

使用这张表时，候选仍必须回到 P4 证据确认，不能只凭症状对号入座。

## 参考来源

- `awesome-ascend-skills/skills/inference/diffusers-ascend`（diffusers NPU pipeline、显存优化、多卡推理）
- `awesome-ascend-skills/skills/inference/wan-ascend-adaptation`（Wan 类 DiT 适配：算子替换、attention 多后端、CFG/TP/SP 并行、VAE patch parallel、量化、RainFusion）
- `awesome-ascend-skills/skills/inference/npu-torchair-infer`（精度/性能评测方法学）
- cannbot `plugins-official/model-infer-optimize`（阶段化优化流程、Plan 状态机、重采 profiling 与最终验收规则）

具体 API 与参数以目标模型的框架版本为准；社区经验需要在本机实测验证后才能固化。
