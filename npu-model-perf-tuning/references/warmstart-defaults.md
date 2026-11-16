# P3 · Warm-start：经验有效的低侵入基础手段

基线量完后，优化阶段从这里开始。先上「经验有效、代码侵入小」的基础手段，落地后**先复测一次**量化收益，再进 P4 深挖。

## 3.1 torch_npu 接入（地基，最重要）

在推理/训练的**入口最前面**（先于任何 torch 调用、模型 import 之前）：

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu   # 自动把 torch.cuda.* 映射到 torch_npu.npu.*
```

- `transfer_to_npu` 会把 PyTorch 的 CUDA 相关 API 自动替换为 NPU 对应 API，是所有优化的前提；漏了它，后续 profiling、融合算子都可能走不到正确设备。
- **等价环境变量**：`TORCH_TRANSFER_TO_NPU=1` 可自动启用该功能（官方 env 文档已确认），不想改代码时可用。
- 改代码前备份；这是低侵入改动（入口一两行）。

## 3.2 环境变量起步集

`assets/env.default.sh`（可作为启动脚本前缀或 `source` 进去）：

```bash
export HCCL_BUFFSIZE=800
export CPU_AFFINITY_CONF=2
export TASK_QUEUE_ENABLE=2
export MULTI_STREAM_MEMORY_REUSE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

| 变量 | 经验值 | 官方简介（已核对 env 文档） |
|---|---|---|
| `HCCL_BUFFSIZE` | 800 | 集合通信缓冲区大小（HCCL/CANN 侧），提升通信吞吐 |
| `CPU_AFFINITY_CONF` | 2 | 粗/细粒度绑核，避免线程抢占与跨 NUMA 访问，减少调度开销 |
| `TASK_QUEUE_ENABLE` | 2 | task_queue 算子下发队列开关 + 优化等级 |
| `MULTI_STREAM_MEMORY_REUSE` | 2 | 多流内存复用开关 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | 控制缓存分配器行为、减少显存碎片（与 `PYTORCH_ALLOC_CONF` 同义，二者不能同时设） |

- 官方 env 文档（按场景分类：设备初始化/算子执行/单算子编译缓存/内存管理/分布式通信/日志告警/特征值/性能优化/图编译）：
  `https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/zh/api/environment_variable/env_variable_list.md`
- **P4 定位到瓶颈后**，按分类到该文档查对应变量再单独试（一次加一个，避免归因不清）。

## 3.3 融合算子首轮

用 `scripts/scan_fusion_ops.py` 扫描目标机 `dir(torch_npu)` 里所有 `npu_*`，按 `references/fusion-op-catalog.md` 的映射，把热点原语换成融合算子：
- attention / SDPA → `npu_fusion_attention` / `npu_fused_infer_attention_score` / `npu_prompt_flash_attention`
- RMSNorm → `npu_rms_norm` / `npu_add_rms_norm`
- RoPE → `npu_rotary_mul` / `npu_apply_rotary_pos_emb` / `npu_interleave_rope` / `npu_mrope`
- FFN/激活 → `npu_swiglu` / `npu_fast_gelu` / `npu_ffn`

详细替换配方（含参数、精度对齐）复用 `model-infer-fusion` skill。

## 落地方式与注意事项
- 环境变量与 import 属**低侵入**；融合算子替换一般也是**低~中侵入**（改几处 forward 调用即可）。
- 每加一批改动：**先复测**（同 P2 口径）→ 记录 before/after → 有效则固化、无效则回滚。
- 融合算子替换后**必须做精度校验**（数值等价/在容差内），有偏差先查 dtype、mask、scale、布局参数。
- 不要把"基础手段"和"深挖优化"混在一轮里，否则无法归因。
