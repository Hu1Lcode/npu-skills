# P3 · Warm-start：基础接入与环境变量

基线量完后，先上"经验有效、代码侵入小"的基础手段，落地后按 P2 同口径复测一次，量化这一组（或一项）的收益，再进入 P4 深挖。

## 3.1 torch_npu 接入检查（O1）

在推理/训练的**入口最前面**（先于任何模型 import、先于设备使用）：

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

- `transfer_to_npu` 会把 `torch.cuda.*`、`.cuda()`、`torch.cuda.synchronize()` 等映射到 NPU，是很多 CUDA 风格代码能在 NPU 上运行的前提。
- **先检查再补**：如果模型/框架已经自己导入了 `torch_npu` 或已经使用 `torch.npu`，不要重复添加；重复导入通常无害，但 `transfer_to_npu` 可能有副作用，需确认。
- 等价环境变量：`TORCH_TRANSFER_TO_NPU=1` 可在不改代码的情况下启用（以目标机 torch_npu 版本支持为准）。
- 这一项作为 **O1** 单独记录、单独测量（如果没有改动则标注"已满足"）。

## 3.2 环境变量候选集（O2，成组测量）

`assets/env.default.sh` 是候选模板。**不要无脑全开**，按场景选择并逐项确认版本支持：

| 变量 | 参考值 | 适用场景 | 说明 / 风险 |
|---|---|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | 按需 | 所有多卡场景 | 指定可见的物理 chip logic id；torch_npu 逻辑序号按此列表重排 |
| `TASK_QUEUE_ENABLE` | `1` 或 `2` | 下发/调度瓶颈 | 版本差异较大：有的环境用 1，有的用 2；必须实测是否支持、是否稳定 |
| `CPU_AFFINITY_CONF` | `2` | CPU 绑核/调度 | 减少跨 NUMA 访问与线程抢占；单卡小模型收益可能不明显 |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | 显存碎片/OOM 边缘 | 与 `PYTORCH_ALLOC_CONF` 同义，**不能同时设** |
| `MULTI_STREAM_MEMORY_REUSE` | `2` | 多流/流水线 | 版本支持需确认 |
| `HCCL_BUFFSIZE` | `800` | 多卡训练/通信 | 仅对 HCCL 通信有意义；单卡推理不要开 |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | 按 CPU 核数 | CPU 预处理重 | 避免线程过多互相抢占 |
| `ASCEND_OPP_CACHE` | 可选 | 首次编译慢 | 指定算子编译缓存目录，跨运行复用 |

**O2 作为一组测量**：在启动脚本前 `source` 或 `export`，跑同口径复测，报告里作为一个优化点记录总收益。若某个变量明显无效或负收益，可以在组内移除并说明。

### 扩散 / Wan 类框架专属变量（仅当目标代码已支持时）

这些变量不是通用环境变量，而是框架（如 Wan-Ascend、mindiesd）自己读取的开关；**代码里没有对应实现时设置了也无效**。确认实现存在后再按组测量：

| 变量 | 参考值 | 作用 | 前提 / 风险 |
|---|---|---|---|
| `ALGO` | `0` / `1` / `3` | Attention 后端：0=fused_attn_score，1=ascend_laser_attention，3=npu_fused_infer_attention_score | 仅 self-attention 生效（1/3）；不同版本支持不同 |
| `FAST_LAYERNORM` | `0` / `1` | 启用 mindiesd fast layernorm | 需 mindiesd 与代码分支 |
| `USE_SUB_HEAD` | `0` 或分组数 | Attention 子头切分，降低单 kernel head 压力 | `num_heads % 分组数 == 0` |
| `T5_LOAD_CPU` | `0` / `1` | T5 文本编码器放 CPU，省 NPU 显存 | 增加 PCIe 传输，可能变慢 |
| `PRECISION` | `0` / `1` | 随机数在 CPU 生成，保证跨平台可复现 | 精度验证用，性能场景可关 |
| `OVERLAP` | `0` / `1` | FA 与 AllToAll 通信重叠（序列并行） | 需 xFuser 长上下文 attention 路径 |

## 3.3 融合算子清单（不在 P3 执行）

运行 `scripts/scan_fusion_ops.py --detail` 列出当前机器 `torch_npu` 实际导出的 `npu_*` 接口及函数签名/说明。把结果保存到 `artifacts/fusion_ops.json`，P5 选算子时用。

融合算子替换必须：
- 先由 P4 profiling 定位到热点（Top-N 算子或 AICPU 热点）
- 一次只换一个
- 保留原实现与开关
- 做精度对比

## 落地方式与注意事项

- 环境变量与入口 import 属**基础接入项**，可以直接做。
- 每加一批改动：先复测（同 P2 口径）→ 记录 before/after → 有效则固化、无效则回滚。
- 环境变量组与 `transfer_to_npu` 分别作为优化点记录，不要混在一起归因。
- 不要把"基础手段"和"深挖优化"混在一轮，否则无法归因。
