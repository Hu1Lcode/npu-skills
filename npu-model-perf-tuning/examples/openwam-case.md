# 实战案例：OpenWAM NPU 适配与优化（真实数据）

来源：`C:\workspace\perf\OpenWAM-NPU-适配与优化复现指南.md`、`OpenWAM_性能总结.md`。作为"从 AICPU 主导到融合算子替换"的完整范例。

## 背景
- 硬件：Ascend 910，16 卡。
- 软件：CANN 9.2.0，torch 2.13 + torch_npu 2.13。
- 任务：训练（视频/世界模型类），batch=128，60 step 实测。

## 诊断（profiling 关键证据）
- 优化前：**AICPU 占比 ~73.3%**——大量算子没落到 AI Core。
- AICore 利用率仅 14.45%。

## 优化手段（按低→中侵入顺序，均为有效且固化）
1. `import torch_npu` + `from torch_npu.contrib import transfer_to_npu`（入口接入）。
2. 环境变量：`ASCEND_RT_VISIBLE_DEVICES`、`TASK_QUEUE_ENABLE`、`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 等。
3. torch.cuda.* → torch_npu.npu.*（设备 API 对齐）。
4. **float64 → float32**：NPU 不支持 float64 高效执行；RoPE / sinusoidal 位置编码的 float64 尤其致命。
5. **HCCL 不支持 float64** → 归约张量改 float32。
6. **融合算子替换**：`npu_fusion_attention`、`npu_rms_norm`，消除 attention/norm 的 AICPU 热点。
7. `torchrun --standalone` 会挂 → 改静态 rendezvous。

## 结果
| 指标 | 优化前 | 优化后 |
|---|---|---|
| step time | 15.79 s/step | **3.02 s/step**（≈5.2×） |
| AICore 利用率 | 14.45% | **72.73%** |
| 16 卡 bs128 60step 实测 | - | **2.64 s/step，48.43 samples/s** |

## 可复用的经验（写进诊断/优化循环）
- AICPU 占比是第一优先信号；先看是否有 float64、是否有未落地算子的 python 逻辑，再找融合算子。
- float64 在 NPU 上是"隐形杀手"：既慢，又可能让 HCCL 直接不支持。
- 融合算子逐个替换、逐个复测，固化后继续，能稳定累积收益（正是单变量循环）。
- 设备 API 与启动方式（rendezvous）这类"非算子"问题也会极大影响可跑性/性能，属 P0/P3 范畴。
