#!/usr/bin/env bash
# P3 · 环境变量候选模板（不是无脑全开的默认值）
#
# 使用方法：
#   1. 按目标场景（训练/推理、单卡/多卡）选择需要的变量；
#   2. 逐项确认当前 CANN / torch_npu 版本是否支持；
#   3. 作为一个"优化点"成组应用、成组测量，并记录到 optimization_ledger.md。
#
# 注意：
#   - PYTORCH_NPU_ALLOC_CONF 与 PYTORCH_ALLOC_CONF 同义，不能同时设置。
#   - HCCL_BUFFSIZE 只对多卡 HCCL 通信有意义，单卡推理不要开。
#   - TASK_QUEUE_ENABLE 的取值（1/2）和 MULTI_STREAM_MEMORY_REUSE 的支持情况随版本不同，
#     必须实测；不要假设所有环境都支持。

# 设备可见性（按实际物理 chip logic id 填写）
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# 下发 / 调度
# export TASK_QUEUE_ENABLE=2

# CPU 绑核
# export CPU_AFFINITY_CONF=2

# 显存分配器（二选一，不能同时设置）
# export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export PYTORCH_ALLOC_CONF=expandable_segments:True

# 多流内存复用
# export MULTI_STREAM_MEMORY_REUSE=2

# 多卡通信缓冲
# export HCCL_BUFFSIZE=800

# CPU 线程数（按机器核数调整，避免线程过多互相抢占）
# export OMP_NUM_THREADS=32
# export MKL_NUM_THREADS=32

# 算子编译缓存目录（首次编译慢时使用；跨运行复用）
# export ASCEND_OPP_CACHE=/path/to/opp_cache

# ---- 扩散 / Wan 类框架专属（仅当目标代码实现了对应开关时生效） ----
# Attention 后端：0=fused_attn_score, 1=ascend_laser_attention, 3=npu_fused_infer_attention_score
# export ALGO=0
# mindiesd fast layernorm
# export FAST_LAYERNORM=0
# Attention 子头切分（num_heads 必须可整除）
# export USE_SUB_HEAD=0
# T5 文本编码器放 CPU（省显存，增加 PCIe 传输）
# export T5_LOAD_CPU=0
# 随机数 CPU 生成，保证跨平台可复现（精度验证时用）
# export PRECISION=0
# FA 与 AllToAll 通信重叠（序列并行场景）
# export OVERLAP=0
