# 起步环境变量（P3 warm-start 默认值）
# 用法：source ./assets/env.default.sh  （或复制到目标项目的 optimizations/env.applied.sh）
# 说明：这 5 个是"经验有效"的起步项；profiling 定位到具体瓶颈后，再按
#       references/diagnosis-playbook.md「环境变量速查」逐项加，务必一次一个、复测归因。

# --- 通信缓冲（多卡）---
export HCCL_BUFFSIZE=800

# --- CPU 绑核，避免跨 NUMA（host-bound / 下发开销）---
export CPU_AFFINITY_CONF=2

# --- task_queue 下发队列优化（算子多、host-bound 时明显）---
export TASK_QUEUE_ENABLE=2

# --- 多流内存复用（省显存、减分配抖动）---
export MULTI_STREAM_MEMORY_REUSE=2

# --- 缓存分配器扩展段（减显存碎片；注意与 PYTORCH_ALLOC_CONF 同义，不能同设）---
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# --- 入口代码必须加（不是环境变量，见 references/warmstart-defaults.md）---
#   import torch_npu
#   from torch_npu.contrib import transfer_to_npu
# 或用环境变量等价开启：
#   export TORCH_TRANSFER_TO_NPU=1
