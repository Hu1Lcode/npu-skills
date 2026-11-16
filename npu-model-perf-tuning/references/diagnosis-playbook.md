# 诊断手册：从瓶颈证据到修复方案

先用 `kernel_summary.py` 的 Task Type 占比 + P2 分段耗时锁定瓶颈类型，再按本表找修复方向。每项都尽量先给**低侵入**手段。

## 症状 → 证据 → 修复对照表

### A. AICPU 占比高（典型 30%~80%）
- **证据**：`kernel_details.csv` 中 Task Type=AICPU 的 kernel 总时长远超 AICore/AIV。
- **原因**：算子没落到 AI Core——常见于自定义 python/numpy 逻辑、float64 运算、torch_npu 未实现的算子落到 CPU 侧、复杂的 host 侧循环。
- **修复（低→中侵入）**：
  1. 把热点原语换成 `torch_npu` 融合算子（attention/RMSNorm/RoPE/FFN，见 `fusion-op-catalog.md`）。
  2. float64 → float32（NPU 不支持 float64 高效执行；归约张量尤其明显）。
  3. 替换/改写 NPU 未实现的算子和逐元素 python 循环为向量化 torch 操作。
  4. 参考真实案例：OpenWAM AICPU ~73.3% → 换融合算子后 step 15.79→3.02 s/step（`examples/openwam-case.md`）。

### B. 首步/首次推理特别慢
- **证据**：第 1 步远慢于后续；E2E 被首步拉高。
- **原因**：编译、内存池分配、算子首次加载（aclnn init）、图构建。
- **修复**：warmup 后再计时；开启算子编译缓存（`ASCEND_OPP_CACHE`/`TASK_QUEUE_ENABLE`）；多流内存复用 `MULTI_STREAM_MEMORY_REUSE=2`。

### C. Host-bound / 下发开销大（算子多、单步 time 碎片化、CPU 忙）
- **证据**：host 侧 CPU 占用高；kernel 都很小但数量极多；profiler 显示大量小 task 间隙。
- **修复**：`TASK_QUEUE_ENABLE=2`（下发队列优化）；算子融合减少数量；`torch.compile`/图模式；CPU 绑核 `CPU_AFFINITY_CONF=2` 避免跨 NUMA。

### D. 数据加载 / 预处理拖慢（训练 step 有空泡）
- **证据**：dataloader 段耗时高；GPU/NPU 利用率周期性掉 0；step time 抖动大。
- **修复**：`num_workers`↑、`pin_memory`、预取；把 CPU 预处理搬到 device 或批量化；`npu_prefetch` 预取权重。

### E. 通信瓶颈（多卡）
- **证据**：Task Type=HCCL 占比高；扩卡不线性；通信与计算未重叠。
- **修复**：`HCCL_BUFFSIZE`↑；梯度累积/更大 batch 提升计算通信比；通算融合（`npu_all_gather_base_mm`/`npu_mm_all_reduce_base`）；并行策略调整（见 `model-infer-parallel-analysis`）。float64 张量会退回 HCCL 不支持（→float32）。

### F. Memory-bound / 带宽受限（AICore 高但算力利用率低）
- **证据**：MBU 高、算力利用率低；大 elementwise/Reduce 类算子。
- **修复**：融合减少访存（如 `npu_add_rms_norm` 融合 add+norm）；`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 减碎片；减少中间张量/原地操作；合理 UB 切分（算子级）。

### G. 显存碎片 / OOM 边缘
- **证据**：偶发 OOM、显存利用率高但有空洞。
- **修复**：`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`；减小 batch/激活重算；多流内存复用。

### H. 大量 format cast / memcpy
- **证据**：memcpy/format_cast 类 task 占比高。
- **修复**：统一 layout（NZ/ND）；减少不必要的 `.contiguous()`/`.to()`/`transpose`；`npu_format_cast` 一次到位。

### I. 小 batch 利用率低
- **证据**：AICore 利用率低、step time 与 batch 不成比例。
- **修复**：增大 batch/并行度；融合小算子；图模式减少调度。

## 环境变量速查（按瓶颈）
> 官方分类文档（设备初始化/算子执行/内存/通信/性能优化…）：
> `https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/zh/api/environment_variable/env_variable_list.md`
> 一次只加一个、复测归因。除起步 5 个外，出现对应瓶颈时再去文档里按分类找具体变量。

| 瓶颈 | 优先尝试 |
|---|---|
| 下发/host-bound | `TASK_QUEUE_ENABLE=2`、`CPU_AFFINITY_CONF=2` |
| 通信 | `HCCL_BUFFSIZE`、并行策略 |
| 内存/碎片 | `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`、`MULTI_STREAM_MEMORY_REUSE=2` |
| AICPU/算子缺失 | 融合算子 + float32 化（非 env） |

## 决定"改代码 vs 改环境变量"
- 能靠 env 解决的（下发、绑核、通信缓冲、内存）→ 优先 env（低侵入）。
- 环境无效、热点明确在算子级 → 融合算子替换（低~中侵入）。
- 需要重写模块/算法 → 高侵入，仅列清单交用户。
