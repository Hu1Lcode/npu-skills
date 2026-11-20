# 诊断手册：从瓶颈证据到优化方向

先用 `profile_summary.py` 的 Task Type 占比 + P2 分段耗时锁定瓶颈类型，再按本表找方向。每项都优先给**无侵入**手段（环境变量、融合算子）；其余改动只列清单，按预期收益从高到低交用户决策。

## 先做一步量化分型（step_trace_time.csv）

在翻下面的症状表之前，先按 `profiling-analysis.md` 的流程算三个占比：

| 判定 | 阈值 | 先看 |
|---|---|---|
| **Hostbound** | `Free > 20%` | C（Host-bound）、D（数据加载）、B（首步编译） |
| **Computing** | `Computing > 85%` | F（memory-bound）、A（AICPU）、扩散模型专项 |
| **Communication** | `Communication > 10%` | E（通信）、慢卡检测 |
| 无明显瓶颈 | 都不满足 | 按 P2 分段找局部热点；平均占比会掩盖问题 |

阈值是启发式：Hostbound 优先判，其次 Computing，最后 Communication。确认分型后再用下面的证据表细化。

## 症状 → 证据 → 修复对照表

### A. AICPU 占比高（典型 30%–80%）
- **证据**：`kernel_details.csv` 中 `Task Type=AICPU` 的 kernel 总时长远超 AICore/AIV。
- **原因**：算子没落到 AI Core——常见于自定义 Python/numpy 逻辑、float64 运算、torch_npu 未实现的算子落到 CPU 侧、复杂 host 侧循环、动态 shape 导致无法编译。
- **修复（只有环境变量/融合算子自动实施，其余列清单）**：
  1. float64 → float32（NPU 对 float64 支持差；RoPE / 位置编码里的 float64 尤其致命）——代码改动，列清单。
  2. 把热点原语换成 `torch_npu` 融合算子（attention/RMSNorm/RoPE/FFN，见 `fusion-op-catalog.md`）——无侵入自动项；先按 P1 模型定义映射定位到具体 `file:line`，并注意融合可能引入舍入误差，必须做精度对比。
  3. 把逐元素 Python 循环改为向量化 torch 操作——代码改动，列清单。
  4. 若某算子确实未实现，考虑自定义算子或换等价表达——有侵入，列清单。
- **案例**：OpenWAM AICPU ~73.3% → 换融合算子 + float64→float32 后 step 15.79→3.02 s/step。

### B. 首步/首次推理特别慢
- **证据**：第 1 步远慢于后续；推理 E2E 或训练 `s/step` 被首步拉高。
- **原因**：编译、内存池分配、算子首次加载（aclnn init）、图构建。
- **修复**：warmup 后再计时；开启算子编译缓存（`ASCEND_OPP_CACHE`）；`MULTI_STREAM_MEMORY_REUSE=2`；必要时预热一次完整 pipeline。

### C. Host-bound / 下发开销大（算子多、单步 time 碎片化、CPU 忙）
- **证据**：host 侧 CPU 占用高；kernel 都很小但数量极多；profiler 显示大量小 task 间隙、`enqueue/dequeue` 多。
- **修复**：`TASK_QUEUE_ENABLE`（1 或 2，按版本实测）、`CPU_AFFINITY_CONF=2`（环境变量，自动）；算子融合减少数量（融合算子，自动）；`torch.compile` / torchair / 图模式（有侵入，列清单）。
- **注意**：图模式/编译属有侵入，先列清单验证收益与兼容性。

### D. 数据加载 / 预处理拖慢（训练 step 有空泡）
- **证据**：dataloader 段耗时高；设备利用率周期性掉 0；step time 抖动大；worker 处理时间低于空闲间隔。
- **修复**：`num_workers`↑、`pin_memory`、prefetch；把 CPU 预处理搬到 device 或批量化；缓存小数据集到 `/dev/shm`；检查 worker 是否绑定到正确设备。

### E. 通信瓶颈（多卡）
- **证据**：Task Type=HCCL / `hcom_*` 占比高；扩卡不线性；通信与计算未重叠。
- **修复**：`HCCL_BUFFSIZE`↑；梯度累积/更大 batch 提升计算通信比；通算融合（`npu_all_gather_base_mm` / `npu_mm_all_reduce_base`）；并行策略调整（有侵入，列清单）；float64 张量改 float32（HCCL 不支持 float64）。

### F. Memory-bound / 带宽受限（AICore 高但算力利用率低）
- **证据**：MBU 高、算力利用率低；大 elementwise / Reduce 类算子多；HBM 带宽接近峰值。
- **修复**：融合减少访存（如 `npu_add_rms_norm` 融合 add+norm）；减少中间张量/原地操作；优化 layout 转换；算子级 tiling（需算子开发）。

### G. 显存碎片 / OOM 边缘
- **证据**：偶发 OOM、显存利用率高但有空洞、allocated 与 reserved 差距大。
- **修复**：`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`；减小 batch / 激活重算（梯度检查点）；`MULTI_STREAM_MEMORY_REUSE`（如支持）；检查是否有未释放的缓存。

### H. 大量 format cast / memcpy
- **证据**：memcpy / SDMA / PCIe / format_cast 类 task 占比高。
- **修复**：统一 layout（NZ/ND）；减少不必要的 `.contiguous()` / `.to()` / `transpose`；`npu_format_cast` 一次到位；中间张量设备常驻，避免反复 H2D/D2H。

### I. 小 batch 利用率低
- **证据**：AICore 利用率低、step time 与 batch 不成比例。
- **修复**：增大 batch/并行度（受显存限制）；融合小算子；图模式减少调度；推理场景考虑动态 batching。

## 扩散模型专项

| 现象 | 可能原因 | 优先检查 |
|---|---|---|
| DiT denoise 占比高 | attention / FFN 未用融合算子；步数多 | Attention 后端、RMSNorm/RoPE/Add+LayerNorm 融合、FFN 融合 |
| VAE decode 占比高 | 逐元素操作多、layout 转换、未融合 | VAE 逐元素融合、减少 transpose/contiguous、输出 pinned memory |
| 输入预处理占比高 | PIL→numpy→tensor 逐帧转换、每 chunk 回传 | 输入预转换、设备侧缩略图、批量 H2D |
| 多 chunk 流水线有气泡 | worker 绑定错误、队列等待 | worker 设备绑定、stream 重叠、队列等待分析 |
| 首帧/末帧特别慢 | 编译/内存分配/边界处理 | warmup、缓存复用、边界逻辑检查 |

## 环境变量速查（按瓶颈）

| 瓶颈 | 优先尝试 |
|---|---|
| 下发 / host-bound | `TASK_QUEUE_ENABLE`（1/2 实测）、`CPU_AFFINITY_CONF=2` |
| 通信 | `HCCL_BUFFSIZE`、并行策略 |
| 内存 / 碎片 | `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`、`MULTI_STREAM_MEMORY_REUSE`（如支持） |
| AICPU / 算子缺失 | 融合算子 + float32 化（非 env） |

> 官方环境变量分类文档（按 CANN/torch_npu 版本）可参考 Ascend/pytorch 的 `docs/zh/api/environment_variable/env_variable_list.md`。一次只加一个（或一组），复测归因。

## 两个专项核对

- **Roofline（算力是否打满）**：对 `aic_mac_ratio ≥ 0.80` 的算子，用 `理论耗时 = max(FLOPs/峰值算力, 访存量/峰值带宽)` 与实测对比。效率 80%–120% 不优先优化单 kernel；40%–70% 建议优化；<40% 需要优化；>150% 说明模型高估。必须用真实硬件峰值，查不到就问用户。详见 `profiling-analysis.md`。
- **慢卡检测**：多卡扩展不线性时，先向用户申请补采全 rank，再做 `msprof-analyze cluster -d ./profiling_data -m slow_rank -o ./result`。`slowAffectCount` 最大值 > 20 时看 Z-score（>0.5 提示、>2 判定）；否则转 plog 查 HCCL 异常。详见 `profiling-analysis.md`。

## 决定"无侵入优化 vs 列侵入清单"

- 能靠 env 解决的（下发、绑核、通信缓冲、内存）→ 优先 env（无侵入）。
- 环境无效、热点明确在算子级 → 融合算子替换（无侵入，但必须保留原实现与开关）。
- 需要重写模块/算法/并行策略、图模式、量化、缓存/稀疏注意力、多流等 → 有侵入，仅列清单，按预期收益降序交用户决策。
