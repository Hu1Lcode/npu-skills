# 优化点生效矩阵（P7.5）

> 每个已采纳优化点一行。证据必须来自 `prof/baseline/` 与 `prof/final/`，且两份 profile 的 profiler 配置、rank、warmup、active/denoise steps、输入规格一致。profiling run 不作为性能数字。

## 采集口径

- baseline：`prof/baseline/`（P2 采集，P3 O2 之前，`NPU_TUNE_PROBE=0`）
- final：`prof/final/`（P7.5 采集，`baseline_commit + all.patch` 干净工作区 + 最终环境变量，`NPU_TUNE_PROBE=0`）
- profiler：Level1 + PipeUtilization；rank=`<N>`；warmup=`<N>`；active/denoise steps=`<N>`；输入=`<shape/dtype/batch>`

## 生效矩阵

| 优化点 ID | 优化点 | 类别 | 预期 profiling 签名 | baseline 证据 | final 证据 | 逐点性能收益 | 状态 | 备注 |
|---|---|---|---|---|---|---|---|---|
| O2 | <环境变量组> | env | <Free 下降 / kernel 间隔缩短 / 下发减少> | <...> | <...> | <推理 E2E / DiT 单步 / 训练 s/step> | verified | |
| O3 | <融合算子> | fusion | <原 kernel 消失 / 新 fused kernel 出现 / AICPU 占比下降> | <...> | <...> | <...> | mechanism_verified | |
| O4 | <其他> | other | <...> | <...> | <...> | <...> | time_only | 附 P7 同卡成对结果 |

## 状态定义

| 状态 | 判定 | 是否计入有效优化 |
|---|---|---|
| `verified` | 预期 profiling 签名出现，且对应指标改善 | 是 |
| `mechanism_verified` | 预期签名出现，但整体指标被其他变化掩盖；结合 P7 成对性能证据 | 是（须注明） |
| `time_only` | profile 无可见机制差异，但有 P7 同卡 control/candidate 成对性能收益 | 是（仅限有性能证据） |
| `failed` | 预期签名未出现且无 P7 性能证据；或 patch 未命中/被 fallback | 否，必须回查并回滚 |

## 结论

- 通过：<N> 项；mechanism_verified：<N> 项；time_only：<N> 项；failed：<N> 项
- 失败项处理：<回滚 / 重采 / 待决策>
- 整体瓶颈迁移：<baseline 主瓶颈 → final 主瓶颈>
