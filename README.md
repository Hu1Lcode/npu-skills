# npu-skills

Ascend NPU 相关技能集合（Claude/Agent skills）。

## Skills

| Skill | 说明 |
|---|---|
| [`npu-model-perf-tuning`](./npu-model-perf-tuning/SKILL.md) | Ascend NPU 上 PyTorch 模型的端到端性能调优：拆调用链 → 分段打点基线 → warm-start（transfer_to_npu + 环境变量 + 融合算子）→ torch_npu.profiler 采集分析 → 单变量迭代固化 → 精度校验 → 结构化报告。适用于训练与推理。 |
