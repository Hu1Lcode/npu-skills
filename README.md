# npu-skills

Ascend NPU 相关技能集合（Claude/Agent skills）。

## Skills

| Skill | 说明 |
|---|---|
| [`npu-model-tuning`](./npu-model-tuning/SKILL.md) | Ascend NPU 上 DiT 类生成模型与 cosmos 类 World Model 的系统化性能调优：环境/模型信息采集 → 调用链与阶段打点 → 无侵入优化（环境变量 / torch_npu 融合算子）→ 侵入性候选按收益排序交用户决策 → 多 Agent 并行单点测试与串行集成验证 → 精度校验与固定 8 节报告。覆盖推理与训练，默认启用经验库自学习。 |
