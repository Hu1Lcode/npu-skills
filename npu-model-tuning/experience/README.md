# 经验库（自学习）

本目录是 `npu-model-tuning` 的长期经验库。每次调优结束后，把**有证据支撑、可复用**的经验沉淀到这里；下次遇到新模型/新框架时先召回匹配项，作为候选假设，再用当前模型的 profiling 和测量结果验证。

## 三类经验

| 文件 | 内容 | 例子 |
|---|---|---|
| `optimization-methods.jsonl` | 常用有效的优化手段 | `npu_rms_norm` 融合、attention 后端替换、VAE 快路径 |
| `device-issues.jsonl` | 设备/驱动/CANN/torch_npu 常见问题与对策 | HCCL 不支持 float64、`torchrun --standalone` 挂起、HBM 预留 |
| `code-reading.jsonl` | 看代码、找调用链、识别框架结构的经验 | diffusers 的 text_encoder/DiT/VAE 位置、自研框架入口追法 |

## 经验条目格式

每行一个 JSON 对象（JSONL，UTF-8）：

```json
{
  "id": "opt-20260101-abc123",
  "category": "optimization",
  "title": "RMSNorm 融合的 fp32→bf16 精度陷阱",
  "symptom": "AICPU 占比高，RMSNorm 未融合；替换 npu_rms_norm 后输出差异变大",
  "action": "使用 npu_rms_norm，但保持 fp32 方差；不要先把 fp32 输入降为 bf16",
  "evidence": {"source_run": "perf_reports/20260101", "gain": "3%", "precision": "cosine 0.9999"},
  "confidence": "verified",
  "tags": ["rmsnorm", "fusion", "precision"],
  "applies_to": {
    "model_types": ["DiT", "VAE", "transformer"],
    "frameworks": ["diffusers", "cosmos-framework"],
    "hardware": ["Ascend910"],
    "software": {"cann": ">=8.0", "torch_npu": ">=2.1"}
  },
  "caveats": ["npu_rms_norm 通常只接受 fp16/bf16"],
  "source_run": "perf_reports/20260101",
  "created_at": "2026-01-01",
  "last_verified": "2026-01-01",
  "usage": {"applied": 0, "success": 0, "failed": 0}
}
```

## 置信度

- `verified`：在当前或同类环境实测过，有明确性能/精度数据。
- `observed`：实践中观察到，但没有严格成对测量。
- `hypothesis`：推测，只作为待验证候选。

**规则**：`hypothesis` 不能直接写进报告结论；`observed` 只能作为候选；`verified` 也要写清适用环境与前提。

检索排序会综合：关键词命中、模型/框架匹配、置信度、历史使用成功率（`usage.success / applied`）。复用后及时 `update`，越用越准。

## 使用命令

```bash
# P0 召回：按模型/框架/关键词检索
python scripts/experience.py search --query "AICPU RMSNorm" --model-type DiT --framework diffusers --json agent/experience_recall.json

# P8 沉淀
python scripts/experience.py add --category optimization \
  --title "..." --symptom "..." --action "..." \
  --confidence observed --tags "fusion,AICPU" \
  --model-types "DiT,VAE" --frameworks "diffusers" \
  --evidence '{"gain":"4.1%","precision":"cosine 0.9999"}' \
  --source-run perf_reports/20260101

# 记录复用结果
python scripts/experience.py update --id opt-20260101-abc123 --result success --note "在 Qwen-Image-Edit 上复现"
```

默认库目录是本目录；设置 `NPU_TUNE_EXPERIENCE_DIR=/path/to/experience` 可改用外部库（适合多项目共享或不想随 skill 分发时）。

## 质量要求

1. **必须有证据**：`source_run`、基线/优化后数据或误差指标；没有证据就标 `hypothesis`。
2. **必须写适用范围**：模型类型、框架、硬件、CANN/torch_npu 版本；不要写“所有模型都有效”。
3. **先去重再新增**：先 `search`，相似条目用 `update` 补充证据，不要重复堆条目。
4. **失败也记录**：某条经验在新环境失败，用 `update --result failed` 降级并写清上下文。
5. **不写敏感信息**：不记录权重路径、客户数据、密钥、私有代码片段；只记录抽象后的经验。
6. **经验是假设**：召回后仍要按当前模型的 profiling 证据排序和验证，不能直接照搬。
