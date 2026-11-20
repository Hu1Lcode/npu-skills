# 融合算子目录与替换映射（torch_npu）

来源参考：torch_npu 接口文档与目标机实际导出。**先用 `scripts/scan_fusion_ops.py --detail` 确认当前版本真正存在哪些接口，并查看函数签名与首行说明**——不同 torch_npu / CANN 版本有增删，目录里的接口不能照抄。

## 按族分类（常见）

| 族 | 代表接口 |
|---|---|
| Attention | `npu_fusion_attention`（v2/v3）、`npu_fused_infer_attention_score`（v2）、`npu_prompt_flash_attention`、`npu_incre_flash_attention`、`npu_multi_head_attention`、`npu_block_sparse_attention` |
| Norm | `npu_rms_norm`、`npu_add_rms_norm`（+ `_cast` / `_quant` / `_v2`）、`npu_rms_norm_quant`、`npu_group_norm_silu` / `npu_group_norm_swish` |
| Add + Norm | `npu_add_layer_norm`（可带 `additional_output`）、`npu_add_rms_norm` |
| RoPE | `npu_rotary_mul`、`npu_apply_rotary_pos_emb`、`npu_interleave_rope`、`npu_mrope`、`npu_kv_rmsnorm_rope_cache` |
| 激活 / FFN | `npu_swiglu`、`npu_fast_gelu`、`npu_gelu`、`npu_gelu_mul`、`npu_ffn` |
| GEMM / 量化 | `npu_quant_matmul`、`npu_linear`、`npu_bmmV2`、`npu_grouped_matmul`、`npu_weight_quant_batchmatmul` |
| MoE | `npu_moe_gating_top_k`、`npu_moe_init_routing`、`npu_moe_token_permute` / `_unpermute`、`npu_moe_distribute_dispatch` / `_combine`、`npu_moe_finalize_routing` |
| 通算融合 | `npu_all_gather_base_mm`、`npu_mm_all_reduce_base`、`npu_mm_reduce_scatter_base` |
| Loss | `npu_cross_entropy_loss`、`npu_fused_cross_entropy_loss_with_max_sum` |
| 其它 | `npu_prefetch`、`npu_dropout_with_add_softmax`、`npu_scaled_masked_softmax`、`npu_confusion_transpose`、`npu_format_cast`、`npu_top_k_top_p_sample` |

## 原语 → 融合算子映射（起始）

| 模型里的原语 | 替换为 | 备注 |
|---|---|---|
| `F.scaled_dot_product_attention` / 手写 QKV attention | `npu_fusion_attention` | 训练/通用；推理可试 `npu_fused_infer_attention_score`（BSND 布局）或 `npu_prompt_flash_attention` |
| 手写 RMSNorm | `npu_rms_norm` | **精度注意**：`npu_rms_norm` 通常只接受 fp16/bf16 输入；diffusers 等分支若把 fp32 输入降为 bf16 再算方差，会产生 0.4% 级误差并逐层累积。关键模型需强制 fp32 方差或改用等价 fp32 实现 |
| 残差 + LayerNorm + modulation | `npu_add_layer_norm` | 可用 `additional_output=True` 一次完成 add + norm + 输出；注意 batch>1 或静态 affine 时的回退 |
| 手写 RoPE（`rotate_half` / 拼接 cos/sin） | `npu_rotary_mul` / `npu_apply_rotary_pos_emb` / `npu_interleave_rope` | 布局需匹配：BSND `[B,S,H,D]` + `1S1D` cos/sin 可直接走 `npu_rotary_mul`，`rotary_mode="interleave"`；避免 BNSD round-trip transpose |
| SwiGLU / GELU / FFN | `npu_swiglu` / `npu_fast_gelu` / `npu_ffn` | 注意激活精确度差异 |
| 量化 MatMul / Grouped MM | `npu_quant_matmul` / `npu_grouped_matmul` | 需匹配量化 scale/dtype；量化属于中/高侵入或需用户明确授权 |

## MindIE SD（mindiesd）补充（常见于 DiT / Wan 类推理）

| 原语 | mindiesd 接口 | 说明 |
|---|---|---|
| 手写复数乘法 RoPE（float64/complex128） | `mindiesd.rotary_position_embedding(x, cos, sin, rotated_mode="rotated_interleaved", fused=True)` | 需在模型 forward 预计算 cos/sin；`freqs_list` 按分辨率缓存并在生成结束后清理 |
| LayerNorm（带 `.float()`） | `mindiesd.fast_layernorm`（`FAST_LAYERNORM=1`） | 框架支持时才可用；先做精度对比 |
| Flash Attention 多后端 | `mindiesd.attention_forward(q, k, v, op_type=...)` | `fused_attn_score` / `ascend_laser_attention` / `npu_fused_infer_attention_score`，只有 self-attention 支持后两者 |
| 量化加载 | `mindiesd.quantize(model, quant_des_path, use_nz)` | W8A8 动态量化产物；NZ layout 优先 |

## 替换流程与校验

1. `python scripts/scan_fusion_ops.py --grep attn --detail` → 确认目标算子在本机存在并查看函数签名；再查官方 API 文档确认 dtype、layout 约束。
2. 只在 P4 已定位的热点上替换；一次改一处，保留原实现作为回退。
3. 传参注意：mask 形状/dtype、scale、layout（NZ/ND）、head_num、输入是否 packed、是否需要 contiguous。
4. **精度校验**：与替换前同输入对比，误差在容差内；不达标先排查 dtype/参数/布局，而不是直接放弃。
5. 复测性能，按 P5 固化/否决。标准融合算子替换属**无侵入**自动项；若需要改接口或大范围重写，就不再是融合算子替换，按有侵入列入 P5b 决策清单。

## Ascend 特有坑位（替换时检查）

- `npu_fused_infer_attention_score` 常见要求：BSND 布局、特定的 head dim / dtype / mask 形状；不满足会报错或回退。
- `npu_rms_norm` 的 fp16/bf16 输入限制是最常见的精度陷阱；参考 `precision-validation.md` 的逐层 dump 方法定位。
- `npu_add_layer_norm` 的 `gamma` / `beta` 形状需与通道维匹配；per-channel modulation 场景要确认广播语义。
- `npu_rotary_mul` 的 `rotary_mode` 必须与模型原本的配对方式一致（interleave vs half），否则结果错误但不报错。
- 融合算子不改变数学语义时也可能因累加顺序产生 BF16 舍入差异，必须实测。
