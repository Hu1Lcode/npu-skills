# 融合算子目录与替换映射（torch_npu）

来源：torch_npu 接口文档 `https://gitcode.com/Ascend/op-plugin/blob/26.2.0/docs/zh/custom_APIs/torch_npu/torch_npu_list.md`。
先在目标机运行 `scripts/scan_fusion_ops.py` 确认当前包实际导出的 `npu_*`（不同版本有增删）。

## 按族分类（常见）

| 族 | 代表接口 |
|---|---|
| Attention | `npu_fusion_attention`(v2/v3)、`npu_fused_infer_attention_score`(v2)、`npu_prompt_flash_attention`、`npu_incre_flash_attention`、`npu_multi_head_attention`、`npu_block_sparse_attention`、`npu_sparse_flash_attention` |
| Norm | `npu_rms_norm`、`npu_add_rms_norm`(+`_cast`/`_quant`/`_v2`)、`npu_rms_norm_quant`、`npu_group_norm_silu`/`npu_group_norm_swish` |
| RoPE | `npu_rotary_mul`、`npu_apply_rotary_pos_emb`、`npu_interleave_rope`、`npu_mrope`、`npu_kv_rmsnorm_rope_cache` |
| 激活 / FFN | `npu_swiglu`(含 `_quant`/`_mx_quant`/`_clipped`)、`npu_fast_gelu`、`npu_gelu`、`npu_gelu_mul`、`npu_ffn`、`npu_dequant_swiglu_quant` |
| GEMM / 量化 | `npu_quant_matmul`、`npu_linear`、`npu_bmmV2`、`npu_grouped_matmul`、`npu_transpose_batchmatmul`、`npu_weight_quant_batchmatmul` |
| MoE | `npu_moe_gating_top_k`(_softmax)、`npu_moe_init_routing`(_v2)、`npu_moe_token_permute`/`_unpermute`、`npu_moe_distribute_dispatch`/`_combine`、`npu_moe_finalize_routing` |
| 通算融合 | `npu_all_gather_base_mm`、`npu_mm_all_reduce_base`、`npu_mm_reduce_scatter_base`、`npu_all_to_all_matmul`、`npu_matmul_all_to_all` |
| Loss | `npu_cross_entropy_loss`、`npu_fused_cross_entropy_loss_with_max_sum`、`npu_softmax_cross_entropy_with_logits` |
| 其它常用 | `npu_prefetch`(权重预取)、`npu_dropout_with_add_softmax`、`npu_scaled_masked_softmax`、`npu_confusion_transpose`、`npu_format_cast`、`npu_top_k_top_p_sample`、`npu_indexing` |

## 原语 → 融合算子映射（起始）

| 模型里的原语 | 替换为 | 备注 |
|---|---|---|
| `F.scaled_dot_product_attention` / 手写 QKV attention | `npu_fusion_attention` | 训练/通用；推理可用 `npu_fused_infer_attention_score` 或 `npu_prompt/incre_flash_attention` |
| 手写 RMSNorm（常接残差 add） | `npu_rms_norm` / `npu_add_rms_norm` | 带 add 的用 add 版本融合 |
| 手写 RoPE（rotate_half / stack 拼 cos/sin） | `npu_rotary_mul` / `npu_apply_rotary_pos_emb` | interleave 布局用 `npu_interleave_rope` |
| SwiGLU / GELU / 融合 FFN | `npu_swiglu` / `npu_fast_gelu` / `npu_ffn` | 注意激活精确度差异 |
| 量化 MatMul / Grouped MM | `npu_quant_matmul` / `npu_grouped_matmul` | 需匹配量化 scale/dtype |
| MoE 路由/分发/合并 | `npu_moe_*` 系列 | 改动较大，通常属中/高侵入 |

## 替换流程与校验
1. `python scripts/scan_fusion_ops.py`（或 `--grep attn`）→ 确认目标算子在本机 torch_npu 存在（列出全部 `npu_*`）；签名/入参见官方 API 文档。
2. 先在**热点**上替换（P4 已定位的 Top-N 算子/模块），一次改一处。
3. 传参注意：mask 形状/dtype、scale、layout(NZ/ND)、head_num、输入是否为 packed。
4. **精度校验**：与替换前同输入对比，误差在容差内（参考 `ops-precision-standard`）；不达标先排查 dtype/参数而非直接放弃。
5. 复测性能，按 P5 固化/否决。融合算子通常**低~中侵入**；MoE/通算融合若需大改代码 → 列清单交用户决策。
