# 融合算子目录与替换映射（torch_npu）

来源参考：torch_npu 接口文档与目标机实际导出。**先用 `scripts/scan_fusion_ops.py --detail` 确认当前版本真正存在哪些接口，并查看函数签名与首行说明**——不同 torch_npu / CANN 版本有增删，目录里的接口不能照抄。

## 收益与精度的基本判断（先读）

融合算子通常能减少 kernel 下发、中间 Tensor 和访存，所以**一般会有性能收益，但收益大小取决于它覆盖的热点占比和原实现能消除的比例**，必须用同轮成对测量确认；不要因为"融合一定更快"就全量替换。

融合算子也可能引入**舍入误差**：常见来源是输入被降为 fp16/bf16、归一化方差在低精度下计算、累加顺序改变、激活函数使用近似实现（如 fast_gelu）。扩散模型的去噪循环会逐步累积误差，VAE decode 和长视频/高分辨率场景对误差尤其敏感。

**决策规则（每个融合算子都要执行）**：

1. 先按 P4 证据估算收益：`该算子/子链路耗时占比 × 预计消除比例`。融合算子通常都有收益，但幅度多与少不同；单点收益小于测量噪声时（经验上达不到 3%–5% 的推理 E2E / 训练 `s/step` 改善）不作为优先项，若多个小融合作用于同一热点链路，可作为一组评估。
2. 在 P1 的模型定义映射中定位具体实现 `file:line`，一次只替换一个算子/一处实现，保留原实现和开关。
3. 固定输入/seed 做输出级对比，主判据 `cosine ≥ 0.999`；同时记录 `max_abs_diff`、`rel_l2`、PSNR/SSIM 用于定位。
4. 精度不通过时按顺序尝试：保持 fp32 方差/累加的精度保持变体 → 只替换非敏感层或非敏感阶段 → 更换融合实现；仍不通过就回滚，**不能为了性能默认保留**。若收益显著但精度始终不达标，把证据（性能收益 + 误差数值）记录到优化台账/决策清单交用户判断，不自动启用。
5. 每个融合点都记录：现象、预期收益及依据、实测性能（推理 E2E + 各阶段 + DiT 单步；训练 s/step + P50/P99 + 吞吐 + 单步内阶段）、cosine/误差、最终采用或回滚结论。

## 常见融合算子的精度风险

| 融合算子 | 常见收益 | 主要精度风险 | 处理建议 |
|---|---|---|---|
| `npu_rms_norm` / `npu_add_rms_norm` | 合并 Cast/Norm/Add，减少小 kernel | fp16/bf16 输入下方差精度下降，误差逐层累积 | 关键层保持 fp32 方差；先用非敏感层验证；逐层 dump 对比 |
| `npu_add_layer_norm` | 合并残差 Add + LayerNorm + modulation | affine/modulation 的 dtype/广播语义变化 | 核对 gamma/beta 形状与 dtype；输出级 + 首层 dump |
| `npu_rotary_mul` / `npu_apply_rotary_pos_emb` | 去掉 transpose/复数运算 | float64→float32 精度损失；`rotary_mode` 配对错误 | 校验 interleave/half 模式与 cos/sin dtype；错误不报错但结果错 |
| `npu_swiglu` / `npu_fast_gelu` | 融合激活/FFN，减少访存 | 近似激活函数误差 | 与原始激活逐元素对比；图像/视频指标 + cosine |
| `npu_fusion_attention` / `npu_fused_infer_attention_score` | 融合 QKV/softmax/输出，显著减少显存与下发 | softmax scale/mask 语义、BF16 QKV 限制、累加顺序 | 固定 shape 对比 logits/输出；必要时逐层 dump attention 输出 |

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
