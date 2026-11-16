#!/usr/bin/env python3
"""扫描当前 torch_npu 包里可用的融合算子 / 自定义 API。

用途（P3 融合算子替换时）：
  1. 确认某个融合算子在本机 torch_npu 版本是否存在（版本差异很大，务必先扫再改）。
  2. 列出所有 npu_* API，按关键字过滤，发现可替换的热点原语。

用法:
    python scan_fusion_ops.py                 # 检查已知融合算子 + 列出全部 npu_*
    python scan_fusion_ops.py --grep attn     # 只看名字含 attn 的
    python scan_fusion_ops.py --grep rope
"""
import argparse
import sys

# 常见融合算子/自定义 API（不同版本命名与可用性不同，仅作检查起点）
KNOWN = [
    # attention
    "npu_fusion_attention", "npu_fusion_attention_v2", "npu_fusion_attention_v3",
    "npu_fused_infer_attention_score", "npu_prompt_flash_attention",
    "npu_incre_flash_attention", "npu_paged_attention",
    # norm / ffn
    "npu_rms_norm", "npu_add_rms_norm", "npu_rms_norm_backward",
    "npu_swiglu", "npu_fast_gelu", "npu_ffn",
    # rope
    "npu_rotary_mul", "npu_apply_rotary_pos_emb",
    "npu_apply_rotary_pos_emb_", "npu_interleave_rope", "npu_mrope",
    # matmul / moe
    "npu_quant_matmul", "npu_grouped_matmul", "npu_moe_gating_top_k",
    "npu_moe_init_routing", "npu_moe_token_permute", "npu_moe_finalize_routing",
    # communication fuse
    "npu_all_gather_base_mm", "npu_mm_all_reduce_base",
    # misc
    "npu_prefetch", "npu_format_cast", "npu_scaled_masked_softmax",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grep", default=None, help="只列出名字包含该子串的 API")
    a = ap.parse_args()

    try:
        import torch_npu
    except Exception as e:  # noqa: BLE001
        print(f"[scan_fusion_ops] 无法 import torch_npu: {e}", file=sys.stderr)
        sys.exit(1)

    ver = getattr(torch_npu, "__version__", "unknown")
    print(f"torch_npu 版本: {ver}\n")

    print("== 已知融合算子可用性 ==")
    print(f"{'api':<40}{'available':<10}")
    for name in KNOWN:
        ok = hasattr(torch_npu, name)
        print(f"{name:<40}{'YES' if ok else 'no':<10}")

    print("\n== 当前包内全部 npu_* API ==")
    names = sorted(n for n in dir(torch_npu) if n.startswith("npu_"))
    if a.grep:
        names = [n for n in names if a.grep.lower() in n.lower()]
    for n in names:
        print(f"  {n}")
    print(f"\n共 {len(names)} 个（过滤: {a.grep or '无'}）")
    print("\n提示：替换前先小样例验证该 API 的入参/精度；热点的 attention/norm/rope/ffn 优先。")


if __name__ == "__main__":
    main()
