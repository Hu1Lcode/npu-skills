#!/usr/bin/env python3
"""扫描当前 torch_npu 实际导出的 npu_* 融合算子，并按族分组。

用法:
    python scan_fusion_ops.py                # 打印所有 npu_* 及分组
    python scan_fusion_ops.py --grep attn    # 只看名字含 attn 的
    python scan_fusion_ops.py --detail       # 附带函数签名与首行说明
    python scan_fusion_ops.py --json out.json
"""
import argparse
import inspect
import json
import sys

GROUPS = [
    ("attention", ("attention", "attn", "flash", "sdp")),
    ("norm", ("norm", "layer_norm", "rms")),
    ("rope", ("rope", "rotary")),
    ("activation_ffn", ("swiglu", "gelu", "silu", "ffn", "activation")),
    ("gemm_quant", ("matmul", "gemm", "linear", "quant", "bmm")),
    ("moe", ("moe", "expert", "gating")),
    ("collective", ("all_reduce", "all_gather", "reduce_scatter", "all_to_all", "mm_all")),
    ("loss", ("loss", "cross_entropy")),
    ("memory", ("prefetch", "format_cast", "copy", "cache")),
    ("other", ()),
]


def collect():
    import torch_npu

    names = sorted(name for name in dir(torch_npu) if name.startswith("npu_"))
    grouped = {group: [] for group, _keywords in GROUPS}
    for name in names:
        lowered = name.lower()
        matched = "other"
        for group, keywords in GROUPS:
            if keywords and any(keyword in lowered for keyword in keywords):
                matched = group
                break
        grouped[matched].append(name)
    return {"torch_npu": torch_npu, "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
            "groups": grouped}


def describe_ops(torch_npu_module, names):
    """获取算子签名与首行 docstring；不同版本/非函数对象缺失时优雅降级。"""
    details = {}
    for name in names:
        obj = getattr(torch_npu_module, name, None)
        try:
            signature = str(inspect.signature(obj))
        except (ValueError, TypeError):
            signature = "(signature unavailable)"
        doc = inspect.getdoc(obj) or ""
        doc_lines = [line.strip() for line in doc.splitlines() if line.strip()]
        prototype = ""
        for line in doc_lines:
            if name in line and "(" in line:
                prototype = line
                break
        headings = {"接口原型", "接口说明", "功能说明", "功能描述", "参数说明", "输出说明", "约束说明", "示例"}
        summary = ""
        for line in doc_lines:
            if line == prototype or line.rstrip("：:") in headings:
                continue
            summary = line
            break
        details[name] = {"signature": prototype or signature, "doc": summary}
    return details


def main():
    parser = argparse.ArgumentParser(description="扫描 torch_npu 融合算子")
    parser.add_argument("--grep", default=None, help="只显示名字包含该字符串的算子")
    parser.add_argument("--detail", action="store_true", help="显示函数签名与首行说明")
    parser.add_argument("--json", default=None, help="输出 JSON 文件")
    args = parser.parse_args()

    try:
        result = collect()
    except Exception as exc:  # noqa: BLE001
        print(f"[scan_fusion_ops] 无法导入 torch_npu 或扫描失败: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.grep:
        needle = args.grep.lower()
        filtered = {
            group: [name for name in names if needle in name.lower()]
            for group, names in result["groups"].items()
        }
        filtered = {group: names for group, names in filtered.items() if names}
    else:
        filtered = {group: names for group, names in result["groups"].items() if names}

    all_names = [name for names in filtered.values() for name in names]
    details = describe_ops(result["torch_npu"], all_names) if args.detail else {}

    print(f"torch_npu: {result['torch_npu_version']}")
    for group, names in filtered.items():
        print(f"\n== {group} ==")
        for name in names:
            if args.detail:
                info = details[name]
                print(f"  {name}: {info['signature']}")
                if info["doc"]:
                    print(f"      {info['doc']}")
            else:
                print(f"  {name}")

    if args.json:
        payload = {"torch_npu_version": result["torch_npu_version"], "groups": filtered}
        if args.detail:
            payload["details"] = details
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\n[scan_fusion_ops] wrote {args.json}")


if __name__ == "__main__":
    main()
