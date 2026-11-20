#!/usr/bin/env python3
"""经验库：检索、沉淀、更新调优经验。

默认经验库目录：<skill_root>/experience/（可用环境变量 NPU_TUNE_EXPERIENCE_DIR 覆盖）。
三类经验：
  - optimization-methods.jsonl : 常用有效的优化手段
  - device-issues.jsonl         : 设备/驱动/CANN/torch_npu 常见问题与对策
  - code-reading.jsonl          : 看代码、找调用链、识别框架结构的经验

用法:
    python experience.py search --query "AICPU RMSNorm" --model-type DiT --framework diffusers
    python experience.py search --category device --query "HCCL hang" --json out.json
    python experience.py add --category optimization --title "..." --symptom "..." --action "..." \
        --confidence observed --tags "fusion,AICPU" --model-types "DiT,VAE" --frameworks "diffusers"
    python experience.py update --id opt-20260101-abc123 --result success --note "在 Qwen-Image 上复现"
    python experience.py stats

经验不是教条：命中只作为候选假设，必须结合当前模型的 profiling/测量结果验证。
"""
import argparse
import json
import os
import sys
import uuid
from datetime import date

SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STORE = os.environ.get("NPU_TUNE_EXPERIENCE_DIR") or os.path.join(SKILL_ROOT, "experience")
CATEGORY_FILES = {
    "optimization": "optimization-methods.jsonl",
    "device": "device-issues.jsonl",
    "code": "code-reading.jsonl",
}
CATEGORY_ALIASES = {
    "optimization_methods": "optimization",
    "optimization-methods": "optimization",
    "device_issue": "device",
    "device-issues": "device",
    "code_reading": "code",
    "code-reading": "code",
}
CONFIDENCES = ("verified", "observed", "hypothesis")


def canonical_category(value, fallback=None):
    value = CATEGORY_ALIASES.get(str(value), str(value))
    if value in CATEGORY_FILES:
        return value
    if fallback in CATEGORY_FILES:
        return fallback
    raise ValueError(f"未知经验类别: {value}")


def store_path(store, category):
    return os.path.join(store, CATEGORY_FILES[category])


def load_entries(store, category=None):
    entries = []
    categories = [category] if category else list(CATEGORY_FILES)
    for cat in categories:
        path = store_path(store, cat)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[experience] 警告：{path}:{line_number} JSON 解析失败：{exc}", file=sys.stderr)
                    continue
                entry["category"] = canonical_category(entry.get("category", cat), fallback=cat)
                entries.append(entry)
    return entries


def append_entry(store, category, entry):
    os.makedirs(store, exist_ok=True)
    path = store_path(store, category)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def rewrite_category(store, category, entries):
    os.makedirs(store, exist_ok=True)
    path = store_path(store, category)
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def entry_text(entry):
    parts = [
        str(entry.get("title", "")),
        str(entry.get("symptom", "")),
        str(entry.get("action", "")),
        str(entry.get("notes", "")),
        " ".join(str(item) for item in entry.get("tags", [])),
        " ".join(str(item) for item in entry.get("caveats", [])),
        " ".join(str(item) for item in entry.get("error_patterns", [])),
    ]
    return " ".join(parts).lower()


def score_entry(entry, terms, model_type=None, framework=None):
    text = entry_text(entry)
    score = 0.0
    for term in terms:
        if term:
            score += text.count(term.lower()) * 2.0
    applies = entry.get("applies_to", {}) or {}
    model_types = [str(item).lower() for item in applies.get("model_types", [])]
    frameworks = [str(item).lower() for item in applies.get("frameworks", [])]
    if model_type:
        needle = model_type.lower()
        if needle in model_types:
            score += 5.0
        elif "any" in model_types:
            score += 2.0
        elif model_types:
            score -= 1.0
    if framework:
        needle = framework.lower()
        if needle in frameworks:
            score += 5.0
        elif "any" in frameworks:
            score += 2.0
        elif frameworks:
            score -= 1.0
    confidence = entry.get("confidence", "observed")
    score += {"verified": 2.0, "observed": 0.5, "hypothesis": 0.0}.get(confidence, 0.0)
    usage = entry.get("usage", {}) or {}
    applied = int(usage.get("applied", 0) or 0)
    if applied:
        success = int(usage.get("success", 0) or 0)
        failed = int(usage.get("failed", 0) or 0)
        score += max(-1.5, min(1.5, 1.5 * (success - failed) / applied))
    return score


def format_entry(entry, index):
    applies = entry.get("applies_to", {}) or {}
    lines = [
        f"[{index}] {entry.get('title', '<no title>')}  ({entry.get('id', '?')})",
        f"    category={entry.get('category')} confidence={entry.get('confidence')} last_verified={entry.get('last_verified')}",
    ]
    if applies:
        lines.append(
            "    applies_to: model=%s framework=%s"
            % (",".join(applies.get("model_types", [])) or "*", ",".join(applies.get("frameworks", [])) or "*")
        )
    if entry.get("symptom"):
        lines.append(f"    symptom: {entry['symptom']}")
    if entry.get("action"):
        lines.append(f"    action : {entry['action']}")
    if entry.get("caveats"):
        lines.append(f"    caveats: {'; '.join(str(item) for item in entry['caveats'])}")
    if entry.get("evidence"):
        lines.append(f"    evidence: {json.dumps(entry['evidence'], ensure_ascii=False)}")
    return "\n".join(lines)


def cmd_search(args):
    entries = load_entries(args.store, args.category)
    terms = [term for term in (args.query or "").split() if term]
    scored = []
    for entry in entries:
        if args.confidence and entry.get("confidence") not in args.confidence.split(","):
            continue
        if args.tag:
            tags = {str(item).lower() for item in entry.get("tags", [])}
            if not all(tag.strip().lower() in tags for tag in args.tag.split(",")):
                continue
        score = score_entry(entry, terms, args.model_type, args.framework)
        if terms and score <= 0 and not args.all:
            continue
        scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("last_verified", "")), str(item[1].get("title", ""))))
    results = [entry for _score, entry in scored[: args.limit]]
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"[experience] {len(results)} 条经验 -> {args.json}")
    for index, entry in enumerate(results, 1):
        print(format_entry(entry, index))
        print()
    if not results:
        print("[experience] 没有匹配的经验；按当前模型继续 profiling 即可。", file=sys.stderr)
    return 0


def build_entry(args):
    today = date.today().isoformat()
    prefix = {"optimization": "opt", "device": "dev", "code": "code"}[args.category]
    entry = {
        "id": f"{prefix}-{today.replace('-', '')}-{uuid.uuid4().hex[:6]}",
        "category": args.category,
        "title": args.title,
        "symptom": args.symptom,
        "action": args.action,
        "evidence": json.loads(args.evidence) if args.evidence else {},
        "confidence": args.confidence,
        "tags": [item.strip() for item in (args.tags or "").split(",") if item.strip()],
        "applies_to": {
            "model_types": [item.strip() for item in (args.model_types or "any").split(",") if item.strip()],
            "frameworks": [item.strip() for item in (args.frameworks or "any").split(",") if item.strip()],
            "hardware": [item.strip() for item in (args.hardware or "").split(",") if item.strip()],
            "software": json.loads(args.software) if args.software else {},
        },
        "caveats": [item.strip() for item in (args.caveats or "").split(";") if item.strip()],
        "source_run": args.source_run or "",
        "created_at": today,
        "last_verified": today,
        "usage": {"applied": 0, "success": 0, "failed": 0},
    }
    if args.error_patterns:
        entry["error_patterns"] = [item.strip() for item in args.error_patterns.split(",") if item.strip()]
    return entry


def cmd_add(args):
    if args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as handle:
            entry = json.load(handle)
        entry["category"] = canonical_category(entry.get("category", args.category), fallback=args.category)
        entry.setdefault("created_at", date.today().isoformat())
        entry.setdefault("last_verified", date.today().isoformat())
        entry.setdefault("usage", {"applied": 0, "success": 0, "failed": 0})
        entry.setdefault("confidence", "observed")
        if not entry.get("id"):
            prefix = {"optimization": "opt", "device": "dev", "code": "code"}[entry["category"]]
            entry["id"] = f"{prefix}-{date.today():%Y%m%d}-{uuid.uuid4().hex[:6]}"
    else:
        entry = build_entry(args)
    for field in ("title", "symptom", "action"):
        if not entry.get(field):
            print(f"[experience] add 失败：缺少 {field}", file=sys.stderr)
            return 1
    append_entry(args.store, entry["category"], entry)
    print(f"[experience] added {entry['id']} -> {store_path(args.store, entry['category'])}")
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    return 0


def cmd_update(args):
    entries = load_entries(args.store)
    target = None
    for entry in entries:
        if entry.get("id") == args.id:
            target = entry
            break
    if target is None:
        print(f"[experience] 找不到 id={args.id}", file=sys.stderr)
        return 1
    if args.result:
        usage = target.setdefault("usage", {"applied": 0, "success": 0, "failed": 0})
        usage["applied"] = int(usage.get("applied", 0)) + 1
        if args.result == "success":
            usage["success"] = int(usage.get("success", 0)) + 1
            target["last_verified"] = date.today().isoformat()
            if target.get("confidence") == "hypothesis":
                target["confidence"] = "observed"
        elif args.result == "failed":
            usage["failed"] = int(usage.get("failed", 0)) + 1
    if args.note:
        target.setdefault("notes", []).append(args.note)
    if args.evidence:
        target.setdefault("evidence", {})
        if isinstance(target["evidence"], dict):
            target["evidence"].setdefault("updates", []).append(args.evidence)
        else:
            target["evidence"] = {"detail": target["evidence"], "updates": [args.evidence]}
    if args.confidence:
        target["confidence"] = args.confidence
    category = target.get("category")
    rewrite_category(args.store, category, [entry for entry in entries if entry.get("category") == category])
    print(f"[experience] updated {args.id}")
    print(json.dumps(target, ensure_ascii=False, indent=2))
    return 0


def cmd_stats(args):
    entries = load_entries(args.store)
    by_category = {}
    by_confidence = {}
    top_tags = {}
    for entry in entries:
        category = entry.get("category", "?")
        confidence = entry.get("confidence", "?")
        by_category[category] = by_category.get(category, 0) + 1
        by_confidence[confidence] = by_confidence.get(confidence, 0) + 1
        for tag in entry.get("tags", []):
            top_tags[tag] = top_tags.get(tag, 0) + 1
    print(f"store: {args.store}")
    print(f"total: {len(entries)}")
    print("by category:", json.dumps(by_category, ensure_ascii=False))
    print("by confidence:", json.dumps(by_confidence, ensure_ascii=False))
    print("top tags:", json.dumps(sorted(top_tags.items(), key=lambda kv: -kv[1])[:20], ensure_ascii=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description="NPU 调优经验库")
    parser.add_argument("--store", default=DEFAULT_STORE, help=f"经验库目录（默认 {DEFAULT_STORE}）")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="检索经验")
    search.add_argument("--query", default=None)
    search.add_argument("--category", choices=list(CATEGORY_FILES), default=None)
    search.add_argument("--model-type", default=None)
    search.add_argument("--framework", default=None)
    search.add_argument("--tag", default=None, help="逗号分隔，必须全部匹配")
    search.add_argument("--confidence", default=None, help="逗号分隔：verified,observed,hypothesis")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--all", action="store_true", help="即使 query 无命中，也列出经验")
    search.add_argument("--json", default=None)

    add = subparsers.add_parser("add", help="新增一条经验")
    add.add_argument("--category", choices=list(CATEGORY_FILES), required=True)
    add.add_argument("--json-file", default=None, help="从 JSON 文件导入（优先于其他字段）")
    add.add_argument("--title", default=None)
    add.add_argument("--symptom", default=None)
    add.add_argument("--action", default=None)
    add.add_argument("--evidence", default=None, help="JSON 字符串，如 '{\"gain\":\"4%%\",\"precision\":\"cosine 0.9999\"}'")
    add.add_argument("--confidence", choices=CONFIDENCES, default="observed")
    add.add_argument("--tags", default=None)
    add.add_argument("--model-types", default="any")
    add.add_argument("--frameworks", default="any")
    add.add_argument("--hardware", default=None)
    add.add_argument("--software", default=None, help="JSON 字符串，如 '{\"cann\":\">=8.0\"}'")
    add.add_argument("--caveats", default=None, help="分号分隔")
    add.add_argument("--error-patterns", default=None)
    add.add_argument("--source-run", default=None)

    update = subparsers.add_parser("update", help="更新经验的使用结果")
    update.add_argument("--id", required=True)
    update.add_argument("--result", choices=("success", "failed", "inconclusive"), default=None)
    update.add_argument("--note", default=None)
    update.add_argument("--evidence", default=None)
    update.add_argument("--confidence", choices=CONFIDENCES, default=None)

    subparsers.add_parser("stats", help="统计经验库")

    args = parser.parse_args()
    handlers = {"search": cmd_search, "add": cmd_add, "update": cmd_update, "stats": cmd_stats}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
