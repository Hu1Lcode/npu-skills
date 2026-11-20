#!/usr/bin/env python3
"""从 git 提交历史生成每个优化点的 patch 和 all.patch。

约定：每个已固化优化点一个 commit，subject 以 `opt(N):` 开头（如 `opt(01): attention 后端替换`）。
用法:
    python make_patches.py --repo /path/to/model --base <基线commit> --output perf_reports/20260101/patches
    python make_patches.py --repo . --base HEAD~5 --output patches
"""
import argparse
import json
import os
import re
import subprocess
import sys


def git(repo, *args):
    result = subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def slugify(subject):
    cleaned = re.sub(r"^opt\(?\d+\)?[:：\s-]*", "", subject, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"[^\w一-鿿.-]+", "-", cleaned)
    cleaned = cleaned.strip("-_.")
    return cleaned[:80] or "optimization"


def main():
    parser = argparse.ArgumentParser(description="按优化 commit 生成 patch")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base", required=True, help="调优前基线 commit")
    parser.add_argument("--output", required=True, help="patch 输出目录")
    parser.add_argument("--pattern", default=r"^opt\(?\d+\)?[:：]", help="匹配优化 commit 的 subject 正则")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    log = git(args.repo, "log", "--reverse", "--format=%H%x00%s", f"{args.base}..HEAD")
    entries = []
    for line in log.splitlines():
        if not line.strip():
            continue
        sha, subject = line.split("\x00", 1)
        entries.append((sha, subject))

    probe_like = [subject for _sha, subject in entries if "probe" in subject.lower()]
    if probe_like:
        print(
            "[make_patches] 警告：历史中存在 probe 相关 commit：\n  " + "\n  ".join(probe_like) +
            "\n请确认最终 patch 不包含打点代码（建议在 P8 前先 revert 这些 commit）。",
            file=sys.stderr,
        )

    matched = []
    for sha, subject in entries:
        if re.match(args.pattern, subject):
            matched.append((sha, subject))

    if not matched:
        print("[make_patches] 未找到以 opt(N): 开头的 commit，只生成 all.patch", file=sys.stderr)

    index = []
    for number, (sha, subject) in enumerate(matched, start=1):
        parent = git(args.repo, "rev-parse", f"{sha}^").strip()
        diff = subprocess.run(
            ["git", "-C", args.repo, "diff", "--binary", parent, sha],
            capture_output=True, check=False,
        )
        if diff.returncode != 0:
            print(f"[make_patches] diff {sha} 失败: {diff.stderr.decode(errors='replace')}", file=sys.stderr)
            continue
        filename = f"{number:03d}-{slugify(subject)}.patch"
        path = os.path.join(args.output, filename)
        with open(path, "wb") as handle:
            handle.write(diff.stdout)
        index.append({"number": number, "commit": sha, "subject": subject, "patch": filename})

    all_diff = subprocess.run(
        ["git", "-C", args.repo, "diff", "--binary", args.base, "HEAD"],
        capture_output=True, check=False,
    )
    all_path = os.path.join(args.output, "all.patch")
    with open(all_path, "wb") as handle:
        handle.write(all_diff.stdout)
    index.append({"number": "all", "commit": "HEAD", "subject": "all optimizations", "patch": "all.patch"})

    with open(os.path.join(args.output, "index.json"), "w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)

    print(f"[make_patches] wrote {len(matched)} per-optimization patch(es) + all.patch -> {args.output}")


if __name__ == "__main__":
    main()
