#!/usr/bin/env python3
"""在临时 worktree 中按顺序应用优化 patch，验证堆叠能干净落地。

这是“集成验证”的机械部分：只验证 patch 可应用性、可选跑一条 smoke/full 命令；
性能与精度的最终判定仍由 C 的成对测试给出。

用法:
    python verify_stack.py --repo /path/to/model --base <commit> --patches perf_reports/20260101/patches
    python verify_stack.py --repo /path/to/model --base <commit> --patches perf_reports/20260101/patches \\
        --command "bash run.sh --smoke"
    python verify_stack.py ... --keep    # 保留临时 worktree 便于排查（默认删除）

输出:
    <patches>/stack_verify.json
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile


def run_git(repo, *args, check=True):
    result = subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def resolve_patches(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.patch")))
        return [item for item in files if os.path.basename(item) != "all.patch"]
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(path)


def main():
    parser = argparse.ArgumentParser(description="验证优化 patch 堆叠")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base", required=True, help="调优前基线 commit")
    parser.add_argument("--patches", required=True, help="patch 目录或单个 patch 文件")
    parser.add_argument("--command", default=None, help="可选：在应用后的工作区执行的命令（如 smoke）")
    parser.add_argument("--output", default=None, help="结果 JSON 路径")
    parser.add_argument("--keep", action="store_true", help="保留临时 worktree")
    args = parser.parse_args()

    patches = resolve_patches(args.patches)
    if not patches:
        print("[verify_stack] 没有找到 patch 文件", file=sys.stderr)
        return 1

    output = args.output or os.path.join(
        args.patches if os.path.isdir(args.patches) else os.path.dirname(args.patches) or ".",
        "stack_verify.json",
    )

    worktree = tempfile.mkdtemp(prefix="npu_tune_stack_")
    result = {
        "repo": os.path.abspath(args.repo),
        "base": args.base,
        "worktree": worktree,
        "patches": [],
        "command": args.command,
        "command_returncode": None,
        "ok": False,
    }
    try:
        run_git(args.repo, "worktree", "add", "--detach", worktree, args.base)
        ok = True
        for patch in patches:
            entry = {"patch": os.path.abspath(patch), "applied": False, "error": None}
            check = run_git(worktree, "apply", "--check", os.path.abspath(patch), check=False)
            if check.returncode != 0:
                entry["error"] = check.stderr.strip() or check.stdout.strip()
                ok = False
                result["patches"].append(entry)
                print(f"[verify_stack] FAIL {os.path.basename(patch)}: {entry['error']}", file=sys.stderr)
                break
            apply_result = run_git(worktree, "apply", os.path.abspath(patch), check=False)
            if apply_result.returncode != 0:
                entry["error"] = apply_result.stderr.strip() or apply_result.stdout.strip()
                ok = False
                result["patches"].append(entry)
                print(f"[verify_stack] FAIL {os.path.basename(patch)}: {entry['error']}", file=sys.stderr)
                break
            entry["applied"] = True
            result["patches"].append(entry)
            print(f"[verify_stack] applied {os.path.basename(patch)}")

        if ok and args.command:
            command_result = subprocess.run(
                args.command, cwd=worktree, shell=True, text=True, check=False
            )
            result["command_returncode"] = command_result.returncode
            if command_result.returncode != 0:
                ok = False
                print(f"[verify_stack] command failed: {args.command}", file=sys.stderr)

        result["ok"] = ok
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"[verify_stack] {'OK' if ok else 'FAILED'} -> {output}")
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"[verify_stack] ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.keep:
            run_git(args.repo, "worktree", "remove", "--force", worktree, check=False)
            if os.path.isdir(worktree):
                shutil.rmtree(worktree, ignore_errors=True)
        else:
            print(f"[verify_stack] kept worktree: {worktree}")


if __name__ == "__main__":
    sys.exit(main())
