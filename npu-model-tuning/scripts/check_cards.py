#!/usr/bin/env python3
"""检查指定 NPU 卡是否空闲（无其他进程、AICore 利用率低），用于测试前卡占用校验。

用法:
    python check_cards.py                          # 检查当前可见卡
    python check_cards.py --devices 0,1,2,3        # 只检查指定 chip logic id
    python check_cards.py --devices 0,1 --wait 120 # 最多等 120 秒直到空闲
    python check_cards.py --devices 0,1 --json out.json

退出码:
    0 = 全部空闲
    1 = 有卡被占用/不空闲
    2 = 查询失败

说明:
    - 卡号使用 npu-smi 的 chip logic id（与 ASCEND_RT_VISIBLE_DEVICES 的取值一致）。
    - 优先依据 npu-smi 进程表判断；HBM 在空闲时也可能有数 GB 预留，不能只看显存。
    - 真正跑性能测试前，还应记录 npu-smi 快照和同卡其他租户信息。
"""
import argparse
import json
import os
import re
import sys
import time

from npu_monitor import _chip_map, _query, _visible_devices, sample


def parse_processes():
    """从 npu-smi info 的进程表解析每张卡的进程。返回 logic_id -> process list。"""
    mapping = _chip_map()
    processes = {}
    try:
        text = _query("info")
    except Exception:  # noqa: BLE001
        return processes
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells:
            continue
        head = cells[0].split()
        # 进程行格式：| NPU Chip | Process id | Process name | Process memory(MB) |
        if len(cells) >= 4 and len(head) >= 2 and head[0].isdigit() and head[1].isdigit():
            pid = cells[1]
            if not pid.isdigit():
                continue
            npu_id, chip_id = int(head[0]), int(head[1])
            logic_id = mapping.get((npu_id, chip_id))
            if logic_id is None:
                continue
            try:
                memory_mib = int(float(cells[3]))
            except ValueError:
                memory_mib = None
            processes.setdefault(logic_id, []).append(
                {"npu_id": npu_id, "chip_id": chip_id, "pid": int(pid), "name": cells[2], "memory_mib": memory_mib}
            )
    return processes


def snapshot(devices=None, max_util_pct=5.0, max_memory_mib=4096):
    mapping = _chip_map()
    processes = parse_processes()
    rows = []
    for index, used, total, aicore, power, temperature in sample(mapping):
        if devices is not None and index not in devices:
            continue
        procs = processes.get(index, [])
        idle = (len(procs) == 0) and (aicore <= max_util_pct)
        rows.append(
            {
                "index": index,
                "memory_used_mib": round(used, 0),
                "memory_total_mib": round(total, 0),
                "utilization_pct": round(aicore, 1),
                "power_w": round(power, 1) if power is not None else None,
                "temperature_c": round(temperature, 0) if temperature is not None else None,
                "process_count": len(procs),
                "processes": procs,
                "idle": idle,
                "memory_high": used > max_memory_mib,
            }
        )
    return sorted(rows, key=lambda item: item["index"])


def main():
    parser = argparse.ArgumentParser(description="检查 NPU 卡是否空闲")
    parser.add_argument("--devices", default=None, help="逗号分隔的 chip logic id；默认使用 ASCEND_RT_VISIBLE_DEVICES 或全部")
    parser.add_argument("--max-util-pct", type=float, default=5.0, help="判定空闲的 AICore 利用率上限")
    parser.add_argument("--max-memory-mib", type=float, default=4096, help="仅用于标记显存偏高的参考阈值")
    parser.add_argument("--wait", type=float, default=0, help="最多等待多少秒直到全部空闲；0=不等待")
    parser.add_argument("--interval", type=float, default=2.0, help="等待时的轮询间隔（秒）")
    parser.add_argument("--json", default=None, help="把结果写到 JSON 文件")
    args = parser.parse_args()

    devices = None
    if args.devices:
        devices = {int(item.strip()) for item in args.devices.split(",") if item.strip()}
    else:
        visible = _visible_devices()
        devices = set(visible) if visible else None

    deadline = time.monotonic() + max(0.0, args.wait)
    while True:
        try:
            rows = snapshot(devices, args.max_util_pct, args.max_memory_mib)
        except Exception as exc:  # noqa: BLE001
            print(f"[check_cards] 查询 npu-smi 失败: {exc}", file=sys.stderr)
            return 2
        all_idle = bool(rows) and all(item["idle"] for item in rows)
        if all_idle or time.monotonic() >= deadline:
            break
        time.sleep(max(0.1, args.interval))

    payload = {
        "all_idle": all_idle,
        "checked_devices": [item["index"] for item in rows],
        "max_util_pct": args.max_util_pct,
        "max_memory_mib": args.max_memory_mib,
        "devices": rows,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"{'chip':<6}{'util%':>8}{'mem(MiB)':>12}{'proc':>8}{'power(W)':>10}{'temp(C)':>9}{'idle':>7}")
    for item in rows:
        print(
            f"{item['index']:<6}{item['utilization_pct']:>8.1f}{item['memory_used_mib']:>12.0f}"
            f"{item['process_count']:>8}{item['power_w'] if item['power_w'] is not None else '-':>10}"
            f"{item['temperature_c'] if item['temperature_c'] is not None else '-':>9}"
            f"{'YES' if item['idle'] else 'NO':>7}"
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"[check_cards] wrote {args.json}")
    return 0 if all_idle else 1


if __name__ == "__main__":
    sys.exit(main())
