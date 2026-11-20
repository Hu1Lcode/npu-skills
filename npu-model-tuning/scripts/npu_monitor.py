#!/usr/bin/env python3
"""采样 Ascend NPU 遥测到 CSV，可用于 benchmark 期间观察 HBM / AICore / 功耗 / 温度。

输出列与 nvidia-smi 风格保持一致:
    timestamp,index,memory_used_mib,memory_total_mib,utilization_pct,power_w,temperature_c
index 为 torch_npu 逻辑设备号：按 ASCEND_RT_VISIBLE_DEVICES 将物理 chip logic id 重映射。

用法:
    python npu_monitor.py --interval-ms 1000 --duration 600 --output perf_reports/date/artifacts/npu_smi.csv
    python npu_monitor.py --interval-ms 2000                 # 前台持续采样，Ctrl-C 结束
    python npu_monitor.py --list                             # 只打印一次采样结果
"""
import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime

try:
    from zoneinfo import ZoneInfo

    _TIMEZONE = ZoneInfo("Asia/Shanghai")
except Exception:  # noqa: BLE001
    _TIMEZONE = None

_NUMBER = r"(?:\d+(?:\.\d+)?|N/A|-)"
_CHIP_METRICS = re.compile(
    rf"^\s*({_NUMBER})\s+({_NUMBER})\s*/\s*({_NUMBER})\s+({_NUMBER})\s*/\s*({_NUMBER})\s*$"
)


def _query(*arguments, timeout=10):
    return subprocess.run(
        ["npu-smi", *arguments], capture_output=True, text=True, check=True, timeout=timeout
    ).stdout


def _chip_map():
    """Map (NPU id, chip id) -> chip logic id used by torch_npu."""
    mapping = {}
    for line in _query("info", "-m").splitlines():
        fields = line.split()
        if len(fields) >= 4 and all(field.isdigit() for field in fields[:4]) and fields[2] != "-":
            mapping[(int(fields[0]), int(fields[1]))] = int(fields[2])
    return mapping


def _visible_devices():
    """Physical chip logic ids exposed to torch_npu, in logical-index order."""
    value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
    if not value:
        return None
    devices = []
    try:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                start, end = (int(part) for part in item.split("-", 1))
                devices.extend(range(start, end + 1))
            else:
                devices.append(int(item))
    except ValueError:
        return None
    return devices or None


def _number(value):
    return float(value) if value not in ("-", "N/A") else 0.0


def sample(mapping):
    """Return one row per chip: (index, memory_used, memory_total, aicore, power, temp)."""
    rows = []
    power = None
    temperature = None
    npu_id = None
    for line in _query("info").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        head = cells[0].split()
        if not head or not head[0].isdigit():
            continue
        if len(head) >= 2 and head[1].isdigit():
            chip_id = int(head[0])
            metrics = _CHIP_METRICS.match(cells[2])
            if not metrics:
                continue
            aicore, memory_used, memory_total, hbm_used, hbm_total = (
                _number(value) for value in metrics.groups()
            )
            if hbm_total:
                memory_used, memory_total = hbm_used, hbm_total
            index = mapping.get((npu_id, chip_id), int(head[1]))
            rows.append((index, memory_used, memory_total, aicore, power, temperature))
        else:
            npu_id = int(head[0])
            fields = cells[2].split()
            if len(fields) >= 2:
                current = _number(fields[0]) if fields[0] not in ("-", "N/A") else None
                if current is not None:
                    power = current
                temperature = _number(fields[1]) if fields[1] not in ("-", "N/A") else None
    return sorted(rows)


def main():
    parser = argparse.ArgumentParser(description="Ascend NPU 遥测采样")
    parser.add_argument("--interval-ms", type=int, default=2000)
    parser.add_argument("--duration", type=float, default=0, help="采样时长（秒），0=一直采直到 Ctrl-C")
    parser.add_argument("--output", default=None, help="CSV 输出路径；不填则打印到 stdout")
    parser.add_argument("--list", action="store_true", help="只采样一次并打印")
    args = parser.parse_args()

    mapping = _chip_map()
    visible = _visible_devices()
    writer = None
    handle = None
    if args.output and not args.list:
        handle = open(args.output, "w", encoding="utf-8")
        writer = handle
        writer.write("timestamp,index,memory_used_mib,memory_total_mib,utilization_pct,power_w,temperature_c\n")
        writer.flush()

    started = time.monotonic()
    try:
        while True:
            tick = time.monotonic()
            now = datetime.now(_TIMEZONE) if _TIMEZONE else datetime.now()
            timestamp = now.strftime("%Y/%m/%d %H:%M:%S.%f")
            try:
                for index, used, total, aicore, power, temperature in sample(mapping):
                    if visible is not None:
                        if index not in visible:
                            continue
                        index = visible.index(index)
                    row = (
                        f"{timestamp},{index},{used:.0f},{total:.0f},"
                        f"{aicore:.0f},{power if power is not None else 0:.1f},"
                        f"{temperature if temperature is not None else 0:.0f}"
                    )
                    if args.list:
                        print(row)
                    elif writer is not None:
                        writer.write(row + "\n")
                    else:
                        print(row, flush=True)
            except Exception as error:  # noqa: BLE001
                print(f"npu-smi sampling failed: {error}", file=sys.stderr, flush=True)
            if args.list:
                return
            if args.duration and time.monotonic() - started >= args.duration:
                break
            delay = args.interval_ms / 1000.0 - (time.monotonic() - tick)
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        if handle is not None:
            handle.close()


if __name__ == "__main__":
    main()
