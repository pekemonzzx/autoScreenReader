"""每 5 分钟打印一次运行状态，供长时间任务监控。

输出格式固定，方便一眼看出"在跑 / 卡住 / 完成"。
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERVAL = 300.0          # 5 分钟
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 24     # 默认报 2 小时

WATCH = ("probe_librosa", "bisect", "tts_prepare", "01_make_corpus",
         "run_until", "test_subprocess", "05_train_voice", "run_with_embedded")


def snapshot() -> str:
    lines = []

    # 1) 相关进程
    try:
        ps = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object ProcessId,CommandLine,CreationDate | ConvertTo-Json -Compress"],
            capture_output=True, text=True, errors="replace", timeout=30).stdout
    except Exception:
        ps = ""

    procs = []
    if ps and ps.strip() not in ("", "null"):
        import json
        try:
            data = json.loads(ps)
            if isinstance(data, dict):
                data = [data]
            for p in data:
                cl = p.get("CommandLine") or ""
                if any(w in cl for w in WATCH):
                    procs.append((p.get("ProcessId"), cl))
        except Exception:
            pass

    if procs:
        lines.append(f"  进程: {len(procs)} 个相关")
        for pid, cl in procs:
            tag = "?"
            for w in WATCH:
                if w in cl:
                    tag = w
                    break
            lines.append(f"    PID {pid}  [{tag}]")
    else:
        lines.append("  进程: 无相关任务在跑")

    # 2) voice03 产出
    w = Path(r"D:\GPT-SoVITS\work\voice03")
    if w.exists():
        parts = []
        for d in ("00_raw", "01_vocal", "01_instrumental", "02_sliced", "03_list"):
            p = w / d
            n = len(list(p.glob("*"))) if p.exists() else 0
            parts.append(f"{d}={n}")
        parts.append(f"train.list={'有' if (w / 'train.list').exists() else '无'}")
        lines.append("  voice03: " + "  ".join(parts))

    # 3) 内存 / GPU
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$os=Get-CimInstance Win32_OperatingSystem; "
             "'{0:N2}' -f ($os.FreePhysicalMemory/1MB)"],
            capture_output=True, text=True, errors="replace", timeout=20).stdout.strip()
        lines.append(f"  内存可用: {out} GB")
    except Exception:
        pass

    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader"], capture_output=True,
                           text=True, errors="replace", timeout=20).stdout.strip()
        lines.append(f"  GPU: {g}")
    except Exception:
        pass

    return "\n".join(lines)


print("=" * 78, flush=True)
print(f"状态监控启动 {datetime.now():%H:%M:%S}  每 {INTERVAL/60:.0f} 分钟报一次，共 {ROUNDS} 次",
      flush=True)
print("=" * 78, flush=True)

for r in range(1, ROUNDS + 1):
    time.sleep(INTERVAL)
    print(f"\n[{datetime.now():%H:%M:%S}]  第 {r}/{ROUNDS} 次回报", flush=True)
    print(snapshot(), flush=True)
