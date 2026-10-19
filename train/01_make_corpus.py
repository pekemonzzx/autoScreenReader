"""② 造语料：抽音轨 → UVR5 去背景 → 切片 → FunASR 自动标注。

    python train\\01_make_corpus.py --input "<素材>" --work "<工作目录>" [--segments "0:00-24:40"]

四个步骤全自动，约 10 分钟（25 分钟素材）。产物：

    <work>/00_raw/       抽出来的音轨（**保持 44.1kHz**，别降采样）
    <work>/01_vocal/     去背景后的人声
    <work>/02_sliced/    切片（3~10 秒）
    <work>/03_list/      FunASR 自动标注
    <work>/train.list    合并后的完整列表

**为什么必须用整合包的内嵌 Python**：UVR5 和 FunASR 都要 torch，
虚拟环境里没有。这个脚本会自动挑对的解释器。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from _common import (EMBED_PY, GSV_ROOT, ROOT, check_env, info, ok, run,
                     step, title, warn)


def free_ram_gb() -> float:
    """当前可用物理内存（GB）。UVR5 会把整段音频读进内存。"""
    if sys.platform != "win32":
        return -1.0
    try:
        import ctypes

        class M(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = M()
        st.dwLength = ctypes.sizeof(M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys / 1024 ** 3
    except Exception:
        return -1.0


def _sec_of(t: str) -> float:
    """'1:23:45' / '24:40' / '90' -> 秒。"""
    t = t.strip()
    parts = t.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except Exception:
        return 0.0


def _fmt(sec: float) -> str:
    """秒 -> '1:23:45'（给 --segments 用）。"""
    s = int(max(0, sec))
    h, m, ss = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{ss:02d}" if h else f"{m}:{ss:02d}"


def probe_duration(path: Path) -> float:
    """用 ffprobe 读时长（秒）。"""
    from _common import find_ffprobe

    out = subprocess.run(
        [find_ffprobe(), "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, errors="replace").stdout
    try:
        return float((out or "0").strip())
    except Exception:
        return 0.0


def parse_segments(s: str) -> list:
    """把 --segments 的字符串解析成 [(start, end)]。"""
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part or "-" not in part:
            continue
        a, b = part.split("-", 1)
        out.append((_sec_of(a), _sec_of(b)))
    return out


def subdivide(a: float, b: float, free_gb: float, max_chunk_min: float,
              target_chunk_min: float) -> list:
    """把 [a, b] 这一段按内存再细分。

    用户给的 `--segments` 可能本身就是"太长的一段"（比如 0:00-20:00），
    直接丢给 UVR5 一样会爆内存。所以自动分段要能作用在**每一段内部**，
    而不是只在"整段素材"这一层。
    """
    span = b - a
    if span <= 0:
        return [(a, b)]
    if uv5_need_gb(span) <= _budget_gb(free_gb):
        return [(a, b)]
    inner = plan_chunks(span, free_gb, max_chunk_min, target_chunk_min)
    return [(a + s, a + e) for s, e in inner]


def _budget_gb(free_gb: float) -> float:
    """这段能占多少内存：留 0.8 GB 给系统，允许 1 GB 超配，上限物理内存 80%。"""
    budget = max(1.0, free_gb - 0.8 + 1.0)
    total = total_ram_gb()
    if total > 0:
        budget = min(budget, total * 0.8)
    return budget


def uv5_need_gb(seconds: float) -> float:
    """UVR5 处理这么长音频要多少内存（GB）。

    **这是实测校准过的，不是估算。**

    本机实测（RTX 4080S + 16GB 内存，HP5_only_main_vocal，agg=10）：

        30 秒   -> 7.8 秒完成    内存很小
        10 分钟 -> 45.8 秒完成   进程峰值 7,224 MB

    即 **约 12 MB / 秒音频**，再加约 0.3 GB 固定开销。

    UVR5 会把**整段音频的频谱图一次性读进内存**（不流式），
    所以内存和时长是**线性**的：

        25 分钟 -> 约 18 GB  >  16 GB 物理内存  ->  必定疯狂换页，跑不完

    早先按 44.1kHz×2ch×float64×4份 估出来是 4 份拷贝，
    实测远不止 —— 会严重低估，导致大文件"看着在跑其实永远跑不完"。
    """
    return seconds * 0.012 + 0.3


def total_ram_gb() -> float:
    """物理内存总量（GB）。"""
    if sys.platform != "win32":
        return -1.0
    try:
        import ctypes

        class M(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = M()
        st.dwLength = ctypes.sizeof(M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullTotalPhys / 1024 ** 3
    except Exception:
        return -1.0


def plan_chunks(total_sec: float, free_gb: float, max_chunk_min: float,
                target_chunk_min: float) -> list:
    """按可用内存算出该切成几段。

    返回 [(start_sec, end_sec)]；只有一段时表示不用切。

    四条原则：
      * **内存装得下就不切** —— 不切质量最好（段边界可能切断一个词）
      * 允许 **1 GB 轻微超配**（页面文件能便宜地吸收一点，实测 10 分钟
        在 4.9 GB 可用下靠换页 46 秒就跑完了），但
      * **硬上限是物理内存的 80%** —— 超过这个量就是"永远跑不完"，
        25 分钟需要 18 GB > 16 GB 物理内存，实测 18 分钟零产出
      * 要切就切均匀，**尾段太短就并进上一段**
    """
    if total_sec <= 0:
        return []

    budget = _budget_gb(free_gb)

    # 内存够就整段跑（不切最好）
    if uv5_need_gb(total_sec) <= budget:
        return [(0.0, total_sec)]

    # 这一段最长能多长（秒）
    max_by_ram_sec = max(120.0, (budget - 0.3) / 0.012)

    chunk_min = min(max_chunk_min, target_chunk_min, max_by_ram_sec / 60)
    chunk_min = max(2.0, chunk_min)          # 别切得太碎，太短影响断句
    chunk_sec = chunk_min * 60

    n = max(1, int(total_sec / chunk_sec) + (1 if total_sec % chunk_sec else 0))
    # 均匀切：避免"5 段 20 分钟 + 最后 19 秒"这种
    even = total_sec / n
    if even < 120.0:                         # 每段不到 2 分钟就别切了
        return [(0.0, total_sec)]

    out = []
    for i in range(n):
        a = i * even
        b = total_sec if i == n - 1 else (i + 1) * even
        out.append((a, b))

    # 尾段太短（< 3 分钟）就并进上一段
    if len(out) >= 2 and (out[-1][1] - out[-1][0]) < 180.0:
        a = out[-2][0]
        out = out[:-2] + [(a, total_sec)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="造语料")
    ap.add_argument("--input", required=True, help="素材文件（视频/音频）")
    ap.add_argument("--work", required=True, help="工作目录（会建在 GPT-SoVITS\\work 下更省事）")
    ap.add_argument("--segments", default=None,
                    help="只取指定时间段，如 '0:00-24:40' 或 '0:00-20:00,20:00-40:00'（不给则取全部）")
    ap.add_argument("--auto-split", action="store_true",
                    help="按可用内存**自动分段**（推荐：大文件不加这个会内存不足）")
    ap.add_argument("--clean-raw", action="store_true",
                    help="把 00_raw 里上次残留的音轨挪到 _stale_raw\\（不加则只警告）")
    ap.add_argument("--target-chunk-min", type=float, default=20.0,
                    help="自动分段时每段目标长度（分钟），默认 20")
    ap.add_argument("--max-chunk-min", type=float, default=30.0,
                    help="自动分段时每段上限（分钟），默认 30")
    ap.add_argument("--model", default="HP5_only_main_vocal", help="UVR5 模型")
    ap.add_argument("--agg", type=int, default=10, help="UVR5 激进度 0-20")
    ap.add_argument("--language", default="zh", choices=["zh", "yue", "auto"])
    ap.add_argument("--asr-size", default="large", choices=["large", "medium", "small"])
    ap.add_argument("--sample-rate", type=int, default=44100,
                    help="抽音轨的采样率。**别调低** —— 擦音特征在 4~10kHz")
    args = ap.parse_args()

    title("造语料")
    if not check_env(need_embed=True):
        return 2

    src = Path(args.input)
    if not src.exists():
        warn(f"素材不存在：{src}")
        return 2
    work = Path(args.work)
    prep = ROOT / "tools" / "tts_prepare.py"
    if not prep.exists():
        warn(f"找不到 {prep}")
        return 2

    total = probe_duration(src)
    free = free_ram_gb()
    info(f"素材时长 {total / 60:.1f} 分钟")
    info(f"可用内存 {free:.1f} GB" if free >= 0 else "可用内存 未知")

    # ---- 决定分段方案 ----
    #
    # 注意：**用户给的 --segments 也要再细分**。
    # 以前只在"没有 --segments"时才自动分段，结果 `--segments "0:00-20:00"`
    # 这种 20 分钟的段直接丢给 UVR5，一样会爆内存（20 分钟约需 15 GB）。
    segments = args.segments
    if args.auto_split and total > 0 and free >= 0:
        step("0", "自动分段（按可用内存算）")
        base = parse_segments(segments) if segments else [(0.0, total)]
        plan = []
        for a, b in base:
            plan.extend(subdivide(a, b, free, args.max_chunk_min,
                                 args.target_chunk_min))
        if len(plan) == 1:
            ok(f"整段就能跑（{(plan[0][1] - plan[0][0]) / 60:.1f} 分钟，"
               f"预计需要 {uv5_need_gb(plan[0][1] - plan[0][0]):.1f} GB）")
        else:
            longest = max(b - a for a, b in plan)
            info(f"最长的一段需要 {uv5_need_gb(longest):.1f} GB，内存不够 —— "
                 f"切成 {len(plan)} 段：")
            for a, b in plan[:6]:
                info(f"    {_fmt(a):>9} - {_fmt(b):<9}  {(b - a) / 60:5.1f} 分钟"
                     f"  峰值约 {uv5_need_gb(b - a):.1f} GB")
            if len(plan) > 6:
                info(f"    …… 共 {len(plan)} 段")
            info("")
            info("  每段是**独立文件**，UVR5 逐个处理 —— 峰值只和最长那段有关。")
            info("  代价：段边界可能切断一个词，边界附近会有几条废片，标注时删掉即可。")
            segments = ",".join(f"{_fmt(a)}-{_fmt(b)}" for a, b in plan)

    if not segments:
        step("0", "内存预检")
        need = uv5_need_gb(total) if total > 0 else 0.0
        if free >= 0 and need:
            info(f"UVR5 预计需要约 {need:.1f} GB")
            if free < need:
                warn("内存不够 —— UVR5 会中途抛 numpy._ArrayMemoryError")
                warn("  解决办法（任选）：")
                warn("   1. 加 --auto-split 让它自动分段（推荐）")
                warn("   2. 自己指定 --segments \"0:00-20:00,20:00-40:00\"")
                warn("   3. 关掉浏览器/游戏再重跑")
                return 3
            ok("内存够")

    # ---- 检查 00_raw 里有没有上次残留的文件 ----
    #
    # UVR5 会处理 `00_raw` 下的**所有** wav。如果上次没带 --segments 抽了整段
    # 留在那儿，这次即使只指定 25 分钟，它也会把那个整段文件一起处理 ——
    # 内存又不够。实测踩过：607 MB / 120 分钟的残留把内存预检重新顶到 20 GB。
    raw_dir = work / "00_raw"
    if raw_dir.is_dir():
        stale = [p for p in raw_dir.glob("*.wav") if p.stat().st_size > 1000]
        if stale:
            step("0", f"00_raw 里有 {len(stale)} 个已有音轨")
            total_stale = 0.0
            for p in stale:
                d = probe_duration(p)
                total_stale = max(total_stale, d)
                info(f"    {p.name}   {d / 60:.1f} 分钟   {p.stat().st_size / 1048576:.0f} MB")
            if segments:
                warn("UVR5 会处理 00_raw 下的**所有** wav —— 包括上面这些")
                warn(f"  其中最长的一段 {total_stale / 60:.1f} 分钟，"
                     f"单独就要 {uv5_need_gb(total_stale):.1f} GB")
                if args.clean_raw:
                    aside = work / "_stale_raw"
                    aside.mkdir(parents=True, exist_ok=True)
                    for p in stale:
                        shutil.move(str(p), str(aside / p.name))
                    ok(f"已把 {len(stale)} 个移到 {aside}（不会参与这次处理）")
                else:
                    warn("  加 --clean-raw 可以把它们挪到 _stale_raw\\，不影响这次处理")
                    warn("  （本次要处理的是你 --segments 指定的那段，会重新抽出来）")

    cmd = [
        # **-u 不能省**：不加的话 stdout 走块缓冲，UVR5 的进度信息
        # 会一直卡在缓冲区里，看上去就像"卡死了没反应"（实测踩过 17 分钟没输出）。
        EMBED_PY, "-u", "-s", prep,
        "--input", str(src),
        "--work", str(work),
        # 显式传整合包根目录，别让 tts_prepare 用它自己的默认值 ——
        # 那个默认值曾经指向 v3lora，和训练用的 v2pro 不是同一个包
        "--gsv-root", str(GSV_ROOT),
        "--model", args.model,
        "--agg", str(args.agg),
        "--language", args.language,
        "--asr-size", args.asr_size,
        "--sample-rate", str(args.sample_rate),
    ]
    if segments:
        cmd += ["--segments", segments]

    step("1-4", "抽音轨 → UVR5 → 切片 → ASR（可能要 10~30 分钟）")
    rc = run(cmd, env_extra={"PYTHONIOENCODING": "utf-8"},
             cwd=str(ROOT))
    if rc != 0:
        warn("流程没跑完，看上面的输出定位问题")
        return rc

    lst = work / "train.list"
    if not lst.exists():
        warn(f"没生成 {lst}")
        return 1

    n = len([l for l in lst.read_text(encoding="utf-8").splitlines() if l.strip()])
    sl = list((work / "02_sliced").glob("*.wav"))
    print()
    title("完成")
    ok(f"切片 {len(sl)} 个，标注 {n} 条")
    info(f"列表：{lst}")
    print()
    info("下一步（机械过滤）：")
    info(f'  python train\\02_filter_list.py --list "{lst}" '
         f'--out "data\\<名字>_train_clean.list" '
         f'--wav-dir "{work / "02_sliced"}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
