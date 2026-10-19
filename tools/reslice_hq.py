"""用高音质人声重新切割，**文件名与偏移量完全沿用旧的**。

为什么这么做：
    重新跑切片器（slicer2 按静音检测）会得到**不同的边界**，
    于是老的标注文本就对不上了 —— 而用户已经人工校对过 185 条，不能重来。

    但切片文件名里编码了原始偏移：
        ..._10.wav_0009080000_0009224320.wav
        0009080000 / 32000 = 283.75s      0009224320 / 32000 = 288.26s
    只要用**相同的时间区间**从新高音质人声里切，写出**相同的文件名**，
    老的 .list 就能原样复用，一个字都不用改。

前提：两份源素材必须是同一录音且时间对齐（用互相关验证，实测 0.9911 @ 0ms）。

用法：
    $env:PYTHONPATH='src'
    .\\.venv\\Scripts\\python.exe tools\\reslice_hq.py ^
        --old-slices D:\\GPT-SoVITS\\work\\voice02\\02_sliced ^
        --vocal      D:\\GPT-SoVITS\\work\\voice02hq\\01_vocal\\vocal_....wav ^
        --out        D:\\GPT-SoVITS\\work\\voice02hq\\02_sliced
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 切片器输出的采样率（GPT-SoVITS 的 slice_audio.py 固定在 32kHz）
SLICE_SR = 32000

FFMPEG = Path(r"D:\GPT-SoVITS\GPT-SoVITS-v2pro-20250604\ffmpeg.exe")

# 文件名末尾的 _<起始样本>_<结束样本>.wav
PAT = re.compile(r"_(\d{6,})_(\d{6,})\.wav$", re.IGNORECASE)


def parse_offsets(name: str):
    m = PAT.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def dur_of(p: Path) -> float:
    try:
        with wave.open(str(p), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return -1.0


def main() -> int:
    ap = argparse.ArgumentParser(description="按旧偏移量重切高音质切片")
    ap.add_argument("--old-slices", required=True, help="旧切片目录（只读文件名）")
    ap.add_argument("--vocal", required=True, help="新的高音质人声文件")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（调试用）")
    args = ap.parse_args()

    old_dir = Path(args.old_slices)
    vocal = Path(args.vocal)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not vocal.exists():
        print(f"人声文件不存在：{vocal}")
        return 2
    if not FFMPEG.exists():
        print(f"找不到 ffmpeg：{FFMPEG}")
        return 2

    olds = sorted(p for p in old_dir.glob("*.wav") if p.stat().st_size > 1000)
    if args.limit:
        olds = olds[: args.limit]
    print("=" * 76)
    print(f"重切：{len(olds)} 个切片")
    print(f"  旧切片目录 {old_dir}")
    print(f"  新高音质人声 {vocal.name}")
    print(f"  输出        {out_dir}")
    print("=" * 76)

    ok = skip = bad = 0
    dur_mismatch = []
    for i, old in enumerate(olds, 1):
        off = parse_offsets(old.name)
        if not off:
            bad += 1
            print(f"  !! 文件名里找不到偏移：{old.name}")
            continue
        start_s, end_s = off
        t0 = start_s / SLICE_SR
        dur = (end_s - start_s) / SLICE_SR
        if dur <= 0:
            bad += 1
            continue

        dst = out_dir / old.name        # **文件名保持一致**
        if dst.exists() and dst.stat().st_size > 1000:
            skip += 1
            continue

        cmd = [
            str(FFMPEG), "-v", "error", "-y",
            "-ss", f"{t0:.6f}", "-i", str(vocal), "-t", f"{dur:.6f}",
            "-acodec", "pcm_s16le", "-ar", str(SLICE_SR), "-ac", "1",
            str(dst),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if r.returncode != 0 or not dst.exists():
            bad += 1
            if bad <= 3:
                print(f"  !! 失败 {old.name}: {(r.stderr or '')[:150]}")
            continue
        ok += 1

        # 时长对齐检查：新切片应该和旧切片几乎一样长
        d_old, d_new = dur_of(old), dur_of(dst)
        if d_old > 0 and abs(d_old - d_new) > 0.05:
            dur_mismatch.append((old.name, d_old, d_new))

        if i % 40 == 0 or i == len(olds):
            print(f"  {i}/{len(olds)} …  已切 {ok}")

    print("\n" + "=" * 76)
    print(f"完成：新切 {ok}，跳过 {skip}，失败 {bad}")
    if dur_mismatch:
        print(f"!! 有 {len(dur_mismatch)} 个切片时长对不上（>50ms）：")
        for n, a, b in dur_mismatch[:8]:
            print(f"   {n}  旧 {a:.3f}s  新 {b:.3f}s")
    else:
        print("时长全部对齐 ✅")
    print("=" * 76)
    return 0 if bad == 0 and not dur_mismatch else 1


if __name__ == "__main__":
    raise SystemExit(main())
