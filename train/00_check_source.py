"""① 素材体检 —— **开始做语料之前一定先跑这个**。

    python train\\00_check_source.py "D:\\path\\to\\素材.m4a"

查三件事，任何一条不过就别往下做了：

  1. **采样率**：< 44.1kHz 说明高频已经没了，擦了练不回来
  2. **高频能量**：8–11kHz 低于 -55 dB 说明 ch/zh/sh/c/z/s 会发闷
  3. **时长**：太短不够用，太长浪费时间

为什么这一步最重要：
本项目的最大一次音质提升就是换素材（16kHz → 48kHz），
8–11kHz 相差 **+23 dB**，ASR 回读从「基本杀」变成「剧本杀」。
信息在源头没了，后面练一万轮也变不出来。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _common import (ROOT, GSV_ROOT, find_ffmpeg, find_ffprobe, human_mb,
                     info, ok, step, title, warn)

# 判定阈值（拿本机已知的好/坏素材校准过）
#   48kHz 音轨的 8-11kHz 实测 -24.0 dB   -> 健康
#   16kHz 素材的 8-11kHz 实测 -45.8 dB   -> 高频已丢
MIN_SR_WARN = 44100        # 低于这个警告
HF_BAD_DB = -40.0          # 8-11kHz 低于这个说明高频基本没了
HF_GOOD_DB = -32.0         # 高于这个算健康
MIN_DUR = 300.0            # 至少 5 分钟素材
GOOD_DUR = 600.0           # 10 分钟以上比较从容


def probe(path: Path) -> dict:
    """读采样率/声道/时长。"""
    ffprobe = find_ffprobe()
    out = subprocess.run(
        [ffprobe, "-v", "error",
         "-show_entries", "stream=sample_rate,channels,codec_name",
         "-show_entries", "format=duration,bit_rate",
         "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True, errors="replace",
    ).stdout
    d = {}
    for line in (out or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def spectrum(path: Path, seconds: float = 60.0) -> dict:
    """抽一段算各频段能量（相对 0.3-1kHz 的 dB）。"""
    import numpy as np

    ffmpeg = find_ffmpeg()
    tmp = ROOT / "data" / "_check_tmp.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    sr = 48000
    subprocess.run([ffmpeg, "-v", "error", "-y", "-t", str(seconds),
                    "-i", str(path), "-vn", "-acodec", "pcm_s16le",
                    "-ar", str(sr), "-ac", "1", str(tmp)],
                   capture_output=True)
    if not tmp.exists():
        return {}
    import wave
    with wave.open(str(tmp), "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    tmp.unlink(missing_ok=True)
    if len(a) < sr:
        return {}

    n = 4096
    hop = n // 2
    win = np.hanning(n)
    acc = None
    used = 0
    for i in range(0, len(a) - n, hop):
        seg = a[i:i + n] * win
        sp = np.abs(np.fft.rfft(seg)) ** 2
        acc = sp if acc is None else acc + sp
        used += 1
    if not used:
        return {}
    psd = acc / used
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    ref = psd[(fr >= 300) & (fr < 1000)].mean()
    bands = [(300, 1000), (1000, 2000), (2000, 4000), (4000, 6000),
             (6000, 8000), (8000, 11000), (11000, 16000)]
    out = {}
    for lo, hi in bands:
        m = (fr >= lo) & (fr < hi)
        if not m.any():
            continue
        v = psd[m].mean()
        out[(lo, hi)] = 10 * np.log10(v / ref) if v > 0 and ref > 0 else -99.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="素材体检")
    ap.add_argument("input", help="素材文件（视频/音频都行）")
    ap.add_argument("--seconds", type=float, default=60.0, help="抽多少秒做频谱分析")
    args = ap.parse_args()

    p = Path(args.input)
    title("素材体检")
    if not p.exists():
        warn(f"文件不存在：{p}")
        return 2

    info(f"文件    {p}")
    info(f"大小    {human_mb(p.stat().st_size)}")

    pr = probe(p)
    sr = int(pr.get("sample_rate", 0) or 0)
    ch = pr.get("channels", "?")
    dur = float(pr.get("duration", 0) or 0)
    br = pr.get("bit_rate", "?")
    info(f"编码    {pr.get('codec_name', '?')}  {sr} Hz  {ch} 声道  {br} bit/s")
    info(f"时长    {dur:.1f} 秒（{dur / 60:.1f} 分钟）")

    step("1", "采样率")
    if sr >= MIN_SR_WARN:
        ok(f"{sr} Hz，够用")
    elif sr >= 32000:
        warn(f"{sr} Hz —— 偏低。16kHz 以上会被砍，擦音会闷")
        warn("   能用，但别指望擦音清楚")
    else:
        warn(f"{sr} Hz —— **太低**。8kHz 以上全没了，ch/zh/sh/c/z/s 一定发闷")
        warn("   建议换素材，别浪费后面几小时")

    step("2", "高频能量（8–11 kHz）")
    spec = spectrum(p, args.seconds)
    if not spec:
        warn("频谱分析失败（ffmpeg 没跑起来？）")
    else:
        for (lo, hi), db in spec.items():
            bar = "#" * int(max(0, (db + 60) / 2))
            mark = ""
            if lo == 8000:
                mark = "   <- 擦音关键区"
            print(f"      {lo/1000:>5.1f}-{hi/1000:<5.1f}kHz {db:>7.1f} dB  {bar}{mark}")
        hf = spec.get((8000, 11000))
        print()
        if hf is None:
            warn("没算出 8-11kHz")
        elif hf >= HF_GOOD_DB:
            ok(f"8–11kHz {hf:.1f} dB —— 高频健康")
        elif hf >= HF_BAD_DB:
            warn(f"8–11kHz {hf:.1f} dB —— 偏高但可用")
        else:
            warn(f"8–11kHz {hf:.1f} dB —— **高频基本没了**，擦音会闷")
            warn("   对比：本机的 48kHz 素材在 -24 dB 左右，16kHz 的在 -46 dB")

    step("3", "时长")
    if dur < MIN_DUR:
        warn(f"只有 {dur / 60:.1f} 分钟，太少。建议至少 10 分钟素材")
    elif dur < GOOD_DUR:
        info(f"{dur / 60:.1f} 分钟 —— 够用（UVR5 分离+切片后大约剩一半）")
    else:
        ok(f"{dur / 60:.1f} 分钟 —— 很充足")

    print()
    title("结论")
    bad = []
    if sr < 32000:
        bad.append("采样率太低")
    if spec and spec.get((8000, 11000), 0) < HF_BAD_DB:
        bad.append("高频缺失")
    if dur < MIN_DUR:
        bad.append("时长不足")

    if not bad:
        ok("素材没问题，可以进入下一步：")
        info(r'  python train\01_make_corpus.py --input "<素材>" --work "<工作目录>"')
        return 0
    warn("问题：" + "、".join(bad))
    warn("**建议先换素材**，否则后面几小时的人工标注会白做。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
