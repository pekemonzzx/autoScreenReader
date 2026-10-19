"""定位瓶颈：对比「源素材 -> 训练切片 -> 合成输出」三段的频谱。

如果输出比训练语料暗 -> 管线还在丢东西，有优化空间。
如果输出和语料一样亮 -> 管线不丢东西了，天花板就是**素材本身**。
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "train"))

from spectrum_db import avg_psd, load  # noqa: E402
from _common import sample_source  # noqa: E402

BANDS = [(300, 1000), (1000, 2000), (2000, 3000), (3000, 4000),
         (4000, 5000), (5000, 6000), (6000, 8000), (8000, 11000), (11000, 16000)]


def report(label: str, files, limit: int = 40):
    files = [Path(f) for f in files][:limit]
    acc = None
    fr = None
    n = 0
    srs = set()
    for f in files:
        try:
            sr, a = load(f)
        except Exception:
            continue
        if len(a) < 8192:
            continue
        srs.add(sr)
        psd, f2 = avg_psd(a, sr)
        if psd is None:
            continue
        acc = psd if acc is None else acc + psd
        fr = f2
        n += 1
    if acc is None:
        print(f"  {label}: 无样本")
        return None
    psd = acc / n
    ref = psd[(fr >= 300) & (fr < 1000)].mean()
    out = {}
    print(f"\n  {label}")
    print(f"    采样率 {sorted(srs)}  样本 {n}")
    for lo, hi in BANDS:
        m = (fr >= lo) & (fr < hi)
        if not m.any():
            continue
        v = psd[m].mean()
        db = 10 * np.log10(v / ref) if v > 0 and ref > 0 else -99
        out[(lo, hi)] = db
        print(f"      {lo/1000:>5.1f}-{hi/1000:<5.1f}kHz {db:>8.1f} dB  "
              + "#" * int(max(0, (db + 60) / 2)))
    return out


print("=" * 78)
print("三段对比：源素材 / 训练切片 / 合成输出")
print("=" * 78)

_sample = sample_source(ROOT)
src = report(f"① 源素材（{_sample.stem if _sample else '未找到'} 的 48kHz 音轨前 60 秒）",
             [ROOT / "data" / "trunc_test" / "t1.wav"] if False else [])
# 源素材用之前抽的 48k 片段；没有就跳过
m4a_wav = ROOT / "data" / "_src48k_new.wav"
if m4a_wav.exists():
    src = report("① 源素材（48kHz 音轨）", [m4a_wav])

slices = sorted(Path(r"D:\GPT-SoVITS\work\voice02hq\02_sliced").glob("*.wav"))
train = report("② 训练切片（voice02hq）", slices, limit=60)

outs = sorted((ROOT / "data").glob("newref_*.wav")) + \
       sorted((ROOT / "data" / "trunc_test").glob("t*.wav"))
synth = report("③ 合成输出", outs, limit=20)

print("\n" + "=" * 78)
print("结论")
print("=" * 78)
if train and synth:
    for band in [(4000, 5000), (6000, 8000), (8000, 11000), (11000, 16000)]:
        t = train.get(band)
        s = synth.get(band)
        if t is None or s is None:
            continue
        diff = s - t
        verdict = ("输出更亮" if diff > 1.5 else
                   "基本一致（管线不丢东西）" if diff > -1.5 else "输出偏暗")
        print(f"  {band[0]/1000:.0f}-{band[1]/1000:.0f}kHz  "
              f"语料 {t:6.1f} dB  输出 {s:6.1f} dB  差 {diff:+5.1f} dB  -> {verdict}")
print("=" * 78)
