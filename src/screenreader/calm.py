"""挑「叙述型、情绪平和」的参考音频。

为什么要专门做这个：GPT-SoVITS 克隆时会**连情绪一起学**。
参考音频如果是"太太好玩了！我第一次玩这么好玩的剧本杀！"这种兴奋语气，
之后念任何正常文本都会带着那股劲儿 —— 用户实测就是这个反馈。

平静的判据分两层：

**文字层**
  * 带 ！？ 的句子扣分（问句/感叹句情绪都高）
  * 带"哈哈/啊啊/天哪"这类语气词的扣分

**声学层**（更可靠）
  * **基频波动**：情绪高时音高上下跳，f0 变异系数大
  * **能量包络波动**：情绪高时忽大忽小
  * **语速**：太快显得急
  * 响度太低（听不清）或削波（喊）都要扣分

声学部分不依赖 librosa，用 numpy 自相关估基频，能在打包环境里跑。
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

# 情绪标记
_EXCITED_CHARS = "！!？?"
_EXCITED_WORDS = ("哈哈", "啊啊", "哎呀", "哇", "天哪", "我天", "卧槽", "笑死",
                  "也太", "真的", "超级", "无敌")


def _read_wav(path) -> Tuple[Optional[int], Optional[np.ndarray]]:
    import wave

    try:
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    except Exception:
        return None, None
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return sr, a


def pitch_track(a: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 350.0,
                frame: int = 1024, hop: int = 256, thresh: float = 0.3) -> np.ndarray:
    """自相关法逐帧估基频，只返回**有声音帧**的 f0。"""
    if len(a) < frame * 2:
        return np.array([])
    a = a - a.mean()
    win = np.hanning(frame)
    lo, hi = int(sr / fmax), int(sr / fmin)
    if hi <= lo:
        return np.array([])

    f0s: List[float] = []
    for i in range(0, len(a) - frame, hop):
        seg = a[i:i + frame] * win
        rms = float(np.sqrt((seg ** 2).mean()))
        if rms < 120:                      # 静音/轻音不参与
            continue
        spec = np.fft.rfft(seg, 2 * frame)
        ac = np.fft.irfft(spec * np.conj(spec))[:hi]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg_ac = ac[lo:hi]
        if not len(seg_ac):
            continue
        k = int(np.argmax(seg_ac))
        if seg_ac[k] < thresh:             # 不像周期信号（清音）
            continue
        f0s.append(sr / (lo + k))
    return np.array(f0s)


def calm_score(path, text: str = "") -> Tuple[float, dict]:
    """返回 (评分, 明细)。**分越高越平和**，适合当叙述型参考。"""
    detail: dict = {}
    sr, a = _read_wav(path)
    if sr is None or a is None or len(a) < sr // 2:
        return -1e9, {"error": "读不了"}

    dur = len(a) / sr
    detail["dur"] = round(dur, 2)

    # ---- 文字层 ----
    pen = 0.0
    n_exc = sum(text.count(c) for c in _EXCITED_CHARS)
    pen += n_exc * 4.0
    for w in _EXCITED_WORDS:
        if w in text:
            pen += 2.0
    detail["excited_marks"] = n_exc

    # ---- 声学层 ----
    # 能量包络起伏
    fr = max(1, sr // 100)
    env = np.array([np.sqrt((a[i:i + fr] ** 2).mean())
                    for i in range(0, max(1, len(a) - fr), fr)])
    env = env[env > 60]
    if len(env) < 5:
        return -1e9, {"error": "几乎没声音"}
    env_cv = float(env.std() / (env.mean() + 1e-9))
    detail["env_cv"] = round(env_cv, 3)

    # 基频波动
    f0 = pitch_track(a, sr)
    if len(f0) >= 8:
        f0_cv = float(f0.std() / (f0.mean() + 1e-9))
        detail["f0_med"] = round(float(np.median(f0)), 1)
        detail["f0_cv"] = round(f0_cv, 3)
    else:
        f0_cv = 0.5
        detail["f0_cv"] = None

    # 响度 / 削波
    rms = float(np.sqrt((a ** 2).mean()))
    clip = float((np.abs(a) >= 32000).mean())
    detail["rms"] = round(rms)
    detail["clip"] = round(clip * 100, 3)

    # 语速（字/秒）
    rate = len(re.sub(r"\s", "", text)) / dur if dur > 0 else 0
    detail["rate"] = round(rate, 2)

    score = 100.0
    score -= f0_cv * 120.0          # 音高越稳越好（权重最大）
    score -= env_cv * 40.0          # 能量越稳越好
    score -= abs(rate - 4.6) * 4.0  # 语速靠近中位 4.6 字/秒
    score -= pen                    # 文字情绪
    if rms < 500:
        score -= 25                 # 太轻，不像正常朗读
    if clip > 0:
        score -= 60                 # 削波=喊
    return score, detail


def rank_candidates(items: List[Tuple], text_of=None) -> List[Tuple]:
    """给候选排序。items 是 [(score, ...)] 之类，这里只是薄封装。"""
    return sorted(items, key=lambda x: -x[0])
