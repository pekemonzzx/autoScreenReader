"""训练标注列表的机械清洗。

ASR 自动标注必然有噪声，人工校对之前先做一轮机械过滤，能明显提升训练数据质量：
  * 丢掉过短（<4 字，多是"哇塞""嗯"这类语气词）和过长（>25 字，模型难对齐）的条目
  * 丢掉几乎不含中文的条目（ASR 把音效/背景人声也标了字）
  * 丢掉音频文件不存在、或时长异常的条目
  * 把音频路径统一成绝对路径

不做的事：不改文本内容本身（同音字错误必须人工听才能改）。

用法：
    .\\.venv\\Scripts\\python tools\\filter_list.py ^
        --list D:\\GPT-SoVITS\\work\\voice01\\03_list\\02_sliced.list ^
        --wav-dir D:\\GPT-SoVITS\\work\\voice01\\02_sliced ^
        --out D:\\GPT-SoVITS\\work\\voice01\\train_clean.list
"""

from __future__ import annotations

import argparse
import re
import sys
import wave
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff]")


def wav_duration(p: Path) -> float:
    try:
        with wave.open(str(p), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return -1.0


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(CJK.findall(text)) / len(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--wav-dir", default=None, help="片段相对路径的基准目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chars", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=60,
                    help="绝对上限，防止异常长的片段（默认 60，比以前的 25 宽松很多）")
    ap.add_argument("--min-cjk-ratio", type=float, default=0.5)
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=12.0)
    ap.add_argument("--max-rate", type=float, default=9.0,
                    help="字数/秒 上限，超过说明 ASR 在瞎编")
    ap.add_argument("--min-rate", type=float, default=1.2,
                    help="字数/秒 下限（长片段才检查），太低说明大半是静音")
    args = ap.parse_args()

    src = Path(args.list)
    wav_dir = Path(args.wav_dir) if args.wav_dir else src.parent
    rows = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]

    kept, reasons = [], {"格式错误": 0, "文件缺失": 0, "时长异常": 0,
                         "太短": 0, "太长": 0, "语速过快": 0, "语速过慢": 0,
                         "中文占比低": 0}
    total_sec = 0.0

    for line in rows:
        parts = line.split("|")
        if len(parts) < 4:
            reasons["格式错误"] += 1
            continue
        wav_name, speaker, lang, text = parts[0], parts[1], parts[2], "|".join(parts[3:])
        text = text.strip()

        p = Path(wav_name)
        if not p.is_absolute():
            p = wav_dir / Path(wav_name).name
        if not p.exists():
            reasons["文件缺失"] += 1
            continue

        d = wav_duration(p)
        if d <= 0 or d < args.min_sec or d > args.max_sec:
            reasons["时长异常"] += 1
            continue

        n = len(text)
        if n < args.min_chars:
            reasons["太短"] += 1
            continue
        if args.max_chars and n > args.max_chars:
            reasons["太长"] += 1
            continue
        if cjk_ratio(text) < args.min_cjk_ratio:
            reasons["中文占比低"] += 1
            continue

        # 用「语速」判定，而不是拍脑袋定字数上限。
        #
        # 教训：以前默认 max-chars=25，把 50 条（25%）正常说话全剔了 ——
        # 那些都是 7~9 秒、5 字/秒 的自然连续语音，对学韵律反而更有用。
        # 正常中文语速约 4~6 字/秒，所以真正该剔的是"快得不可能"的（ASR 幻觉）
        # 和"慢得离谱"的（大半是静音/环境音）。
        rate = n / d
        if rate > args.max_rate:
            reasons["语速过快"] += 1
            continue
        if n > 12 and rate < args.min_rate:
            reasons["语速过慢"] += 1
            continue

        total_sec += d
        kept.append(f"{p}|{speaker}|{lang}|{text}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(kept) + "\n", encoding="utf-8")

    print("=" * 70)
    print(f"输入 {len(rows)} 条  ->  保留 {len(kept)} 条  ({len(kept)/max(len(rows),1)*100:.0f}%)")
    print(f"有效音频总时长 {total_sec/60:.1f} 分钟")
    print("-" * 70)
    for k, v in reasons.items():
        if v:
            print(f"  剔除·{k:<10} {v}")
    print("-" * 70)
    print(f"输出 {out}")
    if len(kept) < 20:
        print("\n!! 保留条目过少（<20），训练效果会很差。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
