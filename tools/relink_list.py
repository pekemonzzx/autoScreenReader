"""把标注列表里的音频路径换到另一个切片目录（文件名保持一致）。

用途：用高音质素材按**相同偏移量**重切之后，切片文件名没变、
人工校对的文本也没变，只是文件所在目录变了 —— 这个脚本负责把路径改过去，
这样 185 条人工校对成果可以原样复用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="重指向列表里的音频目录")
    ap.add_argument("--list", required=True)
    ap.add_argument("--wav-dir", required=True, help="新的切片目录")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = Path(args.list)
    wav_dir = Path(args.wav_dir)
    out = Path(args.out)

    rows = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    new_rows = []
    missing = []
    for line in rows:
        parts = line.split("|")
        if len(parts) < 4:
            continue
        name = Path(parts[0]).name
        p = wav_dir / name
        if not p.exists():
            missing.append(name)
            continue
        new_rows.append("|".join([str(p)] + parts[1:]))

    out.write_text("\n".join(new_rows) + "\n", encoding="utf-8")
    print(f"  {len(rows)} 条 -> 写出 {len(new_rows)} 条")
    print(f"  新目录 {wav_dir}")
    print(f"  输出   {out}")
    if missing:
        print(f"  !! 有 {len(missing)} 条在新目录里找不到：")
        for m in missing[:8]:
            print(f"     {m}")
        return 1
    print("  全部找得到 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
