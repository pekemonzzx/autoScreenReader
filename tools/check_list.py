"""校验训练标注列表：音频是否齐全、格式是否统一、时长统计。"""

import sys
import wave
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/voice02_train_clean.list")
rows = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

bad = []
total = 0.0
srs = {}
for line in rows:
    parts = line.split("|")
    if len(parts) < 4:
        bad.append(("格式错误", line[:60]))
        continue
    f = Path(parts[0])
    if not f.exists():
        bad.append(("缺失", f.name))
        continue
    try:
        with wave.open(str(f), "rb") as w:
            d = w.getnframes() / w.getframerate()
            sr = w.getframerate()
            ch = w.getnchannels()
    except Exception as e:
        bad.append(("读不了", f"{f.name} {e}"))
        continue
    total += d
    srs[(sr, ch)] = srs.get((sr, ch), 0) + 1

print(f"  {len(rows)} 条，问题 {len(bad)} 条")
for k, v in bad[:8]:
    print(f"    [{k}] {v}")
print(f"  总时长 {total / 60:.1f} 分钟，平均 {total / max(1, len(rows)):.2f}s")
print(f"  采样率/声道分布: {srs}")
print(f"  文本样例:")
for line in rows[:3]:
    print(f"    {line.split('|')[3]}")
