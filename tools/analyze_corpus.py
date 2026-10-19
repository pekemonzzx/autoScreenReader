"""统计新语料的时长/字数/语速分布，用来决定过滤规则。"""

import statistics
import sys
import wave
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\GPT-SoVITS\work\voice02")
rows = [l for l in (W / "train.list").read_text(encoding="utf-8").splitlines() if l.strip()]


def dur(p: Path) -> float:
    try:
        with wave.open(str(p), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


items = []
for line in rows:
    parts = line.split("|")
    if len(parts) < 4:
        continue
    p = Path(parts[0])
    text = "|".join(parts[3:]).strip()
    d = dur(p)
    if d > 0:
        items.append((d, len(text), text))

print(f"总 {len(items)} 条")
durs = [i[0] for i in items]
chars = [i[1] for i in items]
print(f"  时长  中位 {statistics.median(durs):.2f}s  范围 {min(durs):.2f}~{max(durs):.2f}")
print(f"  字数  中位 {statistics.median(chars):.0f}    范围 {min(chars)}~{max(chars)}")
print()
print("  字数分布:")
for lo, hi in [(0, 4), (4, 10), (10, 20), (20, 25), (25, 30), (30, 40), (40, 60), (60, 999)]:
    n = sum(1 for c in chars if lo <= c < hi)
    if n:
        print(f"    {lo:>3}-{hi:<3} 字: {n:>3} 条  " + "#" * min(n, 60))
print()
rates = [c / d for d, c, _ in items if d > 0]
print(f"  语速（字/秒）中位 {statistics.median(rates):.1f}  范围 {min(rates):.1f}~{max(rates):.1f}")
print()

over = [(d, c, t) for d, c, t in items if c > 25]
over.sort(key=lambda x: -x[1])
print(f"  超 25 字的共 {len(over)} 条，抽样看是不是真该剔:")
for d, c, t in over[:10]:
    print(f"    {d:5.2f}s {c:>3}字 {c / d:4.1f}字/秒  {t[:56]}")
print()
normal = [1 for d, c, _ in over if 2.0 <= c / d <= 9.0]
print(f"  其中语速正常(2~9 字/秒)的: {len(normal)} 条  <- 这些其实是好数据")
print()

# 时长 + 语速联合过滤能保留多少
def keep(d, c):
    if d < 1.0 or d > 12.0:
        return False
    if c < 4:
        return False
    if c / d > 9.0:          # 语速快得不可能 -> ASR 幻觉
        return False
    if c / d < 1.2 and c > 12:  # 大半是静音/环境音
        return False
    if c > 60:
        return False          # 实在太长
    return True


kept = [(d, c) for d, c, _ in items if keep(d, c)]
print(f"  改用「时长+语速」联合规则: 保留 {len(kept)} 条，"
      f"总时长 {sum(d for d, _ in kept) / 60:.1f} 分钟")
