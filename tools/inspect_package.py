"""检查打包后的 exe 里到底塞了哪些 screenreader 模块。

PyInstaller 把纯 Python 模块压缩进 exe 尾部的 PYZ 归档，
所以直接用记事本/字符串搜索是搜不到的 —— 必须解归档。
"""

import re
import struct
import sys
import zlib
from pathlib import Path

exe = Path(sys.argv[1] if len(sys.argv) > 1 else r"build\installer_tree\读屏台.exe")
data = exe.read_bytes()
print(f"  exe 大小 {len(data) / 1e6:.1f} MB")

MAGIC = b"MEI\014\013\012\013\016"
pos = data.rfind(MAGIC)
if pos < 0:
    print("  没找到 PyInstaller 归档魔数")
    raise SystemExit(1)
print(f"  归档魔数在 {pos}")

# 归档：MAGIC + 目录长度(4, 大端) + zlib(marshal(目录))
off = pos + len(MAGIC)
(n,) = struct.unpack("!i", data[off:off + 4])
blob = data[off + 4: off + 4 + n]
try:
    toc = zlib.decompress(blob)
    print(f"  目录解压成功：{len(toc)} 字节")
except Exception as e:
    print(f"  目录解压失败：{e}")
    raise SystemExit(1)

# 目录里是 marshal 的列表；这里不反序列化，直接找模块名字符串
names = sorted({m.decode("utf-8", "replace")
                for m in re.findall(rb"[A-Za-z_][\w.]{2,60}", toc)
                if b"screenreader" in m})
print(f"\n  archive 目录里的 screenreader 条目（{len(names)}）：")
for nm in names:
    print(f"    {nm}")

# PYZ 本体也搜一遍（PYZ 内部的 TOC 是每个模块 zlib 压缩的，逐个试解）
print("\n  扫描 PYZ 内部（找模块名）...")
hits = set()
step = 4096
for start in range(off, len(data) - 16, step):
    chunk = data[start:start + step + 65536]
    try:
        raw = zlib.decompress(chunk)
    except Exception:
        continue
    for m in re.findall(rb"screenreader[.\w]{0,40}", raw):
        hits.add(m.decode("utf-8", "replace"))
    if len(hits) > 60:
        break

print(f"  找到 {len(hits)} 个：")
for nm in sorted(hits):
    print(f"    {nm}")

want = ["screenreader.segment", "screenreader.board", "screenreader.worker",
        "screenreader.ui.main_window"]
print("\n  关键模块是否在包里：")
alltext = toc + b"".join(h.encode() for h in hits)
for w in want:
    print(f"    {w:<34} {'✅' if w.encode() in alltext else '❌'}")
