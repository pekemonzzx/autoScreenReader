"""验证打包后的 exe 里，screenreader.runtime 的字节码确实引用了 subprocess。

为什么非要这么验：原来的 bug 恰恰是"源码看着没问题、打包后一跑就炸"
（`runtime.py` 用了 `subprocess.STDOUT` 但没 import）。
只检查源码是不够的，必须看**真正打进包里的那份字节码**。
"""

import io
import marshal
import sys
import types
import zipfile
from pathlib import Path

EXE = Path(sys.argv[1] if len(sys.argv) > 1 else r"build\installer_tree\读屏台.exe")
print(f"  exe: {EXE}  ({EXE.stat().st_size / 1e6:.1f} MB)")

# PyInstaller onedir：PYZ 在 exe 内部的 CArchive 里
from PyInstaller.archive.readers import CArchiveReader  # noqa: E402

car = CArchiveReader(str(EXE))
names = [n for n in car.toc]
print(f"  CArchive 条目 {len(names)} 个")
pyz_name = next((n for n in names if n.endswith(".pyz")), None)
print(f"  PYZ: {pyz_name}")
if not pyz_name:
    print("  没找到 PYZ")
    raise SystemExit(1)

blob = car.extract(pyz_name)
print(f"  PYZ 解出 {len(blob) / 1e6:.1f} MB")

# PYZ 本身是个 zip
z = zipfile.ZipFile(io.BytesIO(blob))
entries = z.namelist()
print(f"  PYZ 里 {len(entries)} 个模块")

want = "screenreader/runtime"
match = [e for e in entries if e.startswith(want) and e.endswith(".pyc")]
print(f"  找到 {match}")
if not match:
    print("  !! 包里没有 screenreader.runtime")
    raise SystemExit(1)

code = marshal.loads(z.read(match[0])[16:])   # 跳过 pyc 头


def walk(c, path=""):
    yield path, c
    for k in c.co_consts:
        if isinstance(k, types.CodeType):
            yield from walk(k, f"{path}/{k.co_name}")


all_names = set()
for _p, c in walk(code):
    all_names |= set(c.co_names)

print(f"\n  runtime 模块里引用到的全局名 {len(all_names)} 个")
hits = sorted(n for n in all_names if n in ("subprocess", "STDOUT"))
print(f"  关键名字: {hits}")

# 再逐层看 STDOUT 出现在哪个函数里
print("\n  用到 subprocess 的函数：")
found = False
for p, c in walk(code):
    if "subprocess" in c.co_names:
        extra = [n for n in c.co_names if n == "STDOUT"]
        print(f"    {p or '<模块级>'}   STDOUT={'是' if extra else '否'}")
        found = True
if not found:
    print("    一个都没有")

print()
if "subprocess" in all_names:
    print("  ✅ 包里的 runtime 引用 subprocess —— 修复已生效")
    raise SystemExit(0)
print("  ❌ 包里没有 subprocess —— 修复没进包")
raise SystemExit(1)
