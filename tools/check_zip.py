"""检查 portable zip 的完整性。"""

import sys
import zipfile
from pathlib import Path

z = Path(sys.argv[1] if len(sys.argv) > 1
         else r"build\installer\AutoScreenReader-0.6.3-portable.zip")
print(f"  文件: {z}")
print(f"  大小: {z.stat().st_size:,} 字节  ({z.stat().st_size / 1e6:.2f} MB)")

try:
    with zipfile.ZipFile(z) as f:
        info = f.infolist()
        total = sum(i.file_size for i in info)
        print(f"  条目数: {len(info)}")
        print(f"  解压后总大小: {total / 1e6:.1f} MB")
        print("  前 12 个条目:")
        for i in info[:12]:
            print(f"    {i.filename:<50} {i.file_size:>12,}  压缩后 {i.compress_size:>10,}")
        bad = f.testzip()
        print(f"  完整性: {'OK' if bad is None else '损坏于 ' + str(bad)}")
        names = [i.filename for i in info]
        for key in ("读屏台.exe", "_internal", "data/config.json", "voices", "start.bat"):
            hit = [n for n in names if key in n]
            print(f"  含 {key:<22}: {len(hit)} 个")
except Exception as e:
    print(f"  !! 打不开: {e}")
