"""检查各常见安装位置的可用性，并显示给用户的解释。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import screenreader.runtime as R  # noqa: E402

print(f"是否管理员: {R.is_admin()}")
sug = R.suggest_dest()
print(f"建议位置  : {sug}    剩余 {R.free_space(sug) / 1024 ** 3:.0f} GB")
print()

targets = [
    r"D:\test",
    r"C:\Program Files\AutoScreenReader\gsv",
    r"C:\Windows\gsv",
    "D:\\",
    r"C:\ProgramData\AutoScreenReader\gsv",
    str(Path.home() / "AutoScreenReader" / "gsv"),
]

for p in targets:
    path = Path(p)
    wr = R.check_writable(path)
    mark = "可写 ✔" if wr["ok"] else "不可写 ✘"
    free = R.free_space(path)
    free_s = f"{free / 1024 ** 3:.0f} GB" if free >= 0 else "?"
    print(f"[{mark}] {p}    剩余 {free_s}")
    if not wr["ok"]:
        print(f"      原因: {wr['reason']}")
        for line in R.describe_location(path).splitlines():
            print(f"      {line}")
    print()
