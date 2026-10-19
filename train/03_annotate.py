"""④ 人工标注 —— 启动 SubFix 网页界面。★ 这一步不能跳。

    python train\\03_annotate.py --list "data\\<名字>_train_clean.list"

会在 **http://127.0.0.1:9871** 起一个网页，逐条听 + 改文本 + 删劣质片段。

---

## 怎么标

对每一条：

1. **听** —— 这段在说什么
2. **改** —— ASR 会听错同音字、漏气口，改成你听到的
3. **删** —— 听不清、有杂音、有 BGM 残留、明显念错的重来
4. **点 `Submit Text`** —— **不点不落盘**（这是 SubFix 的坑，会回滚）

## 为什么这么重要

校对质量**直接决定**最终音色。ASR 的错标不会报错，只会让某个字
**永远念不对** —— 属于"静默"损伤，事后很难查。

本项目 189 条里，人工改了 74 处、删了 4 条。
典型修正：`剧本上` → `剧本杀`、`推理吧` → `推理本`。

## 标完

关掉命令行窗口（或 Ctrl+C），然后跑：

    python train\\04_verify_list.py --list "data\\<名字>_train_clean.list"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from _common import (EMBED_PY, GSV_ROOT, ROOT, check_env, info, ok, step,
                     title, warn)


def main() -> int:
    ap = argparse.ArgumentParser(description="启动标注界面")
    ap.add_argument("--list", required=True, help="要校对的 .list")
    ap.add_argument("--port", type=int, default=9871)
    args = ap.parse_args()

    title("人工标注（SubFix）")
    if not check_env(need_embed=True):
        return 2

    lst = Path(args.list)
    if not lst.is_absolute():
        lst = ROOT / lst
    if not lst.exists():
        warn(f"列表不存在：{lst}")
        return 2

    launcher = ROOT / "tools" / "subfix_launch.py"
    if not launcher.exists():
        warn(f"找不到 {launcher}")
        return 2

    n = len([l for l in lst.read_text(encoding="utf-8").splitlines() if l.strip()])
    info(f"待标注 {n} 条")

    step("1", "把音频格式对齐成 32kHz 单声道（SubFix 只吃这个）")
    info("  （切片本来就是这个格式，一般会直接跳过）")

    step("2", f"启动网页界面（端口 {args.port}）")
    proc = subprocess.Popen(
        [str(EMBED_PY), "-u", "-s", str(launcher),
         "--list", str(lst), "--port", str(args.port)],
        cwd=str(ROOT),
    )

    url = f"http://127.0.0.1:{args.port}"
    ok_url = False
    for _ in range(60):
        time.sleep(2)
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    ok_url = True
                    break
        except Exception:
            continue

    print()
    if ok_url:
        title("界面已就绪")
        print(f"""
  用浏览器打开：  {url}

  标注要点：
    · 逐条听，把文本改成你听到的内容
    · 听不清 / 有杂音 / 有 BGM 残留的，删掉
    · **每条都要点 `Submit Text`** —— 不点会回滚！

  标完回来这里按 Ctrl+C，然后跑：
    python train\\04_verify_list.py --list "{args.list}"
""")
        try:
            proc.wait()
        except KeyboardInterrupt:
            info("收到中断，关掉界面…")
            proc.terminate()
    else:
        warn("界面没起来，看上面的输出（可能端口被占）")
        proc.terminate()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
