"""启动 SubFix 音频标注 WebUI（自动放行音频目录）。

为什么需要这个包装脚本：
    SubFix 用 Gradio 4 提供试听。Gradio 默认**只允许服务自身工作目录下的文件**，
    其它一律返回 403（`blocked_paths` 机制）。我们的切片放在
    `D:\\GPT-SoVITS\\work\\voice01\\02_sliced`，而 SubFix 的工作目录是整合包根目录，
    于是音频加载失败、点「Submit Text」就报错。

    解决办法是设置环境变量 `GRADIO_ALLOWED_PATHS`（Gradio 4 官方支持，逗号分隔绝对路径）。
    本脚本从 .list 里读出所有音频所在目录，自动拼好这个变量，再拉起 SubFix。

用法（必须用整合包内嵌 Python）：
    D:\\GPT-SoVITS\\GPT-SoVITS-v2pro-20250604\\runtime\\python.exe -s ^
        tools\\subfix_launch.py --list "D:\\work\\autoScreenReader\\data\\voice01_train_clean.list"

也可以直接双击 `scripts\\启动音频标注.bat`。
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

DEFAULT_GSV_ROOT = Path(r"D:\GPT-SoVITS\GPT-SoVITS-v2pro-20250604")


def main() -> int:
    ap = argparse.ArgumentParser(description="启动 SubFix 音频标注 WebUI（自动放行音频目录）")
    ap.add_argument("--list", required=True, help="要校对的 .list 路径")
    ap.add_argument("--port", type=int, default=9871)
    ap.add_argument("--gsv-root", default=str(DEFAULT_GSV_ROOT))
    args = ap.parse_args()

    root = Path(args.gsv_root).resolve()
    subfix = root / "tools" / "subfix_webui.py"
    if not subfix.exists():
        sys.exit(f"找不到 {subfix}")

    lst = Path(args.list).resolve()
    if not lst.exists():
        sys.exit(f"标注文件不存在：{lst}")

    # ---- 从 list 里收集所有音频所在目录 ----
    dirs: list[str] = []
    missing = 0
    for line in lst.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        p = Path(parts[0])
        if not p.is_absolute():
            p = lst.parent / p
        d = str(p.parent)
        if d not in dirs:
            dirs.append(d)
        if not p.exists():
            missing += 1

    if not dirs:
        sys.exit("这个 .list 里没有解析出任何音频路径，格式应为：wav路径|说话人|语言|文本")

    # 顺带把 list 自身所在目录也放行（SubFix 保存时会写回该文件）
    if str(lst.parent) not in dirs:
        dirs.append(str(lst.parent))

    os.environ["GRADIO_ALLOWED_PATHS"] = ",".join(dirs)

    print("=" * 74)
    print("SubFix 音频标注 WebUI")
    print("=" * 74)
    print(f"  标注文件 : {lst}  （{sum(1 for l in lst.read_text(encoding='utf-8').splitlines() if l.strip())} 条）")
    print(f"  放行目录 :")
    for d in dirs:
        n = len(list(Path(d).glob("*.wav"))) if Path(d).is_dir() else 0
        print(f"      {d}   （{n} 个 wav）")
    if missing:
        print(f"  !! 有 {missing} 条音频文件不存在，这些条目在界面里会加载失败")
    print(f"\n  打开浏览器访问： http://127.0.0.1:{args.port}")
    print("=" * 74)
    print("\n使用要点：")
    print("  1. 改完文本必须点「Submit Text」才会写回文件；直接翻页会回滚。")
    print("  2. 保存后 list 会被就地覆盖（文本末尾会多一个空格，训练时无影响）。")
    print("  3. 校对完重训：重跑 tools\\tts_train.py 即可（1A/1B/1C 一共约 35 秒，全跑最稳）。\n")

    # 以 __main__ 身份运行 SubFix（它靠 if __name__ == "__main__" 里的 set_global 加载 list）
    sys.argv = [
        "subfix_webui.py",
        "--load_list", str(lst),
        "--webui_port", str(args.port),
        "--is_share", "False",
    ]
    os.chdir(root)
    sys.path.insert(0, str(root))
    runpy.run_path(str(subfix), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
