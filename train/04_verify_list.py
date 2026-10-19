"""⑤ 训练前严格校验 —— **别跳，这一步能挡住静默毁模型的问题**。

    python train\\04_verify_list.py --list "data\\<名字>_train_clean.list"

检查：

  1. 每行格式合法（wav|说话人|语言|文本）
  2. 音频存在、可读、32kHz 单声道
  3. 文本非空、无首尾空格、无残留标记
  4. 没有重复引用同一个 wav
  5. **语速合理** —— 错位最灵敏的信号
  6. 和 `.raw` 备份对比，看删了哪些、改了多少处

为什么第 5 条最关键：文本和音频一旦**错位**，训练会**静默**学错 ——
不报任何错，只是韵律全乱。而错位在语速上表现得很明显：
正常中文 4~6 字/秒，错位会立刻变得离谱。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import ROOT, VENV_PY, info, ok, run, step, title, warn


def main() -> int:
    ap = argparse.ArgumentParser(description="训练前校验标注列表")
    ap.add_argument("--list", required=True)
    args = ap.parse_args()

    title("训练前校验")
    tool = ROOT / "tools" / "verify_alignment.py"
    lst = Path(args.list)
    if not lst.is_absolute():
        lst = ROOT / lst

    if not lst.exists():
        warn(f"列表不存在：{lst}")
        return 2

    if not tool.exists():
        # 兜底：至少做个基本检查
        warn(f"找不到 {tool}，做基本检查")
        rows = [l for l in lst.read_text(encoding="utf-8").splitlines() if l.strip()]
        bad = [l for l in rows if len(l.split("|")) < 4]
        info(f"{len(rows)} 条，格式异常 {len(bad)} 条")
        return 1 if bad else 0

    step("跑校验", "格式 / 文件 / 语速 / 重复 / 与 .raw 对比")
    rc = run([VENV_PY, "-u", tool, str(lst)],
             env_extra={"PYTHONPATH": str(ROOT / "src")}, cwd=str(ROOT))

    print()
    if rc == 0:
        title("通过")
        ok("可以进入训练：")
        info(r'  python train\05_train_voice.py --name <名字> '
             f'--list "{args.list}" --wav-dir "<切片目录>" --calm')
    else:
        title("有问题")
        warn("先按上面的提示修好再训练 —— 错标会静默毁掉某个字的发音")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
