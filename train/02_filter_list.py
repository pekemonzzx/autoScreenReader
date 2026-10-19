"""③ 机械过滤：按时长和语速筛掉不合格的片段。

    python train\\02_filter_list.py --list "<work>\\train.list" ^
        --out "data\\<名字>_train_clean.list" --wav-dir "<work>\\02_sliced"

筛掉四类：

  * 太短（< 1 秒）——克隆不稳
  * 太长（> 12 秒）——训练慢且容易断气
  * 语速 > 9 字/秒 —— 基本是 ASR 幻觉（正常中文 4~6 字/秒）
  * 语速 < 1.2 字/秒且字数 > 12 —— 大半是静音/环境音

**不要用"字数上限"来筛**。以前默认卡 25 字，白白扔掉了 25% 的正常连续语音；
实测那些 7~9 秒、5 字/秒 的样本对学韵律反而更有用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import ROOT, VENV_PY, info, ok, run, step, title, warn


def main() -> int:
    ap = argparse.ArgumentParser(description="机械过滤标注列表")
    ap.add_argument("--list", required=True, help="输入的 train.list")
    ap.add_argument("--out", required=True, help="输出的 clean.list（建议放 data\\ 下）")
    ap.add_argument("--wav-dir", default=None, help="切片目录（list 里是相对路径时要用）")
    ap.add_argument("--min-chars", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=60,
                    help="绝对上限，防止异常长的片段（默认 60，不是 25）")
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=12.0)
    ap.add_argument("--max-rate", type=float, default=9.0, help="字数/秒 上限")
    ap.add_argument("--min-rate", type=float, default=1.2, help="字数/秒 下限")
    args = ap.parse_args()

    title("机械过滤")
    tool = ROOT / "tools" / "filter_list.py"
    if not tool.exists():
        warn(f"找不到 {tool}")
        return 2
    src = Path(args.list)
    if not src.exists():
        warn(f"输入不存在：{src}")
        return 2

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [VENV_PY, "-u", tool,
           "--list", str(src), "--out", str(out),
           "--min-chars", str(args.min_chars),
           "--max-chars", str(args.max_chars),
           "--min-sec", str(args.min_sec),
           "--max-sec", str(args.max_sec),
           "--max-rate", str(args.max_rate),
           "--min-rate", str(args.min_rate)]
    if args.wav_dir:
        cmd += ["--wav-dir", str(args.wav_dir)]

    step("跑过滤", "按时长 + 语速判定（不用字数上限）")
    rc = run(cmd, env_extra={"PYTHONPATH": str(ROOT / "src")}, cwd=str(ROOT))
    if rc != 0:
        return rc

    if not out.exists():
        warn("没生成输出")
        return 1
    n = len([l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()])

    # 留一份 .raw 备份：人工标注前原样，之后用来对比"改了什么"
    raw = Path(str(out) + ".raw")
    raw.write_bytes(out.read_bytes())

    print()
    title("完成")
    ok(f"保留 {n} 条")
    info(f"待标注：{out}")
    info(f"备份  ：{raw}")
    print()
    info("下一步（★ 人工标注，不能跳）：")
    info(f'  python train\\03_annotate.py --list "{out}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
