"""一键串联 ①②③（素材体检 → 造语料 → 机械过滤），到"人工标注"前停下。

    python train\\run.py --input "<素材>" --work "<工作目录>" --name <声库名>

会在第 ④ 步（人工标注）停下，把要跑的命令打给你 ——
因为那一步必须人来听、人来改，没法自动化。

    --until check    只做素材体检
    --until corpus   做到造语料
    --until filter   做到过滤（默认）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import ROOT, VENV_PY, info, ok, run, step, title, warn


def main() -> int:
    ap = argparse.ArgumentParser(description="一键串联训练前置流程")
    ap.add_argument("--input", required=True, help="素材文件")
    ap.add_argument("--work", required=True, help="工作目录")
    ap.add_argument("--name", required=True, help="声库名（用于生成 list 文件名）")
    ap.add_argument("--segments", default=None, help='只取指定时间段，如 "0:00-24:40"')
    ap.add_argument("--no-auto-split", action="store_true",
                    help="关掉自动分段（默认开：按可用内存自动决定切几段）")
    ap.add_argument("--until", default="filter",
                    choices=["check", "corpus", "filter"])
    args = ap.parse_args()

    train_dir = Path(__file__).resolve().parent
    py = VENV_PY
    env = {"PYTHONPATH": str(ROOT / "src")}

    title(f"一键流程：{args.name}")
    info(f"素材 {args.input}")
    info(f"工作目录 {args.work}")
    info(f"做到 {args.until} 为止")

    # ---- ① 素材体检 ----
    step("①", "素材体检")
    rc = run([py, "-u", str(train_dir / "00_check_source.py"), args.input],
             env_extra=env, cwd=str(ROOT), check=False)
    if rc != 0:
        warn("素材体检没过 —— 建议先换素材，别急着往下做")
        warn("（如果坚持继续，加 --until corpus 跳过检查）")
        return rc
    if args.until == "check":
        return 0

    # ---- ② 造语料 ----
    step("②", "造语料（抽音轨 → UVR5 → 切片 → ASR）")
    cmd = [py, "-u", str(train_dir / "01_make_corpus.py"),
           "--input", args.input, "--work", args.work]
    if args.segments:
        cmd += ["--segments", args.segments]
    # **不管有没有 --segments 都要 open 自动分段**：
    # 用户给的段本身可能就太长（0:00-20:00 约需 15 GB），
    # 01_make_corpus 会在每一段内部再按内存细分。
    if not args.no_auto_split:
        cmd.append("--auto-split")
    rc = run(cmd, env_extra=env, cwd=str(ROOT), check=False)
    if rc != 0:
        warn("造语料失败，看上面的输出")
        return rc
    if args.until == "corpus":
        return 0

    # ---- ③ 机械过滤 ----
    step("③", "机械过滤")
    lst = Path(args.work) / "train.list"
    out = f"data\\{args.name}_train_clean.list"
    cmd = [py, "-u", str(train_dir / "02_filter_list.py"),
           "--list", str(lst), "--out", out,
           "--wav-dir", str(Path(args.work) / "02_sliced")]
    rc = run(cmd, env_extra=env, cwd=str(ROOT), check=False)
    if rc != 0:
        return rc

    print()
    title("前置流程完成 —— 接下来必须人工")
    print(f"""
  现在轮到你了（★ 这一步不能自动化）：

    python train\\03_annotate.py --list "{out}"

  浏览器会打开 http://127.0.0.1:9871，逐条：
    1. 听这段在说什么
    2. 把文本改成你听到的（ASR 会听错同音字）
    3. 听不清 / 有杂音 / 有 BGM 残留的 —— 删掉
    4. **每条都要点 `Submit Text`**（不点会回滚！）

  标完按 Ctrl+C 回来，继续：

    python train\\04_verify_list.py --list "{out}"
    python train\\05_train_voice.py --name {args.name} --list "{out}" ^
        --wav-dir "{Path(args.work) / '02_sliced'}" --calm
    python train\\07_pack_voice.py --name {args.name}
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
