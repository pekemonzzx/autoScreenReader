"""⑥ 训练声库：建档案 → 挑参考音频 → 训练 → 刷新 → 自检。

    python train\\05_train_voice.py --name 我的声库 ^
        --list "data\\我的声库_train_clean.list" ^
        --wav-dir "D:\\GPT-SoVITS\\work\\我的声库\\02_sliced" --calm

约 3~5 分钟（含预处理）。产物：

    D:\\GPT-SoVITS\\GPT-SoVITS-v2pro-20250604\\
        SoVITS_weights_v2ProPlus\\<名字>_e8_s256.pth
        GPT_weights_v2ProPlus\\<名字>-e12.ckpt
    voices\\<名字>\\            ← 自包含档案（含参考音频和权重指向）

**为什么必须用整合包的内嵌 Python**：训练脚本要 torch，
而且它内部会用 `sys.executable` 去拉子进程，用错解释器会报
`No module named screenreader`（内嵌 Python 是隔离模式，PYTHONPATH 无效，
所以走 `tools/run_with_embedded.py` 这个启动器）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (EMBED_PY, ROOT, VENV_PY, check_env, info, ok, run, step,
                     title, warn)

RUNNER = ROOT / "tools" / "run_with_embedded.py"


def voice_cmd(embed: bool, *args) -> list:
    """拼一条 screenreader voice 命令。

    带 `-u`：训练很慢，输出被块缓冲的话会半天看不到一行，像卡死。
    """
    if embed:
        return [EMBED_PY, "-u", "-s", RUNNER, "voice", *args]
    return [VENV_PY, "-u", "-m", "screenreader", "voice", *args]


def main() -> int:
    ap = argparse.ArgumentParser(description="训练声库")
    ap.add_argument("--name", required=True, help="声库名（英文/数字，用作实验名）")
    ap.add_argument("--list", required=True, help="校验过的 train_clean.list")
    ap.add_argument("--wav-dir", required=True, help="切片目录")
    ap.add_argument("--display", default="", help="显示名（默认用声库名）")
    ap.add_argument("--source", default="", help="来源备注，会写进档案")
    ap.add_argument("--version", default="v2ProPlus")
    ap.add_argument("--calm", action="store_true", default=True,
                    help="按叙述型/平和度挑参考音频（默认开）")
    ap.add_argument("--no-calm", dest="calm", action="store_false")
    ap.add_argument("--min-sec", type=float, default=3.5, help="参考音频最短")
    ap.add_argument("--max-sec", type=float, default=6.0, help="参考音频最长")
    ap.add_argument("--s2-batch", type=int, default=6)
    ap.add_argument("--s2-epoch", type=int, default=8)
    ap.add_argument("--s1-batch", type=int, default=8)
    ap.add_argument("--s1-epoch", type=int, default=15)
    args = ap.parse_args()

    title(f"训练声库「{args.name}」")
    if not check_env(need_embed=True):
        return 2
    if not RUNNER.exists():
        warn(f"找不到启动器 {RUNNER}")
        return 2

    lst = Path(args.list)
    if not lst.is_absolute():
        lst = ROOT / lst
    wav = Path(args.wav_dir)
    if not lst.exists():
        warn(f"标注不存在：{lst}")
        return 2
    if not wav.exists():
        warn(f"切片目录不存在：{wav}")
        return 2

    n = len([l for l in lst.read_text(encoding="utf-8").splitlines() if l.strip()])
    info(f"标注 {n} 条")
    info(f"切片 {wav}")
    if n < 30:
        warn(f"只有 {n} 条，偏少。建议至少 50~100 条（3 分钟以上）")

    env = {"PYTHONPATH": str(ROOT / "src")}

    # ---- 1) 建档案 ----
    step("1/5", "建声库档案")
    from screenreader.config import AppConfig
    from screenreader.voice import VoiceStore

    _store = VoiceStore(AppConfig.load().voices_root())
    if _store.exists(args.name):
        # 迭代重跑时档案已经在了 —— 直接复用，别报 FileExistsError 中断
        info(f"声库「{args.name}」已存在，复用（要重来先删 voices\\{args.name}）")
    else:
        cmd = voice_cmd(True, "new", args.name,
                        "--display", args.display or args.name,
                        "--version", args.version,
                        "--list", str(lst), "--wav-dir", str(wav))
        if args.source:
            cmd += ["--source", args.source]
        if run(cmd, env_extra=env, cwd=str(ROOT), check=False) != 0:
            warn("建档案失败")
            return 1

    # ---- 2) 挑参考音频 ----
    step("2/5", "挑参考音频")
    cmd = voice_cmd(True, "pick-ref", args.name,
                    "--min-sec", str(args.min_sec), "--max-sec", str(args.max_sec))
    if args.calm:
        cmd.append("--calm")
    run(cmd, env_extra=env, cwd=str(ROOT), check=False)

    # ---- 3) 训练 ----
    step("3/5", "训练（1ABC 预处理 + SoVITS + GPT）")
    info("  显存不够会在这里报 CUDA out of memory —— 关掉游戏再重跑即可")
    cmd = voice_cmd(True, "train", args.name,
                    "--s2-batch", str(args.s2_batch), "--s2-epoch", str(args.s2_epoch),
                    "--s1-batch", str(args.s1_batch), "--s1-epoch", str(args.s1_epoch))
    if run(cmd, env_extra=env, cwd=str(ROOT), check=False) != 0:
        warn("训练失败，看上面的输出")
        warn("  显存不足 -> 关游戏，或加大 --s2-batch 之外的参数（减小 batch）")
        return 1

    # ---- 4) 刷新权重指向 ----
    step("4/5", "刷新权重指向 + 生成推理配置")
    run(voice_cmd(False, "refresh", args.name), env_extra=env, cwd=str(ROOT), check=False)

    # ---- 5) 自检 ----
    step("5/5", "自检")
    run(voice_cmd(False, "verify", args.name), env_extra=env, cwd=str(ROOT), check=False)

    print()
    title("完成")
    ok(f"声库「{args.name}」已训练好")
    info(f"档案：{ROOT / 'voices' / args.name}")
    print()
    info("下一步（试听 + 打包）：")
    info(f'  python train\\07_pack_voice.py --name {args.name}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
