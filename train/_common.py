"""train/ 下各脚本的公共部分。

只放三样东西：路径、找 Python、打印工具。保持薄，逻辑都在各步骤里。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# 项目根目录（train/ 的上一层）
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 默认的 GPT-SoVITS 整合包位置。可以用环境变量 GPT_SOVITS_ROOT 覆盖。
GSV_ROOT = Path(os.environ.get("GPT_SOVITS_ROOT",
                               r"D:\GPT-SoVITS\GPT-SoVITS-v2pro-20250604"))

# 虚拟环境的 Python（跑界面/工具链用）
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
# 整合包内嵌 Python（跑训练/UVR5/ASR 用 —— 只有它有 torch）
EMBED_PY = GSV_ROOT / "runtime" / "python.exe"


def title(text: str) -> None:
    print("=" * 78)
    print(f"  {text}")
    print("=" * 78)


def step(n: str, text: str) -> None:
    print(f"\n[{n}] {text}")


def info(text: str = "") -> None:
    print(f"  {text}" if text else "")


def warn(text: str) -> None:
    print(f"  !! {text}")


def ok(text: str) -> None:
    print(f"  [OK] {text}")


def check_env(need_embed: bool = False) -> bool:
    """检查需要的 Python / 整合包在不在。"""
    good = True
    if not VENV_PY.exists():
        warn(f"找不到虚拟环境 Python：{VENV_PY}")
        good = False
    if need_embed:
        if not EMBED_PY.exists():
            warn(f"找不到整合包内嵌 Python：{EMBED_PY}")
            warn("（训练/UVR5/ASR 必须有 torch，只有它有）")
            warn("  用环境变量 GPT_SOVITS_ROOT 指向你的整合包目录")
            good = False
        if not (GSV_ROOT / "ffmpeg.exe").exists():
            warn(f"整合包根目录没有 ffmpeg.exe：{GSV_ROOT}")
            warn(r"  把 vendor\ffmpeg\bin\ffmpeg.exe 复制过去即可")
            good = False
    return good


def run(cmd: list, env_extra: dict = None, cwd=None, check: bool = True) -> int:
    """跑一个子进程，实时显示输出。返回退出码。"""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if env_extra:
        env.update(env_extra)
    print(f"  $ {' '.join(str(c) for c in cmd[:4])} …")
    r = subprocess.run([str(c) for c in cmd], env=env,
                       cwd=str(cwd) if cwd else None)
    if check and r.returncode != 0:
        warn(f"退出码 {r.returncode}")
    return r.returncode


def human_mb(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def sample_source(root: Path = None):
    """找一份「可用于测试/分析的源素材」，**不在代码里写死具体文件名**。

    为什么抽出来：源码里到处写 `data/BVxxxxxxxxxx_p1.f30280.m4a`，等于把
    "我的训练语料来自哪个视频"公开出去，别人也没法用自己的素材跑。
    改成按顺序查找，谁都能用：

      1. 环境变量 `SCREENREADER_SAMPLE`（最优先，适合 CI）
      2. `data/sample_source.txt` 里的一行路径
      3. `data/` 下任意 `*.m4a` / `*.wav`（取修改时间最新的）

    找不到返回 `None` —— 调用方应当**跳过**而不是报错。
    """
    root = Path(root or ROOT)
    p = os.environ.get("SCREENREADER_SAMPLE")
    if p and Path(p).exists():
        return Path(p).resolve()
    txt = root / "data" / "sample_source.txt"
    if txt.exists():
        try:
            line = txt.read_text(encoding="utf-8").strip().splitlines()[0].strip()
            if line and (root / line).exists():
                return (root / line).resolve()
            if line and Path(line).exists():
                return Path(line).resolve()
        except Exception:
            pass
    data = root / "data"
    if data.is_dir():
        cands = [f for ext in ("*.m4a", "*.wav", "*.mp4", "*.flac")
                 for f in data.glob(ext)]
        cands = [c for c in cands if c.is_file() and c.stat().st_size > 1024 * 1024]
        if cands:
            return max(cands, key=lambda c: c.stat().st_mtime).resolve()
    return None


def find_ffmpeg() -> str:
    """找一个能用的 ffmpeg：优先整合包自带，其次系统 PATH，最后 vendor。"""
    cand = GSV_ROOT / "ffmpeg.exe"
    if cand.exists():
        return str(cand)
    p = shutil.which("ffmpeg")
    if p:
        return p
    v = ROOT / "vendor" / "ffmpeg" / "bin" / "ffmpeg.exe"
    if v.exists():
        return str(v)
    return "ffmpeg"


def find_ffprobe() -> str:
    cand = GSV_ROOT / "ffprobe.exe"
    if cand.exists():
        return str(cand)
    p = shutil.which("ffprobe")
    if p:
        return p
    v = ROOT / "vendor" / "ffmpeg" / "bin" / "ffprobe.exe"
    if v.exists():
        return str(v)
    return "ffprobe"
