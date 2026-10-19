"""用整合包内嵌 Python 跑 screenreader 命令。

为什么需要这个：内嵌 Python 带 `python39._pth`，处于**隔离模式**——
`PYTHONPATH` 会被忽略，所以 `python -m screenreader` 报
`No module named screenreader`。

这里手动把仓库的 `src` 插进 sys.path，再转发给 CLI。

用法（注意用内嵌 Python）：
    <GSV>\\runtime\\python.exe -s tools\\run_with_embedded.py voice train voice02
    <GSV>\\runtime\\python.exe -s tools\\run_with_embedded.py voice list
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def ensure_writable_temp() -> Path:
    """保证有一个**可写的临时目录 + 各类缓存目录**。

    踩过的坑（两个，都会表现为"CPU 满载但永远不返回"）：

    1. **临时目录**：这些工具常常 `os.chdir(<GSV_ROOT>)`（UVR5 之类要求），
       而 `tempfile` 在系统 TEMP 不可写时会**退回到当前目录** ——
       整合包目录若也不可写，就得到：
           FileNotFoundError: No usable temporary directory found in [...]
       表现是 import gradio 一启动就崩，报错还很难懂。

    2. **numba 编译缓存**：`librosa` 依赖 numba，numba 默认把 JIT 结果写进
       **源码旁边的 __pycache__**。整合包目录不可写时它会**无限重试**：
           CPU 满载 + 零 I/O + 永不返回
       实测 `import librosa` 卡 25 秒超时；`2-get-hubert-wav32k.py`（1B）
       卡 20 分钟烧掉 1124 秒 CPU。设了 NUMBA_CACHE_DIR 之后 1.4 秒通过。

    这些子进程是**孙进程**（run_with_embedded → 训练脚本 → 更进一步），
    环境变量会继承，所以在这里设一次就够。
    """
    cands = [
        ROOT / "data" / "_tmp",
        Path(os.environ.get("TEMP") or "."),
        Path(tempfile.gettempdir()),
    ]
    for d in cands:
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except Exception:
            continue

        os.environ["TEMP"] = os.environ["TMP"] = str(d)
        tempfile.tempdir = str(d)
        # 各类缓存也一并指过来，避免子进程再踩"不可写"的坑
        cache = ROOT / "data" / "_tmp" / "cache"
        for sub, var in (("numba", "NUMBA_CACHE_DIR"),
                         ("mpl", "MPLCONFIGDIR"),
                         ("librosa", "LIBROSA_CACHE_DIR")):
            p = cache / sub
            try:
                p.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault(var, str(p))
            except Exception:
                pass
        # 让 modelscope / huggingface 别去联网查更新（网络不通会永久挂起）
        os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return d
    return ROOT / "data"


ensure_writable_temp()

from screenreader.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
