"""子进程调用工具：**保证不闪黑窗**。

GUI 程序（PyInstaller `console=False`）里调用 `subprocess` 时，如果目标是个
控制台程序（`nvidia-smi`、`whoami`、`cmd`…），Windows 会**弹出一个控制台窗口**。
哪怕只跑 100 毫秒，用户也会看到屏幕上一闪。

实测症状：读屏台点「开始识别」后**每 10 秒闪一次黑窗** —— 因为显存看门狗
每 10 秒调一次 `nvidia-smi` 查显存。

处理：一律走这里的 `run_hidden()` / `popen_hidden()`，
它们会带上 `CREATE_NO_WINDOW`（Windows）并配合 `STARTUPINFO` 隐藏窗口。
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any, List, Optional, Union

log = logging.getLogger(__name__)

CREATE_NO_WINDOW = 0x08000000


def _hidden_kwargs() -> dict:
    """返回让子进程不弹窗的 kwargs。非 Windows 上返回空 dict。"""
    if sys.platform != "win32":
        return {}
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0          # SW_HIDE
    return {"startupinfo": si, "creationflags": CREATE_NO_WINDOW}


def run_hidden(cmd: Union[str, List[str]], timeout: float = 15, **kw) -> subprocess.CompletedProcess:
    """跑一个子进程并等它结束，**不弹控制台窗口**。

    默认 `capture_output=True, text=True`，方便直接读 stdout。
    """
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    kw.setdefault("errors", "replace")
    kw.setdefault("timeout", timeout)
    kw.update(_hidden_kwargs())
    return subprocess.run(cmd, **kw)


def popen_hidden(cmd: Union[str, List[str]], **kw) -> subprocess.Popen:
    """启动一个长期运行的子进程，**不弹控制台窗口**。

    用于启动 TTS 推理服务这类后台进程。
    """
    kw.update(_hidden_kwargs())
    return subprocess.Popen(cmd, **kw)


def relaunch(cmd: Union[str, List[str]], cwd=None, wait_s: float = 2.0) -> dict:
    """重启自己：拉起新进程，并**确认它真的起来了**。

    用户反馈过「点了现在重启，当前程序关了但新的没起来」。
    原因通常是新进程被当成当前进程的子进程，父进程退出时被一起带走，
    或者只是拉起来了但我们没等就退了，看起来像"没启动"。

    所以这里：
      * `DETACHED_PROCESS` + `CREATE_NEW_PROCESS_GROUP` —— 新进程独立于我们
      * `CREATE_BREAKAWAY_FROM_JOB` —— 万一我们被放进某个 job object，
        新进程也要能脱离（失败就退回不带这个标志）
      * 起来之后**等一下确认没立刻退出**，把结果返回给调用方
    """
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    CREATE_BREAKAWAY_FROM_JOB = 0x01000000
    CREATE_NO_WINDOW = 0x08000000

    base = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    out = {"ok": False, "pid": None, "detail": ""}

    for flags in (base | CREATE_BREAKAWAY_FROM_JOB, base, base | CREATE_NO_WINDOW):
        try:
            kw = {"cwd": cwd} if cwd else {}
            if kw:
                kw["close_fds"] = True
            p = subprocess.Popen(cmd, creationflags=flags,
                                 stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, **kw)
        except Exception as e:
            out["detail"] = f"Popen 失败（flags={flags:#x}）：{e}"
            continue

        # 等一下看它是不是立刻退了（立刻退通常是参数/权限问题）
        import time as _t

        t0 = _t.time()
        while _t.time() - t0 < wait_s:
            rc = p.poll()
            if rc is not None:
                out["detail"] = f"新进程立刻退出（返回码 {rc}），flags={flags:#x}"
                break
            _t.sleep(0.1)
        else:
            out.update({"ok": True, "pid": p.pid,
                        "detail": f"已启动 PID={p.pid}（flags={flags:#x}）"})
            return out

    return out


def describe() -> str:
    ok = sys.platform == "win32"
    return f"hidden-subprocess={'on' if ok else 'n/a'} ({sys.platform})"
