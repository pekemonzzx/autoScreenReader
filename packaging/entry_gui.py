"""PyInstaller 的入口脚本。

单独放一个文件是因为 PyInstaller 需要一个可执行脚本，不能直接指向
`python -m screenreader gui` 这种模块调用。

两个要点：
  1. 打包后 `sys.frozen` 为真，`screenreader.config._app_data_base()` 会返回
     exe 所在目录，于是 data/ 和 voices/ 都落在安装目录旁边。
  2. **崩溃必须留痕**。windowed 模式（console=False）下未捕获的异常不一定会
     弹出任何东西，双击没反应会让人完全无从下手。这里把 traceback 写到
     `data/crash.log`，并在可能的情况下弹个消息框。
"""

import os
import sys
import traceback
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)

if not FROZEN:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# 让 Qt 别去找系统主题，避免在部分机器上启动很慢
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")


def _base_dir() -> Path:
    return Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parents[1]


def _write_crash(exc_text: str) -> Path:
    d = _base_dir() / "data"
    try:
        d.mkdir(parents=True, exist_ok=True)
        f = d / "crash.log"
        with open(f, "a", encoding="utf-8") as fh:
            fh.write("\n" + "=" * 70 + "\n")
            fh.write(exc_text)
        return f
    except Exception:
        return d / "crash.log"


def _die(exc_text: str) -> int:
    log = _write_crash(exc_text)
    sys.stderr.write(exc_text)
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            None,
            f"程序启动失败，详细信息已写入：\n{log}\n\n{exc_text[-1200:]}",
            "自动读屏台 - 启动失败",
            0x10,
        )
    except Exception:
        pass
    return 1


def main() -> int:
    # 命令行模式：目标机上没有 Python，排障只能靠这个 exe。
    #   读屏台.exe --runtime status
    #   读屏台.exe --runtime install --manifest https://.../runtime_manifest.json
    #   读屏台.exe --voice list
    argv = sys.argv[1:]
    if argv and argv[0] in ("--runtime", "--voice", "--cli"):
        return _run_cli(argv)

    try:
        import logging

        from screenreader.ui.app import main as gui_main

        # 日志也落到文件，方便排障
        d = _base_dir() / "data"
        d.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.FileHandler(d / "screenreader.log", encoding="utf-8"),
                logging.StreamHandler(sys.stderr),
            ],
        )
        logging.getLogger(__name__).info(
            "启动：frozen=%s  程序目录=%s", FROZEN, _base_dir()
        )
        return gui_main()
    except SystemExit:
        raise
    except BaseException:
        return _die(traceback.format_exc())


def _run_cli(argv) -> int:
    """把 exe 当命令行用：`读屏台.exe --runtime ...` / `--voice ...`。

    打包后没有 Python，用户不可能跑 `python -m screenreader ...`。
    这里提供一个等价入口（同时把输出写进日志，windowed 模式下看不到控制台）。
    """
    try:
        import logging

        from screenreader.cli import main as cli_main

        d = _base_dir() / "data"
        d.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            handlers=[logging.FileHandler(d / "screenreader.log", encoding="utf-8")],
        )
        # 去掉 --cli 这个占位
        args = [a for a in argv if a != "--cli"]
        if args and args[0] == "--runtime":
            args = ["runtime"] + args[1:]
        elif args and args[0] == "--voice":
            args = ["voice"] + args[1:]

        # windowed 模式没有控制台，把 stdout 同时接到一个日志文件
        class _Tee:
            def __init__(self, stream, fh):
                self.stream = stream
                self.fh = fh

            def write(self, s):
                try:
                    if self.stream:
                        self.stream.write(s)
                except Exception:
                    pass
                try:
                    self.fh.write(s)
                    self.fh.flush()
                except Exception:
                    pass
                return len(s)

            def flush(self):
                for f in (self.stream, self.fh):
                    try:
                        if f:
                            f.flush()
                    except Exception:
                        pass

        fh = open(d / "cli_output.txt", "w", encoding="utf-8")
        sys.stdout = _Tee(sys.stdout, fh)
        sys.stderr = _Tee(sys.stderr, fh)
        rc = cli_main(args)
        print(f"\n（以上输出也写入了 {d / 'cli_output.txt'}）")
        return rc
    except SystemExit as e:
        return int(e.code or 0)
    except BaseException:
        return _die(traceback.format_exc())


if __name__ == "__main__":
    raise SystemExit(main())
