"""界面入口。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def main(argv=None) -> int:
    from PySide6.QtWidgets import QApplication

    from ..config import AppConfig, default_config_path

    argv = list(sys.argv if argv is None else argv)

    # 极简参数解析（GUI 程序不需要 argparse 全套）
    cfg_path = None
    if "-c" in argv:
        i = argv.index("-c")
        if i + 1 < len(argv):
            cfg_path = Path(argv[i + 1])
    cfg = AppConfig.load(cfg_path or default_config_path())
    # 配置里指定了声库就用档案里的推理参数（参考音频/prompt_text/权重）。
    # 没指定会自动选第一个可用的 —— 否则会出现"界面显示着某个音色，
    # 但 ref_audio 是空的，每句合成都失败"这种迷惑现象。
    _before = (cfg.tts.voice or "").strip()
    _got = cfg.apply_voice()
    if _got and _got != _before:
        try:
            cfg.save(cfg_path or default_config_path())
            log.info("已把自动选中的声库 %r 写回配置", _got)
        except Exception as e:
            log.warning("写回声库选择失败：%s", e)
    cfg.apply_log_level()

    app = QApplication(argv)
    app.setApplicationName("自动读屏台")
    app.setStyleSheet(
        """
        QMainWindow, QWidget { background: #0d1117; color: #c9d1d9; }
        QToolBar { background: #161b22; border: none; spacing: 4px; padding: 4px; }
        QToolButton { color: #c9d1d9; padding: 5px 10px; border-radius: 6px; }
        QToolButton:hover { background: #21262d; }
        QToolButton:checked { background: #1f6feb; color: #ffffff; }
        QListWidget { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; }
        QListWidget::item { padding: 4px; }
        QListWidget::item:selected { background: #1f6feb; }
        QScrollArea { border: 1px solid #30363d; border-radius: 6px; background: #0d1117; }
        QPushButton { background: #21262d; border: 1px solid #30363d; border-radius: 6px;
                      padding: 3px 6px; color: #c9d1d9; }
        QPushButton:hover { background: #30363d; }
        QPushButton:disabled { color: #484f58; background: #161b22; }
        QFrame#cardRow { border: 1px solid #21262d; border-radius: 6px; }
        QStatusBar { background: #161b22; color: #8b949e; }
        QLabel { color: #c9d1d9; }
        """
    )

    from .main_window import MainWindow

    win = MainWindow(cfg)
    win.show()
    log.info("界面已启动。配置文件：%s", cfg_path or default_config_path())

    # 首次启动检查：TTS 运行时（~9GB）是按需下载的，没装就提示怎么装
    _check_runtime(win, cfg)

    return app.exec()


def _check_runtime(win, cfg) -> None:
    """首次启动检查：TTS 运行时是按需下载的，没装就引导安装。

    目标机上**没有 Python**，所以不能只给命令行——必须在这里能装完。
    """
    from PySide6.QtCore import QTimer

    from .. import runtime as R

    def check():
        st = R.status()
        if st["installed"]:
            return
        from .runtime_dialog import RuntimeInstallDialog

        dlg = RuntimeInstallDialog(win, first_run=True)
        dlg.exec()

    # 等窗口画出来再弹，避免对话框抢焦点导致主窗口不显示
    QTimer.singleShot(800, check)


if __name__ == "__main__":
    raise SystemExit(main())
