"""主界面：独立窗口 + 悬浮字幕条 + 区域角标，可自由组合。

线程边界：BoardEngine 的回调来自工作线程，**绝不能直接碰控件**。
这里用 EngineBridge 把回调转成 Qt 信号，Qt 会自动走队列连接投递到主线程。

三个"视图"共享同一份卡片状态：
    card_list        独立窗口里的主列表
    overlay.cards    悬浮字幕条
    region_overlay   屏幕上画的区域角标（点击穿透）
所以卡片事件要同步刷到前两个视图上——**漏掉任何一个都会出现"悬浮窗没内容"这类问题**。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

from PySide6.QtCore import QEvent, QObject, QPoint, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QCursor, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizeGrip,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..board import Card
from ..config import AppConfig, default_config_path
from ..hotkey import HotkeyListener, parse_binding
from ..segment import merge_lines_to_sentences
from ..voice import VoiceStore
from ..worker import BoardEngine, EngineConfig
from .cards import CardList
from .region_overlay import RegionOverlay
from .region_picker import RegionPicker
from .hotkey_edit import HotkeyCapture
from .snap_picker import SnapPicker, qimage_to_rgb_array

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
class EngineBridge(QObject):
    """把引擎回调（工作线程）转成 Qt 信号（主线程）。"""

    event = Signal(dict)


class ServiceWorker(QThread):
    """后台检查 / 拉起 TTS 推理服务。

    启动一次要几十秒加载权重，**不能卡在界面线程上**。
    """

    done = Signal(dict)

    def __init__(self, cfg, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg

    def run(self) -> None:
        try:
            from .. import runtime as R

            res = R.ensure_tts_service(self.cfg, say=lambda m: log.info("%s", m))
        except Exception as e:
            log.exception("拉起 TTS 服务失败")
            res = {"ok": False, "message": str(e)}
        self.done.emit(res)


# --------------------------------------------------------------------------- #
def _safe_int(v, default: int, lo: int, hi: int) -> int:
    """把配置里的值安全地转成整数并夹到 [lo, hi]。

    老配置文件里可能没有这个键（.get 拿到 None），或者被手改成奇怪的值，
    直接 int() 会抛异常把界面搞崩。
    """
    try:
        return max(lo, min(hi, int(v)))
    except Exception:
        return default


def _is_checked(state) -> bool:
    """QCheckBox.stateChanged 传进来的是不是"已勾选"。

    **不能用 `state == Qt.Checked`。** PySide6 里 `Qt.CheckState` 是普通 Enum
    而不是 IntEnum，`2 == Qt.CheckState.Checked` 恒为 False —— 实测勾选框的
    状态变了、信号也发了，但处理函数里判定永远是"未勾选"，于是开关无效。
    统一按数值比较：0=Unchecked / 1=Partially / 2=Checked。
    """
    try:
        return int(state) == int(Qt.CheckState.Checked)
    except Exception:
        return bool(state)


class _ShutdownDialog(QDialog):
    """关闭时显示"正在清理"的窗口。

    用户要求：**不论是否在识别**，关程序都要显示清理 TTS 的窗口，
    并且能看出后台进程有没有真的被清掉。

    做成无按钮 + 自动关闭，用户不用操作；最后一行会写明结果。
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("正在清理")
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        v = QVBoxLayout(self)
        self.lbl_title = QLabel("正在停止后台服务…")
        f = self.lbl_title.font()
        f.setBold(True)
        self.lbl_title.setFont(f)
        v.addWidget(self.lbl_title)
        self.txt = QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMinimumHeight(120)
        self.txt.setStyleSheet("font-family:Consolas,monospace; font-size:11px;")
        v.addWidget(self.txt)

    def add(self, msg: str) -> None:
        self.txt.appendPlainText(msg)
        QApplication.processEvents()

    def finish(self, ok: bool = True) -> None:
        self.lbl_title.setText("清理完成" if ok else "清理完成（有残留，见下）")
        QApplication.processEvents()


class OverlayWindow(QWidget):
    """悬浮字幕条：无边框、置顶、半透明。

    交互设计（按用户反馈改的）：
      * **拖动条只在鼠标移进来时才显示**，平时完全不挡画面
      * 拖动条右上角有个 ✕，点了就隐藏悬浮窗（工具栏按钮同步取消勾选）
      * 右下角有尺寸手柄，可以**拖动调整悬浮窗大小**
      * 拖动条本身是拖拽移动窗口的把手
    """

    toggle_requested = Signal(str)
    retry_requested = Signal(str)      # 卡片上的「重新生成」
    closed = Signal()          # 通知主窗口把工具栏的勾选去掉
    play_all_requested = Signal()   # 悬浮窗的「全部播放」
    run_toggle_requested = Signal()      # 悬浮窗的「开始/停止识别」
    autoread_toggled = Signal(bool)      # 悬浮窗的「自动朗读」
    opacity_changed = Signal(float)      # 悬浮窗透明度（0.2~1.0）

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setMinimumSize(260, 120)
        self.resize(460, 300)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(4)

        # ---- 拖动条（默认隐藏）----
        self.header = QWidget()
        self.header.setCursor(Qt.SizeAllCursor)
        hb = QHBoxLayout(self.header)
        hb.setContentsMargins(6, 2, 2, 2)
        hb.setSpacing(3)

        # 标题：**可被压缩**（Ignored 策略），窗口窄的时候自动让位给按钮，
        # 否则拖动条会被标题撑得挤掉后面的控件。
        self.lbl_title = QLabel("读屏台")
        self.lbl_title.setStyleSheet("color:#8b949e;font-size:11px;")
        self.lbl_title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_title.setToolTip("拖动这里移动悬浮窗")
        hb.addWidget(self.lbl_title, 1)

        _HEAD_BTN = (
            "QPushButton{background:#21262d;border:1px solid #30363d;border-radius:4px;"
            "color:#c9d1d9;padding:0 6px;font-size:11px;}"
            "QPushButton:hover{background:#30363d;color:#e6edf3;}"
            "QPushButton:checked{background:#238636;border-color:#2ea043;color:#ffffff;}"
            "QPushButton:disabled{color:#6e7681;}"
        )

        # **开始/停止识别**（用户要求放进拖动条）
        self.btn_run = QPushButton("开始识别")
        self.btn_run.setFixedHeight(20)
        self.btn_run.setCursor(Qt.ArrowCursor)
        self.btn_run.setToolTip("开始 / 停止识别")
        self.btn_run.setStyleSheet(_HEAD_BTN)
        self.btn_run.clicked.connect(self.run_toggle_requested)
        hb.addWidget(self.btn_run)

        # **自动朗读**（用户要求放进拖动条）—— 可勾选的开关
        self.btn_autoread = QPushButton("自动")
        self.btn_autoread.setFixedHeight(20)
        self.btn_autoread.setCheckable(True)
        self.btn_autoread.setCursor(Qt.ArrowCursor)
        self.btn_autoread.setToolTip("自动朗读：每张卡片生成就自动播一遍")
        self.btn_autoread.setStyleSheet(_HEAD_BTN)
        self.btn_autoread.toggled.connect(self.autoread_toggled)
        hb.addWidget(self.btn_autoread)

        # 悬浮窗也要有「全部播放」（用户要求）。
        # 第一句一生成就可用，后面的句子在播放过程中自动接进队列。
        self.btn_play_all = QPushButton("▶ 全部")
        self.btn_play_all.setFixedHeight(20)
        self.btn_play_all.setCursor(Qt.ArrowCursor)
        self.btn_play_all.setToolTip("从上到下顺序朗读整段（第一句就绪即可点）")
        self.btn_play_all.setStyleSheet(
            "QPushButton{background:#238636;border:none;border-radius:4px;"
            "color:#ffffff;padding:0 8px;font-size:11px;}"
            "QPushButton:hover{background:#2ea043;}"
            "QPushButton:disabled{background:#30363d;color:#6e7681;}"
        )
        # **常亮**（用户要求）：点了就从列表第一项开始读，
        # 第一句还没合成好也没关系，播放循环会原地等它
        self.btn_play_all.setEnabled(True)
        self.btn_play_all.clicked.connect(self.play_all_requested)
        hb.addWidget(self.btn_play_all)

        # **透明度**（用户要求）。用滑条而不是菜单，好一眼看出当前值。
        # 映射：滑条 20~100 -> 窗口不透明度 0.20~1.00。
        # 只调窗口透明度（`setWindowOpacity`），不动背景色 ——
        # 这样文字和按钮一起半透明，不会被"卡片还实心"衬得很怪。
        self.sld_opacity = QSlider(Qt.Horizontal)
        self.sld_opacity.setRange(20, 100)
        self.sld_opacity.setValue(100)
        self.sld_opacity.setFixedWidth(64)
        self.sld_opacity.setFixedHeight(18)
        self.sld_opacity.setCursor(Qt.ArrowCursor)
        self.sld_opacity.setToolTip("悬浮窗不透明度（也可以在同名工具栏设置里调）")
        self.sld_opacity.setStyleSheet(
            "QSlider::groove:horizontal{height:3px;background:#30363d;border-radius:2px;}"
            "QSlider::handle:horizontal{width:9px;height:9px;margin:-3px 0;"
            "background:#8b949e;border-radius:5px;}"
            "QSlider::sub-page:horizontal{background:#238636;border-radius:2px;}"
        )
        self.sld_opacity.valueChanged.connect(self._on_opacity_slider)
        hb.addWidget(self.sld_opacity)

        self.btn_close = QPushButton("✕")
        self.btn_close.setFixedSize(20, 20)
        self.btn_close.setToolTip("隐藏悬浮窗")
        self.btn_close.setCursor(Qt.ArrowCursor)
        self.btn_close.setStyleSheet(
            "QPushButton{background:#30363d;border:none;border-radius:4px;color:#c9d1d9;}"
            "QPushButton:hover{background:#f85149;color:#ffffff;}"
        )
        self.btn_close.clicked.connect(self.hide_overlay)
        hb.addWidget(self.btn_close)
        root.addWidget(self.header)
        # 拖动条**常显、不隐藏**（用户要求）。
        #
        # 之前试过"移出布局再放回"，会让下面列表跳一格；
        # 又试过"留在布局里但隐藏内部控件"，结果 hover 事件过滤器
        # 装在那些隐藏控件上收不到鼠标移动，拖动条和「全部」按钮就再也不出来了。
        # 直接常显最省事，也不再需要那套"鼠标移进移出"的定时器。
        self.header.setFixedHeight(26)
        self._header_on = True
        self._set_header_visible(True)

        # ---- 卡片列表 ----
        self.cards = CardList(compact=True)
        self.cards.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        self.cards.toggle_requested.connect(self.toggle_requested)
        self.cards.retry_requested.connect(self.retry_requested)
        root.addWidget(self.cards, 1)

        # ---- 右下角尺寸手柄 ----
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addStretch(1)
        self.grip = QSizeGrip(self)
        self.grip.setFixedSize(16, 16)
        self.grip.setToolTip("拖动调整大小")
        grip_row.addWidget(self.grip, 0, Qt.AlignRight | Qt.AlignBottom)
        root.addLayout(grip_row)

        self.setStyleSheet(
            "OverlayWindow{background:rgba(13,17,23,220); border:1px solid #30363d;"
            " border-radius:10px;}"
        )
        self._drag: Optional[QPoint] = None

        # 鼠标一动就显示拖动条，停 1.8 秒后自动收起
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(1800)
        self._hide_timer.timeout.connect(self._hide_header)
        self._install_hover_watch()

    # ------------------------------------------------------------------ #
    def _install_hover_watch(self) -> None:
        """给所有子控件装事件过滤器，任何鼠标移动都能唤醒拖动条。

        为什么要递归装：鼠标移到卡片、滚动条上时，事件不会冒泡到顶层窗口，
        只靠顶层 enterEvent 会漏掉大部分情况。
        """
        for w in [self, self.header, self.cards, self.cards.viewport(), self.grip]:
            w.setMouseTracking(True)
            w.installEventFilter(self)
        QTimer.singleShot(0, self._watch_children)

    def _watch_children(self) -> None:
        for w in self.findChildren(QWidget):
            w.setMouseTracking(True)
            w.installEventFilter(self)

    def eventFilter(self, obj, ev) -> bool:
        if ev.type() == QEvent.MouseMove and self.isVisible():
            self._poke_header()
        elif ev.type() == QEvent.Enter and self.isVisible():
            self._poke_header()
        return super().eventFilter(obj, ev)

    def _poke_header(self) -> None:
        """拖动条常显，这里只保证它没被意外收起来。"""
        if not self._header_on:
            self._set_header_visible(True)
            self._header_on = True

    # ---------------------------------------------------- 拖动条上的控件 --
    def _on_opacity_slider(self, v: int) -> None:
        """透明度滑条 -> 窗口不透明度。"""
        val = max(0.20, min(1.0, v / 100.0))
        self.setWindowOpacity(val)
        self.opacity_changed.emit(val)

    def set_opacity_pct(self, pct: int) -> None:
        """外部设置透明度（百分比 20~100），不重复发信号。"""
        v = max(20, min(100, int(pct)))
        self.sld_opacity.blockSignals(True)
        self.sld_opacity.setValue(v)
        self.sld_opacity.blockSignals(False)
        self.setWindowOpacity(v / 100.0)

    def set_running(self, running: bool) -> None:
        """同步「开始/停止识别」按钮的文案和颜色。"""
        self.btn_run.setText("停止识别" if running else "开始识别")
        self.btn_run.setStyleSheet(
            "QPushButton{background:#da3633;border:1px solid #f85149;border-radius:4px;"
            "color:#ffffff;padding:0 6px;font-size:11px;}"
            "QPushButton:hover{background:#f85149;}"
            if running else
            "QPushButton{background:#238636;border:1px solid #2ea043;border-radius:4px;"
            "color:#ffffff;padding:0 6px;font-size:11px;}"
            "QPushButton:hover{background:#2ea043;}"
        )

    def set_autoread(self, on: bool) -> None:
        """同步「自动朗读」按钮的勾选状态（不触发信号，避免回环）。"""
        self.btn_autoread.blockSignals(True)
        self.btn_autoread.setChecked(bool(on))
        self.btn_autoread.blockSignals(False)

    def _hide_header(self) -> None:
        """拖动条现在**常显**，不做自动收起（用户要求）。"""
        return

    def _set_header_visible(self, on: bool) -> None:
        """显示/隐藏拖动条（当前只用于常显）。

        保留这个方法是为了以后想改回自动收起时有地方下手；
        注意如果真要隐藏，**不能**只隐藏内部控件 ——
        hover 事件过滤器装在它们身上，一隐藏就再也收不到鼠标移动了。
        """
        for w in (self.lbl_title, self.btn_play_all, self.btn_close):
            if w is not None:
                w.setVisible(on)
        self.header.setStyleSheet("" if on else "background:transparent;")
        self.header.setCursor(Qt.SizeAllCursor if on else Qt.ArrowCursor)
        self._header_on = bool(on)

    # ------------------------------------------------------------------ #
    def hide_overlay(self) -> None:
        self.hide()
        self.closed.emit()

    # ------------------------------------------------------------------ #
    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            # 拖动条区域内按下才移动窗口；其它地方留给卡片按钮
            if self.header.isVisible() and ev.position().y() <= self.header.height() + 10:
                self._drag = ev.globalPosition().toPoint() - self.frameGeometry().topLeft()
                self._hide_timer.stop()

    def mouseMoveEvent(self, ev) -> None:
        if self._drag is not None:
            self.move(ev.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, ev) -> None:
        self._drag = None
        self._hide_timer.start()

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        # 新建的卡片也要能唤醒拖动条
        self._watch_children()

    def enterEvent(self, ev) -> None:
        self._poke_header()
        super().enterEvent(ev)


# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    # OCR 在后台线程里跑，结果通过这个信号回到主线程（跨线程碰控件会崩）
    snap_ocr_done = Signal(object)

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("自动读屏台")
        self.resize(1120, 700)

        self.engine = BoardEngine(
            cfg, EngineConfig(
                poll_interval_s=cfg.pipeline.min_interval_s,
                skip_unchanged_frame=bool(getattr(cfg.ocr, "skip_unchanged_frame", True)),
                auto_read_all=bool(getattr(cfg.pipeline, "auto_read_all", False)),
            )
        )
        self.bridge = EngineBridge()
        self.bridge.event.connect(self._on_event)
        self.engine.on_event = self.bridge.event.emit

        self.overlay = OverlayWindow()
        self.overlay.toggle_requested.connect(self._on_toggle)
        self.overlay.retry_requested.connect(self._on_retry_card)
        # 拖动条上的三个新控件（用户要求）
        self.overlay.run_toggle_requested.connect(self._toggle_engine)
        self.overlay.autoread_toggled.connect(self._set_autoread_from_overlay)
        self.overlay.opacity_changed.connect(self._on_overlay_opacity)
        # 用之前存下的透明度
        self.overlay.set_opacity_pct(getattr(self.cfg, "overlay_opacity", 100) or 100)
        # 悬浮窗的「自动」按钮初始状态要跟配置一致
        self.overlay.set_autoread(bool(getattr(self.cfg.pipeline, "auto_read_all", False)))
        self.overlay.closed.connect(self._on_overlay_closed)
        self.overlay.play_all_requested.connect(self._on_play_all_paragraphs)
        self.overlay.hide()

        self.region_overlay = RegionOverlay()

        self._build_ui()

        # ---- 截图朗读的一次性状态 ----
        self._snap_timer = QTimer(self)
        self._snap_timer.setInterval(150)
        self._snap_timer.timeout.connect(self._snap_tick)
        self._snap_rid: Optional[str] = None
        self._snap_enabled: Dict[str, bool] = {}
        self._snap_started = False
        self._snap_pending = False
        self._snap_t0 = 0.0
        self._snap_play_t0 = 0.0
        self._snap_hotkey_failed = False
        # OCR 在后台线程跑，结果经这个信号回到主线程
        self.snap_ocr_done.connect(self._on_snap_ocr_done)
        self._snap_picker = None
        self._snap_screen_geo = None
        self._snap_monitor = 0
        self._snap_vis: Dict[str, bool] = {}
        # 普通框选层（同一时刻只允许一个框选层，否则会叠两层）
        self._picker = None
        # **框选层保险丝**：全屏置顶窗口一旦出事会把用户困住
        # （实测严重到只能重启电脑），所以必须有兜底：
        #   1) 全局 Esc（走低级钩子，不依赖焦点）
        #   2) 硬超时：超过 3 分钟自动取消
        self._snap_guard = QTimer(self)
        self._snap_guard.setSingleShot(True)
        self._snap_guard.timeout.connect(self._snap_hard_cancel)
        # 识别健康看门狗：检测线程万一卡死/退出，能自动救回来
        self._detect_seen_polls = -1
        self._detect_stall_ticks = 0

        # ---- 全局快捷键（截图朗读）----
        self._hotkeys = HotkeyListener()
        if getattr(self.cfg.snap, "hotkey", ""):
            self._hotkeys.set_bindings({"snap": self.cfg.snap.hotkey})
        if not self._hotkeys.start():
            self._snap_hotkey_failed = True
            log.info("全局快捷键未启用（不影响按钮和菜单）")
        self._hotkey_timer = QTimer(self)
        self._hotkey_timer.setInterval(120)
        self._hotkey_timer.timeout.connect(self._poll_hotkeys)
        self._hotkey_timer.start()

        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(500)
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._stats_timer.start()

    def _picker_busy(self) -> bool:
        """现在是否有框选层开着（截图朗读的 / 普通框选的）。

        框选期间**必须屏蔽全局快捷键**：否则再按一下会叠出第二个全屏框选层，
        两个都置顶，第二个盖住第一个，看起来就是"卡住了/点不动"。
        """
        if getattr(self, "_snap_picker", None) is not None:
            return True
        if getattr(self, "_snap_pending", False):
            return True
        try:
            if self._snap_guard.isActive():
                return True
        except Exception:
            pass
        p = getattr(self, "_picker", None)
        if p is not None:
            try:
                if p.isVisible():
                    return True
            except Exception:
                # 已经 deleteLater 的 C++ 对象访问会抛，当作没有
                return False
        return False

    def _poll_hotkeys(self) -> None:
        """把全局快捷键命中转成界面动作。

        钩子回调在**它自己的线程**里，所以只能往队列里塞；
        这里在 Qt 主线程轮询取出来再动界面，跨线程才安全。
        """
        try:
            while True:
                name = self._hotkeys.poll()
                if name is None:
                    break
                if name == "snap":
                    # **框选期间忽略快捷键**（用户要求）。
                    # 否则会在已有全屏框选层上再叠一层，两个都置顶，
                    # 看起来就是"界面不动了"。
                    if self._picker_busy():
                        log.info("正在框选，忽略截图朗读快捷键")
                        continue
                    log.info("全局快捷键触发截图朗读")
                    self.snap_read()
                elif name == "__escape__":
                    # **框选层的逃生出口**：全局 Esc，不依赖焦点。
                    if getattr(self, "_snap_picker", None) is not None:
                        log.info("全局 Esc：关闭截图朗读框选层")
                        self._snap_hard_cancel()
                    elif getattr(self, "_picker", None) is not None:
                        try:
                            if self._picker.isVisible():
                                self._picker.hide()
                                self._picker.cancelled.emit()
                        except Exception:
                            pass
        except Exception as e:
            log.warning("处理全局快捷键失败：%s", e)

    # ------------------------------------------------------------------ #
    @property
    def views(self) -> List[CardList]:
        """所有需要同步卡片状态的视图。加视图时记得挂进来。"""
        return [self.card_list, self.overlay.cards]

    @property
    def list_views(self) -> List[CardList]:
        """**按当前选中区域过滤**的视图（只有主列表）。"""
        return [self.card_list]

    @property
    def all_views(self) -> List[CardList]:
        """**显示所有区域**的视图（悬浮窗）。

        两个区域（比如一个单行、一个整段）同时开着时，悬浮窗必须把所有卡片
        都显示出来 —— 以前 `_refresh_cards()` 会把"非当前选中区域"的卡片
        从这个视图里删掉，于是切一下区域，悬浮窗里另一个区域的卡片就凭空
        消失，看起来像"卡住/不刷新"。
        """
        return [self.overlay.cards]

    def _build_ui(self) -> None:
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        # 图标 + 文字：只有文字时按钮宽度随机、容易被挤到看不见
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        def _act(text: str, tip: str, slot=None, checkable: bool = False) -> QAction:
            a = QAction(text, self)
            a.setToolTip(tip)
            a.setStatusTip(tip)
            if checkable:
                a.setCheckable(True)
            if slot is not None:
                a.triggered.connect(slot)
            return a

        # ---- 组 1：框选类（"我要读哪里"）----
        self.act_pick = _act(
            "▭ 框选区域",
            "在屏幕上框出要持续读取的区域（可框多个，Esc 取消）。\n"
            "框好的区域会一直在后台识别。",
            self._pick_regions)
        tb.addAction(self.act_pick)

        # **截图朗读**：点一下（或按快捷键）-> 框一段 -> 立刻读一遍 -> 自动清场
        self.act_snap = _act(
            "◱ 截图朗读",
            "一次性读取：框一段，立刻识别并整段朗读一遍，读完自动清理。\n"
            "画面会先被冻结，框到的就是读到的。\n"
            "可以在「≡ 更多 → 快捷键」里绑键盘组合键或鼠标侧键。",
            self.snap_read)
        tb.addAction(self.act_snap)

        tb.addSeparator()

        # ---- 组 2：运行控制（最常用的两个）----
        self.act_toggle = _act("▶ 开始识别", "开始 / 停止后台识别（停止时也会停掉朗读）",
                               self._toggle_engine)
        tb.addAction(self.act_toggle)

        # **「删除区域」放到一级页面**（用户要求：这个还挺常用）。
        # 原来的「全部停止」并进「开始识别/停止识别」——两个按钮做同一类事，
        # 用户会分不清该点哪个。需要"只停朗读、不关识别"的场景留在「更多」里。
        self.act_del = _act("✕ 删除区域",
                            "删除左侧列表里选中的区域（也可以选中后按 Delete）",
                            self._remove_selected_region)
        tb.addAction(self.act_del)

        tb.addSeparator()

        # ---- 组 3：显示类（悬浮窗放前面，保证默认宽度下看得见）----
        self.act_overlay = _act(
            "▤ 悬浮窗",
            "显示 / 隐藏悬浮字幕条（可拖动、可调透明度）",
            self._toggle_overlay, checkable=True)
        tb.addAction(self.act_overlay)

        self.act_marks = _act(
            "◎ 区域角标",
            "在屏幕上显示区域边框和编号，方便确认框到哪儿了",
            self._toggle_marks, checkable=True)
        self.act_marks.setChecked(True)
        tb.addAction(self.act_marks)

        tb.addSeparator()

        # ---- 组 4：声库 ----
        tb.addWidget(QLabel(" 声库 "))
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(130)
        self.voice_combo.setToolTip("切换音色（声库）。装新声库请用「≡ 更多 → 语音运行时」。")
        self.voice_combo.currentIndexChanged.connect(self._on_voice_changed)
        tb.addWidget(self.voice_combo)

        # ---- 组 5：低频功能收进「更多」菜单 ----
        # 为什么收起来：默认窗口宽度下工具栏放不下，末尾的按钮会被挤掉，
        # 用户根本找不到（用户反馈"默认宽度看不到悬浮窗按钮"）。
        # 现在高频的 6 个按钮固定可见，其余进菜单。
        tb.addSeparator()
        self.btn_more = QToolButton()
        self.btn_more.setText("≡ 更多")
        self.btn_more.setToolTip("删除区域 / 清空卡片 / 快捷键 / 运行时 / 高级设置")
        self.btn_more.setPopupMode(QToolButton.InstantPopup)
        more = QMenu(self.btn_more)

        self.act_clear = more.addAction("清空卡片")
        self.act_clear.setToolTip("清空卡片列表（不影响区域设置）")
        self.act_clear.triggered.connect(self._clear_cards)

        # 只停朗读、保持识别在跑。一级页面的「停止识别」会把识别也停掉，
        # 所以这个留在二级菜单给需要的场景用（用户反馈：那两个按钮重复了）。
        self.act_stop_all = more.addAction("停止朗读（不停识别）")
        self.act_stop_all.setToolTip("停掉当前朗读并清空播放队列，但继续识别")
        self.act_stop_all.triggered.connect(self._stop_all)

        more.addSeparator()
        self.act_hotkey = more.addAction("截图朗读快捷键…")
        self.act_hotkey.setToolTip("给「截图朗读」设全局快捷键（键盘组合键 / 鼠标侧键）")
        self.act_hotkey.triggered.connect(self._open_hotkey_dialog)

        self.act_settings = more.addAction("高级设置…")
        self.act_settings.setToolTip("OCR 线程数、识别上限、缓存等**需要重启识别才生效**的配置")
        self.act_settings.triggered.connect(self._open_settings_dialog)

        more.addSeparator()
        self.act_runtime = more.addAction("语音运行时…")
        self.act_runtime.setToolTip("安装 / 查看 TTS 推理运行时（约 9GB，按需下载）")
        self.act_runtime.triggered.connect(self._open_runtime_dialog)

        self.btn_more.setMenu(more)
        tb.addWidget(self.btn_more)

        # ---- 左：区域列表；右：卡片列表 ----
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 4, 8)
        lv.addWidget(QLabel("识别区域（勾选=启用）"))
        self.region_list = QListWidget()
        self.region_list.itemChanged.connect(self._on_region_item_changed)
        self.region_list.currentItemChanged.connect(self._on_region_selected)
        # **选中区域后按 Delete 删除**（用户要求，比去点按钮快）
        self.region_list.installEventFilter(self)
        self.region_list.setToolTip("勾选=启用；选中后按 Delete 可删除该区域")
        lv.addWidget(self.region_list, 1)

        # 识别模式：作用于**选中的那个区域**
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("识别模式"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("单行 —— 每行一句，手动点播", "line")
        self.cmb_mode.addItem("整段 —— 按标点分句，自动连续朗读", "paragraph")
        self.cmb_mode.setToolTip(
            "整段模式会把折行的部分合并回句子（行尾没标点=自动折行），\n"
            "再按标点分句，然后从第一句开始自动按顺序连续朗读。"
        )
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.cmb_mode, 1)
        lv.addLayout(mode_row)

        self.lbl_mode_hint = QLabel()
        self.lbl_mode_hint.setWordWrap(True)
        self.lbl_mode_hint.setStyleSheet("color:#8b949e; font-size:11px;")
        lv.addWidget(self.lbl_mode_hint)

        # 整段朗读：**手动点**。滚动时内容一直在变，自动播只会不停念
        # "刚出现的前几个字"；等页面稳定了再由用户点这里，从上到下顺序读。
        self.btn_play_para = QPushButton("▶  朗读这个区域（整段）")
        self.btn_play_para.setToolTip(
            "把选中区域的句子从上到下依次读完。\n"
            "页面还在滚动时不会自动播报——稳定下来再点这里。"
        )
        self.btn_play_para.clicked.connect(self._on_play_paragraph)
        self.btn_play_para.setEnabled(False)
        lv.addWidget(self.btn_play_para)

        # 播放音量。**100 = 原始输出，最大 300**（用户反馈默认偏小，
        # 允许往上放大；超过 100 播放器会走软限幅，不会硬削波）。
        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("音量"))
        self.sld_volume = QSlider(Qt.Horizontal)
        self.sld_volume.setRange(0, 300)
        self.sld_volume.setValue(_safe_int(getattr(self.cfg.audio, "volume", 100), 100, 0, 300))
        self.sld_volume.setToolTip(
            "语音播放音量。100 = 原始音量，可放大到 300\n（超过 100 会自动软限幅，不会破音）"
        )
        self.sld_volume.valueChanged.connect(self._on_volume_changed)
        vol_row.addWidget(self.sld_volume, 1)
        self.lbl_volume = QLabel(f"{self.sld_volume.value()}%")
        self.lbl_volume.setFixedWidth(44)
        self.lbl_volume.setStyleSheet("color:#8b949e;")
        vol_row.addWidget(self.lbl_volume)
        # 双击标签回到 100%（原始音量）
        self.lbl_volume.setToolTip("双击恢复 100%（原始音量）")
        self.lbl_volume.mouseDoubleClickEvent = lambda e: self.sld_volume.setValue(100)
        lv.addLayout(vol_row)

        # 读取间隔 —— **CPU 占用的第一控制项**。
        #
        # 实测 OCR 是唯一的大头（2560x1440 一帧 825ms，截屏只有 5.9ms），
        # 所以间隔调大、或者画面没变时跳过，就能把 CPU 让出来。
        int_row = QHBoxLayout()
        int_row.addWidget(QLabel("读取间隔"))
        self.cmb_interval = QComboBox()
        for label, sec in (
            ("0.3 秒（最快）", 0.3),
            ("0.5 秒（默认）", 0.5),
            ("0.8 秒（省 CPU）", 0.8),
            ("1.2 秒（更省）", 1.2),
            ("2.0 秒（最省）", 2.0),
            ("3.0 秒（几乎不占）", 3.0),
        ):
            self.cmb_interval.addItem(label, sec)
        cur_iv = float(getattr(self.cfg.pipeline, "min_interval_s", 0.5) or 0.5)
        idx_iv = self.cmb_interval.findData(cur_iv)
        self.cmb_interval.setCurrentIndex(idx_iv if idx_iv >= 0 else 1)
        self.cmb_interval.setToolTip(
            "两轮识别之间的间隔。**越大越省 CPU**。\n"
            "OCR 单次要几百毫秒，间隔太小会让识别线程一直不空闲。"
        )
        self.cmb_interval.currentIndexChanged.connect(self._on_interval_changed)
        int_row.addWidget(self.cmb_interval, 1)
        lv.addLayout(int_row)

        self.ck_skip_frame = QCheckBox("画面没变时跳过识别（省 CPU）")
        self.ck_skip_frame.setChecked(bool(getattr(self.engine.ecfg, "skip_unchanged_frame", True)))
        self.ck_skip_frame.setToolTip(
            "屏幕内容静止时不再重复 OCR —— 这是省 CPU 最有效的一项。\n"
            "关闭后每一轮都会重新识别（费 CPU，但极端情况下更灵敏）。"
        )
        self.ck_skip_frame.stateChanged.connect(self._on_skip_frame_changed)
        lv.addWidget(self.ck_skip_frame)

        # **自动朗读**：勾选后每张卡片生成就自动顺序播一遍（各一次）。
        self.ck_autoread = QCheckBox("自动朗读（生成就播，每句只播一次）")
        self.ck_autoread.setChecked(bool(getattr(self.engine.ecfg, "auto_read_all", False)))
        self.ck_autoread.setToolTip(
            "勾选后，每张卡片合成好就自动排进播放队列，按顺序读一遍。\n"
            "相当于自动点「全部播放」——但不影响你手动点单张。"
        )
        self.ck_autoread.stateChanged.connect(self._on_autoread_changed)
        lv.addWidget(self.ck_autoread)

        # **播放速率**：0.5~2.0，步进 0.1。
        #
        # 走 GPT-SoVITS 的 speed_factor（合成侧变速，音色不变调），
        # 不是播放时重采样（那样快放会变尖、慢放会变闷）。
        spd_row = QHBoxLayout()
        spd_row.addWidget(QLabel("语速"))
        self.sld_speed = QSlider(Qt.Horizontal)
        self.sld_speed.setRange(5, 20)          # 0.5x ~ 2.0x，值 = 倍数*10
        self.sld_speed.setSingleStep(1)
        self.sld_speed.setPageStep(2)
        cur_spd = float(getattr(self.cfg.tts, "speed_factor", 1.0) or 1.0)
        self.sld_speed.setValue(_safe_int(round(cur_spd * 10), 10, 5, 20))
        self.sld_speed.setToolTip(
            "合成语速。0.5~2.0 倍，步进 0.1。\n"
            "走模型自己的变速（音色不变调）。\n"
            "改完之后**新合成**的句子用新速率；已生成的句子要点⟳重新生成。"
        )
        self.sld_speed.valueChanged.connect(self._on_speed_changed)
        spd_row.addWidget(self.sld_speed, 1)
        self.lbl_speed = QLabel(f"{self.sld_speed.value() / 10:.1f}x")
        self.lbl_speed.setFixedWidth(42)
        self.lbl_speed.setStyleSheet("color:#8b949e;")
        spd_row.addWidget(self.lbl_speed)
        self.lbl_speed.setToolTip("双击恢复 1.0x")
        self.lbl_speed.mouseDoubleClickEvent = lambda e: self.sld_speed.setValue(10)
        lv.addLayout(spd_row)

        self.lbl_region_hint = QLabel("还没有区域。点工具栏「框选区域」。")
        self.lbl_region_hint.setWordWrap(True)
        self.lbl_region_hint.setStyleSheet("color:#8b949e;")
        lv.addWidget(self.lbl_region_hint)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 8, 8, 8)
        rv.addWidget(QLabel("识别结果（左侧按钮播放/停止；文字从屏幕消失后自动移除）"))
        self.card_list = CardList()
        self.card_list.toggle_requested.connect(self._on_toggle)
        self.card_list.retry_requested.connect(self._on_retry_card)
        rv.addWidget(self.card_list, 1)

        # **左栏放进滚动区**：默认窗口宽度下左栏放不下所有设置项
        # （读取间隔、自动朗读、语速、画面跳过…），以前末尾的控件会被裁掉，
        # 用户既看不到也没有滚动条提示，等于"功能消失"。
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.left_scroll.setFrameShape(QScrollArea.NoFrame)
        self.left_scroll.setWidget(left)
        self.left_scroll.setMinimumWidth(280)

        split.addWidget(self.left_scroll)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([340, 800])
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)
        # 保证默认宽度能放下工具栏上的 6 个主按钮 + 声库 + 更多
        self.setMinimumWidth(980)

        self.status = self.statusBar()
        # **TTS 服务状态灯**（用户要求）：一眼看出语音服务在不在，
        # 不用去翻日志。绿=就绪，黄=启动中，红=没在跑。
        self.lbl_tts = QLabel()
        self.lbl_tts.setToolTip("语音（TTS）推理服务状态。点它可以手动启动。\n"
                                "绿=就绪  黄=启动中  红=没在运行")
        self.lbl_tts.setCursor(Qt.PointingHandCursor)
        self.lbl_tts.mousePressEvent = lambda e: self._ensure_service()
        self.status.addPermanentWidget(self.lbl_tts)
        self._tts_light_on = False
        self._tts_light_next = 0.0
        self._set_tts_light("wait", "检测中…")

        self.lbl_stats = QLabel("待启动")
        self.status.addPermanentWidget(self.lbl_stats)

        self._reload_voices()
        # 恢复上次的识别区域（含单行/整段模式）
        self._restore_regions()
        self._refresh_regions()
        # 初始化模式下拉框：没有选中区域时是禁用的，提示文字也要先填上。
        # 必须放在 _build_ui 最后 —— 它会调 _refresh_cards()，而那需要 card_list 已建好。
        self._on_region_selected(None)
        # 界面起来后就去检查 TTS 服务，没跑就自动拉起来
        QTimer.singleShot(1200, self._ensure_service)

    # ------------------------------------------------------------------ #
    # TTS 推理服务
    # ------------------------------------------------------------------ #
    def _ensure_service(self, quiet: bool = False) -> None:
        """检查推理服务；没起来就在后台拉起来（不阻塞界面）。

        用户实测反馈："点开始识别，卡片全是'失败'，估计是服务没起来"——
        光装好运行时还不够，还得有 api_v2.py 在跑。目标机没有 Python，
        不能让用户自己去开命令行，所以这里自动处理。
        """
        from .. import runtime as R

        port = R.service_port(self.cfg)
        if R.service_ready(port):
            self.status.showMessage(f"TTS 推理服务运行中（端口 {port}）", 4000)
            return
        if getattr(self, "_svc_worker", None) is not None and self._svc_worker.isRunning():
            return
        if not quiet:
            self.status.showMessage("TTS 推理服务没在跑，正在自动启动…（首次要几十秒加载模型）")
        self._svc_worker = ServiceWorker(self.cfg, self)
        self._svc_worker.done.connect(self._on_service_done)
        self._svc_worker.start()

    def _on_service_done(self, res: dict) -> None:
        if res.get("ok"):
            # 记下 PID —— 关程序时只停**我们自己拉起来的**那个服务，
            # 用户自己在终端开的服务不该被我们杀掉
            if res.get("pid"):
                self._tts_pid = int(res["pid"])
            extra = "（本来就是运行中）" if res.get("already") else f"（PID {res.get('pid')}）"
            self.status.showMessage(f"TTS 推理服务已就绪{extra}", 8000)
        else:
            msg = res.get("message", "未知原因")
            self.status.showMessage(f"TTS 服务启动失败：{msg}", 20000)
            log.warning("TTS 服务启动失败：%s", res)

    # ------------------------------------------------------------------ #
    # 声库
    # ------------------------------------------------------------------ #
    def _open_runtime_dialog(self) -> None:
        from .runtime_dialog import RuntimeInstallDialog

        dlg = RuntimeInstallDialog(self)
        if dlg.exec() == QDialog.Accepted:
            # 装完可能换过安装位置，重新读一下配置
            from ..config import AppConfig

            fresh = AppConfig.load(default_config_path())
            if fresh.tts.runtime_root != self.cfg.tts.runtime_root:
                self.cfg.tts.runtime_root = fresh.tts.runtime_root
                self.status.showMessage("运行时位置已更新，重启 TTS 服务后生效", 8000)

    def _reload_voices(self) -> None:
        """扫描 voices/ 目录填充下拉框。往里解压新声库后重开界面即可看到。"""
        self.voice_store = VoiceStore(self.cfg.voices_root())
        names = self.voice_store.list_names()
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()
        for n in names:
            try:
                p = self.voice_store.load(n)
                self.voice_combo.addItem(p.display, userData=n)
            except Exception:
                self.voice_combo.addItem(f"{n}（档案损坏）", userData=n)
        cur = (self.cfg.tts.voice or "").strip()
        idx = self.voice_combo.findData(cur) if cur else -1
        if idx >= 0:
            self.voice_combo.setCurrentIndex(idx)
        self.voice_combo.blockSignals(False)
        if not names:
            self.voice_combo.addItem("（没有声库）")
        self.voice_combo.setEnabled(bool(names))

    def _on_voice_changed(self, idx: int) -> None:
        name = self.voice_combo.itemData(idx)
        if not name or name == (self.cfg.tts.voice or ""):
            return
        try:
            self.cfg.tts.voice = name
            self.cfg.apply_voice(self.cfg.voices_root())
            p = self.voice_store.load(name)
            p.write_api_config(
                self.voice_store.dir_of(name),
                runtime_root=(self.cfg.tts.runtime_root or None),
            )
            self.cfg.save(default_config_path())
        except Exception as e:
            QMessageBox.warning(self, "切换声库失败", str(e))
            return
        self.status.showMessage(
            f"已切换到声库「{name}」——需要重启 TTS 推理服务（api_v2.py）才生效", 12000
        )

    # ------------------------------------------------------------------ #
    # 区域
    # ------------------------------------------------------------------ #
    def _open_settings_dialog(self) -> None:
        """高级设置：**需要重启识别才生效**的项集中放这里。

        用户要求："不能运行中修改的配置放到二级页面统一管理"。
        主界面只留可以即时生效的东西（读取间隔、语速、音量、自动朗读…），
        OCR 线程数/识别上限/缓存这类改完要重建会话的，统一收到这个对话框。
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("高级设置（改完需要「停止识别 → 开始识别」才生效）")
        dlg.setMinimumWidth(460)
        v = QVBoxLayout(dlg)
        form = QFormLayout()

        sp_threads = QSpinBox()
        sp_threads.setRange(0, 32)
        sp_threads.setValue(int(getattr(self.cfg.ocr, "intra_threads", 0) or 0))
        sp_threads.setToolTip("0 = 自动标定。想少占 CPU 就填 4~6（8~10 会明显挤占游戏）。")
        form.addRow("OCR 线程数（0=自动）", sp_threads)

        sp_det = QSpinBox()
        sp_det.setRange(320, 1920)
        sp_det.setSingleStep(32)
        sp_det.setValue(int(getattr(self.cfg.ocr, "det_limit_side_len", 960) or 960))
        sp_det.setToolTip("检测边长上限。改小更快但小字识别变差；960 是默认。")
        form.addRow("检测边长上限", sp_det)

        sp_lines = QSpinBox()
        sp_lines.setRange(5, 200)
        sp_lines.setValue(int(getattr(self.cfg.ocr, "max_lines", 25) or 25))
        sp_lines.setToolTip("每轮最多识别多少行。OCR 耗时与行数近似线性。")
        form.addRow("每轮最多行数", sp_lines)

        sp_score = QDoubleSpinBox()
        sp_score.setRange(0.0, 1.0)
        sp_score.setSingleStep(0.05)
        sp_score.setValue(float(getattr(self.cfg.ocr, "min_score", 0.5) or 0.5))
        sp_score.setToolTip("低于这个置信度的行会被丢掉。识别结果太脏可以调高。")
        form.addRow("最低置信度", sp_score)

        sp_ttl = QSpinBox()
        sp_ttl.setRange(0, 3600)
        sp_ttl.setValue(int(getattr(self.cfg.cache, "ttl_s", 30) or 0))
        sp_ttl.setToolTip("音频缓存多少秒没被用到就自动删除。0 = 不自动过期。")
        form.addRow("缓存自动清理（秒，0=关）", sp_ttl)

        v.addLayout(form)
        note = QLabel("提示：以上都会在下次「开始识别」时重新初始化。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8b949e;")
        v.addWidget(note)

        bb = QHBoxLayout()
        bb.addStretch(1)
        ok = QPushButton("保存")
        cancel = QPushButton("取消")
        ok.setDefault(True)
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        bb.addWidget(ok)
        bb.addWidget(cancel)
        v.addLayout(bb)

        if dlg.exec() != QDialog.Accepted:
            return
        self.cfg.ocr.intra_threads = int(sp_threads.value())
        self.cfg.ocr.det_limit_side_len = int(sp_det.value())
        self.cfg.ocr.max_lines = int(sp_lines.value())
        self.cfg.ocr.min_score = float(sp_score.value())
        self.cfg.cache.ttl_s = float(sp_ttl.value())
        try:
            self.engine.cache.ttl_s = float(sp_ttl.value())
        except Exception:
            pass
        try:
            self.cfg.save(default_config_path())
        except Exception as e:
            log.warning("保存高级设置失败：%s", e)
        self.status.showMessage("高级设置已保存；停止识别再开始即可生效", 8000)

    def _open_hotkey_dialog(self) -> None:
        """设置「截图朗读」的全局快捷键：**点按钮 -> 直接按**。

        为什么用"按钮 + 全局钩子"而不是输入框 + Qt 按键事件（用户反馈的 bug）：
          * Qt 的 `mousePressEvent` **在很多鼠标/驱动下收不到前进侧键**
            （用户实测：前进侧键设不上）。判断靠 `Qt.ForwardButton`，
            只有驱动把它当标准浏览器前进键时才成立。
          * 走全局低级钩子（WH_MOUSE_LL / WH_KEYBOARD_LL）抓，就与鼠标
            品牌、驱动、焦点全都没关系。
        """
        import queue as _q

        dlg = QDialog(self)
        dlg.setWindowTitle("截图朗读 · 快捷键")
        dlg.setMinimumWidth(420)
        v = QVBoxLayout(dlg)

        hint = QLabel(
            "点下面的按钮，然后**直接按**：\n"
            "  · 键盘组合键 —— 例如 Ctrl+Alt+S\n"
            "  · 鼠标侧键   —— 直接按鼠标侧面的前进/后退键\n"
            "正在等待时按 Esc 可放弃。这是**全局**快捷键，游戏全屏在前台时也有效。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8b949e;")
        v.addWidget(hint)

        cur0 = getattr(self.cfg.snap, "hotkey", "") or ""
        lbl_now = QLabel("当前：" + (cur0 or "未设置"))
        lbl_now.setStyleSheet("color:#58a6ff;font-size:14px;")
        v.addWidget(lbl_now)

        btn = QPushButton("点击设置快捷键")
        btn.setMinimumHeight(38)
        v.addWidget(btn)

        state = {"spec": cur0, "armed": False}

        poll = QTimer(dlg)
        poll.setInterval(60)

        def stop_capture() -> None:
            state["armed"] = False
            poll.stop()
            try:
                self._hotkeys.end_capture()
            except Exception:
                pass
            btn.setText("点击设置快捷键")

        def on_captured() -> None:
            try:
                spec = self._hotkeys.captured.get_nowait()
            except _q.Empty:
                return
            if spec == "esc":
                # 等待中按 Esc = 放弃设置（Esc 本身还是可以手动写成快捷键，
                # 但"正在等待时按 Esc"用户显然是想取消）
                lbl_now.setText("当前：" + (state["spec"] or "未设置") + "（已放弃）")
                stop_capture()
                return
            if parse_binding(spec) is None:
                return
            state["spec"] = spec
            lbl_now.setText("当前：" + spec)
            stop_capture()

        poll.timeout.connect(on_captured)

        def arm() -> None:
            if state["armed"]:
                stop_capture()
                return
            state["armed"] = True
            btn.setText("请按键…（按 Esc 放弃）")
            try:
                self._hotkeys.begin_capture()
            except Exception as e:
                log.warning("开始捕获快捷键失败：%s", e)
                state["armed"] = False
                btn.setText("点击设置快捷键")
                return
            poll.start()

        btn.clicked.connect(arm)

        if self._snap_hotkey_failed:
            warn = QLabel("⚠ 全局快捷键没能启用（权限或系统限制），捕获和触发都可能无效。\n"
                          "按钮功能不受影响。")
            warn.setWordWrap(True)
            warn.setStyleSheet("color:#d29922;")
            v.addWidget(warn)

        bb = QHBoxLayout()
        bb.addStretch(1)
        ok = QPushButton("保存")
        cancel = QPushButton("取消")
        ok.setDefault(True)
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        bb.addWidget(ok)
        bb.addWidget(cancel)
        v.addLayout(bb)

        try:
            accepted = dlg.exec() == QDialog.Accepted
        finally:
            stop_capture()
        if not accepted:
            return

        spec = (state["spec"] or "").strip()
        if not spec:
            self.status.showMessage("快捷键没设置（点按钮后按一下键）", 5000)
            return
        if parse_binding(spec) is None:
            QMessageBox.warning(self, "快捷键无效", f"看不懂这个写法：{spec}")
            return

        self.cfg.snap.hotkey = spec
        self._hotkeys.set_bindings({"snap": spec})
        try:
            self.cfg.save(default_config_path())
        except Exception as e:
            log.warning("保存快捷键失败：%s", e)
        self.status.showMessage(f"截图朗读快捷键已设为 {spec}", 5000)

    def _pick_regions(self) -> None:
        """打开全屏框选层。

        用户要求：点框选时**自动隐藏主窗口**，免得挡住要框的内容；
        框选结束（确认或取消）再显示回来。
        """
        self._was_visible = self.isVisible()
        self.hide()
        # 给窗口管理器一点时间真正隐藏，否则框选层可能被主窗口挡住
        QTimer.singleShot(120, self._open_picker)

    # ------------------------------------------------ 截图朗读（新场景） --
    def snap_read(self) -> None:
        """「截图朗读」入口：**先把屏幕冻结成一张图**，再让用户在图上裁切。

        为什么先冻结：框选过程中游戏/视频还在动，如果框完再去截屏，
        拿到的可能已经不是同一帧 —— 表现就是"框了却没读对"。
        冻结之后"框到的"就是"读到的"。
        """
        # **已经有框选层开着就别再开一个**（用户要求：框选期间不允许快捷键
        # 再次唤出截图朗读）。叠两个全屏置顶层会看起来像"界面卡住"。
        if self._picker_busy():
            self.status.showMessage("正在框选中，请先完成或取消当前框选", 4000)
            return
        # 清理上一次没走完的（比如用户连按了两次快捷键）
        self._end_snap(silent=True)
        self._snap_pending = True
        self._was_visible = self.isVisible()

        # **抓图前必须把本程序自己的窗口藏起来**，否则会被拍进图里
        # （主窗、悬浮条、区域角标）。记下原来谁可见，抓完恢复。
        self._snap_vis = {
            "overlay": self.overlay.isVisible(),
            "marks": self.region_overlay.isVisible(),
        }
        self.hide()
        try:
            self.overlay.hide()
            self.region_overlay.hide()
        except Exception:
            pass
        # 等窗口真正从屏幕上消失（Windows 上 hide() 是异步的）
        QTimer.singleShot(160, self._open_snap_picker)

    def _open_snap_picker(self) -> None:
        """抓「鼠标所在那个屏幕」的静态图，然后在图上框选。"""
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        if screen is None:
            self._snap_abort("没能确定鼠标在哪个屏幕上")
            return
        try:
            frozen = screen.grabWindow(0)
        except Exception as e:
            self._snap_abort(f"抓屏失败：{e}")
            return
        if frozen is None or frozen.isNull():
            self._snap_abort("抓屏返回空图（可能被显卡/录屏软件占用）")
            return

        # 记下屏幕几何，后面把局部坐标换算成全局坐标
        self._snap_screen_geo = screen.geometry()
        try:
            self._snap_monitor = QGuiApplication.screens().index(screen)
        except Exception:
            self._snap_monitor = 0

        # 图已经抓好了，把窗口恢复回来（框选层是置顶全屏，会盖住它们）
        self._restore_after_pick()

        self._snap_picker = SnapPicker(frozen)
        self._snap_picker.picked.connect(self._guarded(self._on_snap_picked, "截图朗读确定"))
        self._snap_picker.cancelled.connect(self._guarded(self._on_snap_cancelled, "截图朗读取消"))
        # **框选层期间开全局 Esc**：它是全屏置顶窗口，一旦失了焦点
        # Qt 就收不到 Esc，用户会被困在框选界面里（用户实测：只能重启电脑）。
        try:
            self._hotkeys.set_escape(True)
        except Exception:
            pass
        self._snap_guard.start(180_000)      # 3 分钟保险丝
        self._snap_picker.start(screen.geometry())
        self.status.showMessage("截图朗读：画面已冻结，框选要读的那一段（Esc 取消）", 6000)

    def _guarded(self, fn, what: str):
        """把槽函数包成**绝不抛异常**的版本。

        为什么必须这样：这是全屏置顶的框选层。槽函数里哪怕抛一个异常，
        框选层就不会被关掉，又没有别的入口能退出 —— 用户只能重启电脑。
        实测确实发生过，所以这里统一兜住，异常只记日志。
        """
        def _wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                log.exception("%s 处理失败：%s", what, e)
                try:
                    self._snap_hard_cancel()
                except Exception:
                    pass
                return None
        return _wrapped

    def _snap_hard_cancel(self) -> None:
        """**兜底取消**：无论框选层处于什么状态，都强制关掉它并恢复一切。

        可以重复调用（幂等）。三个地方会用到：
          * 全局 Esc（框选层失焦时也能退出）
          * 3 分钟超时保险丝
          * 槽函数抛异常时
        """
        log.warning("截图朗读：强制取消框选层")
        try:
            self._snap_guard.stop()
        except Exception:
            pass
        try:
            self._hotkeys.set_escape(False)
        except Exception:
            pass
        p = getattr(self, "_snap_picker", None)
        self._snap_picker = None
        if p is not None:
            try:
                p.hide()
            except Exception:
                pass
            try:
                p.close()
            except Exception:
                pass
            try:
                p.deleteLater()
            except Exception:
                pass
        self._snap_pending = False
        try:
            self._restore_after_pick()
        except Exception:
            pass
        try:
            self._restore_snap_windows()
        except Exception:
            pass
        # 截手中途被打断时，其他区域可能已经被停用了 —— 一并恢复
        try:
            snap = getattr(self, "_snap_enabled", None)
            if snap:
                self.engine.apply_region_enabled_map(snap)
                self._snap_enabled = {}
            self._refresh_regions()
        except Exception:
            pass
        self.status.showMessage("截图朗读：已取消", 3000)

    def _snap_abort(self, msg: str) -> None:
        """抓屏/框选失败时的统一善后。"""
        log.warning("截图朗读取消：%s", msg)
        self._snap_pending = False
        self._restore_after_pick()
        self._restore_snap_windows()
        self.status.showMessage(f"截图朗读：{msg}", 8000)

    def _restore_snap_windows(self) -> None:
        """把抓图前藏起来的窗口按原样恢复。"""
        vis = getattr(self, "_snap_vis", None) or {}
        try:
            if vis.get("marks"):
                self.region_overlay.show_overlay()
        except Exception:
            pass
        try:
            if vis.get("overlay"):
                self.overlay.show()
        except Exception:
            pass

    def _on_snap_cancelled(self) -> None:
        """右键 / Esc 取消：什么都不做，把状态清干净。"""
        self._snap_pending = False
        self._snap_picker = None
        try:
            self._snap_guard.stop()
        except Exception:
            pass
        try:
            self._hotkeys.set_escape(False)
        except Exception:
            pass
        self._restore_after_pick()
        self._restore_snap_windows()
        self.status.showMessage("截图朗读：已取消", 3000)

    def _on_snap_picked(self, rect, image) -> None:
        """框选确定：**这时才**暂停其他区域 -> 一次性 OCR -> 建卡 -> 朗读。"""
        self._snap_pending = False
        self._snap_picker = None
        try:
            self._snap_guard.stop()
        except Exception:
            pass
        try:
            self._hotkeys.set_escape(False)
        except Exception:
            pass
        geo = getattr(self, "_snap_screen_geo", None)
        mon = int(getattr(self, "_snap_monitor", 0) or 0)

        # ① 记下其他区域的开关状态，并**暂停它们**（用户要求：确定之后才暂停）
        self._snap_enabled = self.engine.region_enabled_map()
        for rid in self._snap_enabled:
            self.engine.update_region(rid, enabled=False)

        # ② 建临时区域：**enabled=False —— 它不参与轮询**，这是关键。
        #    以前它参与轮询，OCR 每轮重跑、内容轻微抖动就会让整段卡片重建，
        #    于是同一段被读第二遍。现在识别是"对这张冻结图做一次"，与轮询无关。
        gx, gy = (geo.left(), geo.top()) if geo is not None else (0, 0)
        global_rect = (gx + int(rect.left()), gy + int(rect.top()),
                       int(rect.width()), int(rect.height()))
        mode = getattr(self.cfg.snap, "mode", "paragraph") or "paragraph"
        r = self.engine.add_region("截图朗读", global_rect, mon, mode=mode, temp=True)
        self.engine.update_region(r.id, enabled=False)
        self._snap_rid = r.id
        self._snap_started = False
        self._snap_t0 = time.time()

        # ③ **无论引擎本来在不在跑，都要确认 TTS 服务活着。**
        # 以前只在"引擎没跑"时才拉起服务 —— 如果引擎在跑但服务已经挂了，
        # 截图朗读的卡片会一直停在"合成中"（用户截图里的现象）。
        self._ensure_service()
        # 引擎没跑就拉起来（用户选择：读完保持运行，不主动停）
        if not self.engine.running:
            self.engine.start()
            self.act_toggle.setText("■ 停止识别")
            self.overlay.set_running(True)

        self._restore_after_pick()
        self._restore_snap_windows()
        self._refresh_regions()
        self._refresh_cards()
        self.status.showMessage("截图朗读：正在识别这段…", 6000)

        # ④ OCR 放后台线程，避免大区域（实测整屏 800ms）卡住界面
        try:
            arr = qimage_to_rgb_array(image)
        except Exception as e:
            self._end_snap()
            self.status.showMessage(f"截图朗读：图片转换失败（{e}）", 8000)
            return
        # **看门狗必须现在就启动**。
        # 以前 `_snap_timer` 只在 `_on_snap_ocr_done` 里才 start() ——
        # 于是 OCR 线程一旦卡住不返回，**根本没有任何东西会来清理**，
        # 界面就永远停在"正在识别这段…"（用户实测的"卡在识别中，不读也不清理"）。
        self._snap_play_t0 = 0.0
        self._snap_timer.start(150)
        log.info("截图朗读：开始 OCR（%d×%d）", arr.shape[1], arr.shape[0])
        threading.Thread(target=self._snap_ocr_worker, args=(r.id, arr),
                         name="snap-ocr", daemon=True).start()

    def _snap_ocr_worker(self, rid: str, arr) -> None:
        """后台跑一次 OCR，结果发信号回主线程。"""
        try:
            lines = self.engine.read_frame_once(arr)
        except Exception as e:
            log.warning("截图朗读 OCR 线程异常：%s", e)
            lines = []
        log.info("截图朗读：OCR 返回 %d 行", len(lines))
        self.snap_ocr_done.emit((rid, lines))

    def _on_snap_ocr_done(self, payload) -> None:
        rid, lines = payload
        if not rid or rid != getattr(self, "_snap_rid", None):
            return                      # 已经清场了，结果作废
        texts = [t for t, _t, _l in lines]
        spans = [(t, l) for _x, t, l in lines]
        if not texts:
            self.status.showMessage("截图朗读：这块区域没识别到文字", 6000)
            self._end_snap()
            return

        # 整段规则：合并折行 + 按标点断句（和整段模式完全一致）
        try:
            sentences = merge_lines_to_sentences(
                texts, max_chars=self.cfg.segment.max_chars or 80, spans=spans)
        except Exception as e:
            log.warning("截图朗读合并分句失败，退回逐行：%s", e)
            sentences = texts
        if not sentences:
            self.status.showMessage("截图朗读：这块区域没识别到文字", 6000)
            self._end_snap()
            return

        out = self.engine.observe_text(rid, sentences)
        log.info("截图朗读识别到 %d 行 / %d 句 / 建卡 %d 张",
                 len(texts), len(sentences), out)
        self._refresh_regions()
        self._refresh_cards()

        # 顺序朗读一遍（第一句没好也没关系，播放循环会原地等）
        self.engine.play_paragraph(rid)
        self._snap_started = True
        self._snap_play_t0 = time.time()      # 清场定时器的"最早可清场时刻"从这里算
        self.status.showMessage(f"截图朗读：开始朗读（{len(sentences)} 句）", 5000)
        self._snap_timer.start(150)

    def _snap_tick(self) -> None:
        """截图朗读的看门狗 + 清场判定。

        三个阶段，每个都有上界，**不允许无限停在某个状态**：

          A. 等 OCR      —— 超过 20 秒判超时（OCR 线程可能卡在锁或模型上）
          B. 等开播      —— 有卡片后由 `_on_snap_ocr_done` 开播
          C. 等播完      —— 播完清场；整段超过 180 秒也强制清场

        用户实测"卡在识别中，不读也不清理"就是 A 阶段没有超时导致的。
        """
        rid = getattr(self, "_snap_rid", None)
        if not rid:
            self._snap_timer.stop()
            return
        if self.engine.store.region(rid) is None:
            self._end_snap(silent=True)
            return

        age = time.time() - getattr(self, "_snap_t0", 0.0)

        # A. 还在等 OCR 结果
        if not getattr(self, "_snap_started", False):
            if age > 20:
                log.warning("截图朗读：OCR 超过 20 秒没结果，放弃")
                self.status.showMessage(
                    "截图朗读：识别超时（20 秒没有结果），已清理", 8000)
                self._end_snap()
            return

        # C. 已经在播：先保证最低驻留（防竞态），再看是否播完
        if time.time() - getattr(self, "_snap_play_t0", 0.0) < 1.5:
            return
        if self.engine.region_playback_done(rid):
            self._end_snap()
            return
        # 整段总时长兜底：卡住的句子不该把界面永远占住
        if age > 180:
            log.warning("截图朗读：整段超过 180 秒仍未播完，强制清场")
            self.status.showMessage("截图朗读：朗读超时，已清理", 8000)
            self._end_snap()

    def _end_snap(self, silent: bool = False) -> None:
        """清场：删掉临时区域 + 恢复常驻区域原来的开启/停用状态。

        **必须 try/finally**：只要这里半途抛异常，"其他区域"就会永久停在
        停用状态 —— 界面看起来就是"识别一直卡住"（用户实测）。
        所以无论发生什么，最后一定把开关恢复回去。
        """
        rid = getattr(self, "_snap_rid", None)
        try:
            self._snap_timer.stop()
        except Exception:
            pass
        self._snap_started = False
        try:
            if rid:
                # 把临时区域从"朗读中/整段状态"里摘干净，避免留下悬空记录
                try:
                    self.engine._playing_regions.discard(rid)      # noqa: SLF001
                    self.engine._para_state.pop(rid, None)         # noqa: SLF001
                except Exception:
                    pass
                try:
                    self.engine.remove_region(rid)
                except Exception as e:
                    log.warning("删除临时区域失败：%s", e)
        finally:
            self._snap_rid = None
            # **无论前面有没有出错，都恢复其他区域**
            snap = getattr(self, "_snap_enabled", None)
            if snap:
                try:
                    self.engine.apply_region_enabled_map(snap)
                except Exception as e:
                    log.warning("恢复区域开关失败：%s", e)
                self._snap_enabled = {}
            try:
                self._refresh_regions()
                self._refresh_cards()
            except Exception:
                pass
        if not silent:
            self.status.showMessage("截图朗读：读完，已清理临时区域", 4000)

    def _open_picker(self) -> None:
        self._picker = RegionPicker()
        self._picker.picked.connect(self._on_picked)
        # 取消也要把主窗口显示回来，否则程序就"消失"了
        try:
            self._picker.cancelled.connect(self._on_pick_cancelled)
        except Exception:
            pass
        self._picker.start()

    def _restore_after_pick(self) -> None:
        if getattr(self, "_was_visible", True):
            self.show()
            self.raise_()
            self.activateWindow()

    def _on_pick_cancelled(self) -> None:
        # 取消时也要清掉截图朗读的待办标记 —— 否则下一次用普通的
        # 「框选区域」会被当成截图朗读走掉（实测过的 bug）。
        self._snap_pending = False
        self._picker = None
        self._restore_after_pick()
        self.status.showMessage("已取消框选", 3000)

    def _on_picked(self, rects: List) -> None:
        self._picker = None
        self._restore_after_pick()
        if not rects:
            self.status.showMessage("没有框选任何区域", 4000)
            return
        # 新框的区域沿用上次用过的模式：连着框好几个的时候不用一个个改
        mode = self.cmb_mode.currentData() if self.cmb_mode.isEnabled() else "line"
        for rect, mon in rects:
            self.engine.add_region(f"区域{len(self.engine.store.regions) + 1}", rect, mon,
                                   mode=mode or "line")
            log.info("新增区域 %s", self.engine.store.regions[-1].label())
        self._refresh_regions()
        self._save_regions()
        tag = "整段" if mode == "paragraph" else "单行"
        self.status.showMessage(f"已添加 {len(rects)} 个区域（{tag}模式）", 5000)

    def _restore_regions(self) -> None:
        """启动时恢复上次的区域（含单行/整段模式）。"""
        try:
            items = self.cfg.load_regions()
        except Exception as e:
            log.warning("读取识别区域失败：%s", e)
            return
        if not items:
            return
        n = self.engine.store.restore_regions(items)
        if n:
            log.info("恢复了 %d 个识别区域", n)
            self.status.showMessage(f"已恢复 {n} 个识别区域", 5000)

    def _save_regions(self) -> None:
        """区域有任何变动就落盘 —— 否则重启后整段/单行的设置会丢。"""
        try:
            self.cfg.save_regions(self.engine.store.regions)
        except Exception as e:
            log.warning("保存识别区域失败：%s", e)

    def _on_region_selected(self, item, _prev=None) -> None:
        """选中区域时同步「识别模式」下拉框 + 只显示该区域的卡片。"""
        self.cmb_mode.blockSignals(True)
        if item is not None:
            r = self.engine.store.region(item.data(Qt.UserRole))
            if r is not None:
                idx = self.cmb_mode.findData(r.mode)
                if idx >= 0:
                    self.cmb_mode.setCurrentIndex(idx)
            self.cmb_mode.setEnabled(True)
        else:
            self.cmb_mode.setEnabled(False)
        self.cmb_mode.blockSignals(False)
        self._update_mode_hint()
        self._refresh_cards()

    def _on_mode_changed(self, idx: int) -> None:
        item = self.region_list.currentItem()
        if item is None:
            return
        rid = item.data(Qt.UserRole)
        mode = self.cmb_mode.itemData(idx)
        r = self.engine.store.region(rid)
        if r is None or r.mode == mode:
            return
        self.engine.update_region(rid, mode=mode)
        # 换模式后旧的卡片语义就不对了（单行的碎句 vs 整段的合并句），清掉重来
        self.engine.clear_region_cards(rid)
        self._refresh_regions()
        self._refresh_cards()
        self._update_mode_hint()
        self._save_regions()
        label = "整段（自动连续朗读）" if mode == "paragraph" else "单行（手动点播）"
        self.status.showMessage(f"区域已切换为「{label}」——重新开始识别后生效", 6000)

    def _update_mode_hint(self) -> None:
        mode = self.cmb_mode.currentData()
        if mode == "paragraph":
            self.lbl_mode_hint.setText(
                "整段：折行会合并、按标点分句。<b>页面稳定后</b>点下面的按钮，"
                "从上到下顺序朗读；滚动中不播报。"
            )
        else:
            self.lbl_mode_hint.setText(
                "单行：每行一句，识别后在右边手动点播放。"
            )
        self._update_play_button()

    def _first_para_card(self):
        """列表里第一张"整段模式"的卡片（界面上最上面那张）。

        播放按钮的可用状态**就绑在它身上**（用户要求）——
        只要第一句能播，整段朗读就有意义；第一句还没就绪就别让人点。
        """
        for c in self.engine.store.sorted_cards():
            r = self.engine.store.region(c.region_id)
            if r is not None and r.mode == "paragraph":
                return c
        return None

    def _on_autoread_changed(self, state: int) -> None:
        """自动朗读开关：立刻生效并记住。"""
        on = _is_checked(state)
        try:
            self.engine.ecfg.auto_read_all = on
        except Exception as e:
            log.warning("设置自动朗读失败：%s", e)
            return
        self.cfg.pipeline.auto_read_all = on
        self.overlay.set_autoread(on)        # 悬浮窗的「自动」按钮同步
        try:
            self.cfg.save(default_config_path())
        except Exception:
            pass
        if on:
            # 打开时把**已经在列表里、还没播过**的卡片也排进队列，
            # 否则用户会以为"勾了却只有新卡片才自动播"。
            n = 0
            for c in self.engine.store.sorted_cards():
                if not c.auto_done:
                    c.queued = True
                    n += 1
            self.status.showMessage(
                f"自动朗读：已开启（队列里补了 {n} 句，按顺序播一遍）" if n
                else "自动朗读：已开启（新生成的句子会自动播）",
                5000,
            )
        else:
            self.engine.stop_all_paragraphs()
            self.status.showMessage("自动朗读：已关闭", 3000)

    def _on_speed_changed(self, v: int) -> None:
        """语速滑条：0.5~2.0 倍，步进 0.1。"""
        factor = max(0.5, min(2.0, v / 10.0))
        self.lbl_speed.setText(f"{factor:.1f}x")
        try:
            self.engine.set_speed(factor)
        except Exception as e:
            log.warning("设置语速失败：%s", e)
        self.cfg.tts.speed_factor = factor
        if not getattr(self, "_spd_saving", False):
            self._spd_saving = True
            QTimer.singleShot(500, self._save_speed)

    def _save_speed(self) -> None:
        self._spd_saving = False
        try:
            self.cfg.save(default_config_path())
        except Exception as e:
            log.warning("保存语速失败：%s", e)

    def _on_interval_changed(self, idx: int) -> None:
        """读取间隔：立刻生效并写回配置。

        `_detect_loop` 每轮都会重新读 `self.ecfg.poll_interval_s`，
        所以不用重启识别 —— 下一轮就按新间隔走。
        """
        sec = self.cmb_interval.itemData(idx)
        if sec is None:
            return
        try:
            self.engine.ecfg.poll_interval_s = float(sec)
        except Exception as e:
            log.warning("设置读取间隔失败：%s", e)
            return
        self.cfg.pipeline.min_interval_s = float(sec)
        if not getattr(self, "_iv_saving", False):
            self._iv_saving = True
            QTimer.singleShot(400, self._save_interval)
        self.status.showMessage(f"读取间隔已设为 {sec} 秒（下一轮生效）", 4000)

    def _save_interval(self) -> None:
        self._iv_saving = False
        try:
            self.cfg.save(default_config_path())
        except Exception as e:
            log.warning("保存读取间隔失败：%s", e)

    def _on_skip_frame_changed(self, state: int) -> None:
        # **注意用 _is_checked**：PySide6 的 Qt.CheckState 不是 IntEnum，
        # 直接 `state == Qt.Checked` 恒为 False，会让这个开关完全失效。
        on = _is_checked(state)
        try:
            self.engine.ecfg.skip_unchanged_frame = on
        except Exception as e:
            log.warning("设置帧跳过失败：%s", e)
            return
        self.cfg.ocr.skip_unchanged_frame = on
        try:
            self.cfg.save(default_config_path())
        except Exception:
            pass
        self.status.showMessage(
            "画面没变时跳过 OCR：已开启（省 CPU）" if on else "画面没变时跳过 OCR：已关闭",
            3000,
        )

    def _on_volume_changed(self, v: int) -> None:
        """音量滑动条：实时改播放器增益，并写回配置。"""
        self.lbl_volume.setText(f"{v}%")
        try:
            self.engine.set_volume(v)
        except Exception as e:
            log.warning("设置音量失败：%s", e)
        self.cfg.audio.volume = int(v)
        if not getattr(self, "_vol_saving", False):
            # 拖动过程中会连续触发，防抖一下再落盘，别把磁盘刷爆
            self._vol_saving = True
            QTimer.singleShot(600, self._save_volume)

    def _save_volume(self) -> None:
        self._vol_saving = False
        try:
            self.cfg.save(default_config_path())
        except Exception as e:
            log.warning("保存音量失败：%s", e)

    def _update_play_button(self) -> None:
        """「朗读这个区域」按钮：选中了区域就可用（单行/整段都行）。

        **不做"第一句就绪"的门槛**（用户要求）：按钮常亮，
        点了就从列表第一项开始读 —— 第一句还没合成好也没关系，
        播放循环会原地等它，不会跳过。
        """
        item = self.region_list.currentItem()
        self.btn_play_para.setEnabled(item is not None)

    def _on_play_paragraph(self) -> None:
        item = self.region_list.currentItem()
        if item is None:
            return
        rid = item.data(Qt.UserRole)
        n = self.engine.play_paragraph(rid)
        if n:
            self.status.showMessage(f"开始整段朗读，共 {n} 句（从上到下）", 5000)
        else:
            self.status.showMessage("这个区域还没有可读的内容", 4000)

    def _on_play_all_paragraphs(self) -> None:
        """悬浮窗 / 左侧的「全部播放」：把**所有区域**按顺序读一遍。

        用户要求：单行模式也要能顺序播放，和整段一样。
        队列全局按出现顺序出队，所以听感就是界面上从上到下。
        """
        n = self.engine.play_paragraphs()
        if n:
            self.status.showMessage(f"顺序朗读开始，共 {n} 句（从上到下）", 5000)
        else:
            self.status.showMessage("还没有可读的内容（先框选并等页面稳定）", 4000)

    def _update_overlay_play(self) -> None:
        """悬浮窗的「全部播放」：**常亮**（用户要求）。

        点了就从列表**第一项**开始，从上到下顺序读。
        第一句还没合成好也没关系 —— 播放循环会原地等它。
        """
        if hasattr(self.overlay, "btn_play_all"):
            self.overlay.btn_play_all.setEnabled(True)

    def _refresh_regions(self) -> None:
        # 重建列表会让选中项丢失，之后「识别模式」下拉就没作用对象了
        # （切换模式 -> 刷新 -> 想再切一次就不响应）。所以按 id 记住再恢复。
        keep = None
        cur = self.region_list.currentItem()
        if cur is not None:
            keep = cur.data(Qt.UserRole)

        self.region_list.blockSignals(True)
        self.region_list.clear()
        for r in self.engine.store.regions:
            it = QListWidgetItem(r.label())
            it.setData(Qt.UserRole, r.id)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if r.enabled else Qt.Unchecked)
            it.setForeground(QColor(r.color))
            self.region_list.addItem(it)
        self.region_list.blockSignals(False)

        if keep:
            for i in range(self.region_list.count()):
                if self.region_list.item(i).data(Qt.UserRole) == keep:
                    self.region_list.setCurrentRow(i)   # 会触发 _on_region_selected
                    break

        self.lbl_region_hint.setVisible(not self.engine.store.regions)
        colors = {r.id: r.color for r in self.engine.store.regions}
        for v in self.views:
            v.set_region_colors(colors)
        self._sync_marks()

    def _sync_marks(self) -> None:
        self.region_overlay.set_regions(self.engine.store.regions)
        if self.act_marks.isChecked() and self.engine.store.regions:
            self.region_overlay.show_overlay()
        else:
            self.region_overlay.hide()

    def _toggle_marks(self, on: bool) -> None:
        if on:
            self._sync_marks()
        else:
            self.region_overlay.hide()

    def _on_region_item_changed(self, item: QListWidgetItem) -> None:
        rid = item.data(Qt.UserRole)
        self.engine.update_region(rid, enabled=item.checkState() == Qt.Checked)
        self._sync_marks()
        self._save_regions()

    def eventFilter(self, obj, ev) -> bool:
        """区域列表里按 Delete / Backspace 直接删除选中区域（用户要求）。"""
        try:
            if obj is getattr(self, "region_list", None) and ev.type() == QEvent.KeyPress:
                if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                    self._remove_selected_region()
                    return True
        except Exception:
            pass
        return super().eventFilter(obj, ev)

    def _remove_selected_region(self) -> None:
        item = self.region_list.currentItem()
        if item is None:
            self.status.showMessage("先在左边选中一个区域", 4000)
            return
        rid = item.data(Qt.UserRole)
        n = self.engine.remove_region(rid)
        self._refresh_regions()
        self._refresh_cards()
        self._save_regions()
        self.status.showMessage(f"已删除区域，连带移除 {n} 张卡片", 4000)

    # ------------------------------------------------------------------ #
    # 卡片（两个视图一起刷）
    # ------------------------------------------------------------------ #
    def _refresh_cards(self) -> None:
        sel = self.region_list.currentItem()
        sel_id = sel.data(Qt.UserRole) if sel else None

        # 主列表：只显示当前选中的区域
        wanted: Dict[str, Card] = {}
        for c in self.engine.store.sorted_cards():
            if sel_id and c.region_id != sel_id:
                continue
            wanted[c.id] = c

        # 悬浮窗：显示**所有**区域（多个区域同时开着才看得到全部）
        everyone: Dict[str, Card] = {c.id: c for c in self.engine.store.sorted_cards()}

        for v in self.list_views:
            for cid in list(v._rows):  # noqa: SLF001
                if cid not in wanted:
                    v.remove(cid)
            for c in wanted.values():
                v.add_or_update(c)
        for v in self.all_views:
            for cid in list(v._rows):  # noqa: SLF001
                if cid not in everyone:
                    v.remove(cid)
            for c in everyone.values():
                v.add_or_update(c)
        self._update_play_button()
        self._update_overlay_play()

    def _refresh_card(self, cid: str) -> None:
        c = self.engine.store.card(cid)
        for v in self.views:
            if c is None:
                v.remove(cid)
            else:
                v.add_or_update(c)

    def _remove_card(self, cid: str) -> None:
        for v in self.views:
            v.remove(cid)

    def _clear_cards(self) -> None:
        n = self.engine.clear_cards()
        for v in self.views:
            v.clear_all()
        self.status.showMessage(f"已清空 {n} 张卡片", 4000)

    def _on_retry_card(self, cid: str) -> None:
        """卡片上的「重新生成」：重跑一次 TTS（绕过缓存）。

        用户反馈"个别语音会少半句话"。GPT-SoVITS 是采样生成的，重跑有机会
        拿到完整的一份；绕过缓存才有意义，否则会把同一份坏音频拿回来。
        """
        if not self.engine.retry_synth(cid):
            self.status.showMessage("这张卡片已经不在列表里了", 2500)
            return
        self._refresh_card(cid)
        self.status.showMessage("已重新生成这一句，合成好后会自动更新", 4000)

    def _on_toggle(self, cid: str) -> None:
        """行首那一个按钮：正在播就停，否则播。"""
        if self.engine.playing_id == cid:
            self.engine.stop_playback()
            self._refresh_card(cid)
            self.status.showMessage("已停止", 1500)
            return
        ok = self.engine.play_card(cid)
        if not ok:
            self.status.showMessage("还没合成好，合成完会自动播放", 2500)
        self._refresh_card(cid)

    def _stop_all(self) -> None:
        self.engine.stop_playback()
        self._refresh_cards()
        self.status.showMessage("已全部停止", 2000)

    def _shutdown_all(self, include_service: bool = True, show_ui: bool = False) -> dict:
        """完整停机：停识别 -> 停播放 -> 清 TTS 服务 -> 复核。

        关闭程序走的就是这条路。用户反馈"确认关闭后 python.exe 还在"，
        所以这里不只调 stop_service 就完事，而是三重保险清服务
        （进程树 / 按端口 / 按运行时路径），最后**复核端口**并把结果返回。

        `show_ui=True` 时弹一个清理窗口，把每一步实时显示出来 ——
        用户要求"不论是否在识别都要显示清理 TTS 的窗口"，
        这样清理过程是看得见的，出问题也知道卡在哪一步。
        """
        out: dict = {}
        dlg = None
        if show_ui:
            try:
                dlg = _ShutdownDialog(self)
                dlg.show()
                QApplication.processEvents()
            except Exception as e:
                log.warning("清理窗口打不开：%s", e)
                dlg = None

        def step(msg: str) -> None:
            log.info("停机：%s", msg)
            if dlg is not None:
                dlg.add(msg)

        try:
            step("停止播放…")
            self.engine.stop_playback()
            out["playback"] = "stopped"

            step("停止识别引擎…")
            if self.engine.running:
                self.engine.stop()
            out["engine"] = "stopped"

            # **关闭时清空音频缓存**（用户要求）。
            # 缓存是为了"同一句话重复出现时不重复合成"，但跨会话留着只是占磁盘，
            # 而且下次启动语速/声库可能已经变了，旧音频反而容易串味。
            if bool(getattr(self.cfg.cache, "clear_on_exit", True)):
                try:
                    n = self.engine.cache.clear()
                    out["cache_cleared"] = n
                    step(f"清理音频缓存（{n} 条）")
                except Exception as e:
                    log.warning("清理缓存失败：%s", e)
                    step(f"清理缓存失败：{e}")

            if include_service:
                step("停止语音（TTS）服务…")
                out["service"] = self._stop_tts_service(step=step)
            else:
                step("跳过语音服务清理")
        except Exception as e:
            log.exception("停机过程中出错")
            out["error"] = str(e)
            step(f"出错：{e}")

        svc = out.get("service") or {}
        left = svc.get("left") or []
        if include_service:
            if left:
                step(f"⚠ 仍有残留进程：{left}")
            else:
                killed = svc.get("killed") or []
                step(f"✓ 语音服务已停止（清理了 {len(killed)} 个进程，端口已释放）")
        step("✓ 清理完成")

        if dlg is not None:
            dlg.finish(ok=not left)
            # 让窗口短暂停留，用户看得见结果
            try:
                QTimer.singleShot(1200, dlg.accept)
                dlg.exec()
            except Exception:
                pass
        return out

    # ------------------------------------------------------------------ #
    # 引擎
    # ------------------------------------------------------------------ #
    def _toggle_engine(self) -> None:
        if self.engine.running:
            self.engine.stop()
            self.act_toggle.setText("▶ 开始识别")
            self.overlay.set_running(False)      # 悬浮窗按钮同步
            self.status.showMessage("已停止识别", 3000)
            return
        if not self.engine.store.regions:
            QMessageBox.information(self, "还没有区域", "先用工具栏的「框选区域」选一块屏幕区域。")
            return
        # 开始识别前确认 TTS 服务在跑，否则卡片会一片"失败"
        self._ensure_service()
        self.engine.start()
        self.act_toggle.setText("■ 停止识别")
        self.overlay.set_running(True)           # 悬浮窗按钮同步
        self._sync_marks()
        self.status.showMessage("识别中…", 3000)

    def _set_autoread_from_overlay(self, on: bool) -> None:
        """悬浮窗的「自动朗读」按钮被点了：等价于勾左侧的勾选框。"""
        self.ck_autoread.setChecked(bool(on))
        # setChecked 会触发 _on_autoread_changed，那边会写配置并同步回悬浮窗

    def _on_overlay_opacity(self, val: float) -> None:
        """悬浮窗透明度变了：记住它（百分比存整数）。"""
        pct = int(round(max(0.2, min(1.0, val)) * 100))
        if getattr(self.cfg, "overlay_opacity", None) == pct:
            return
        self.cfg.overlay_opacity = pct
        if not getattr(self, "_op_saving", False):
            self._op_saving = True
            QTimer.singleShot(500, self._save_opacity)

    def _save_opacity(self) -> None:
        self._op_saving = False
        try:
            self.cfg.save(default_config_path())
        except Exception as e:
            log.warning("保存悬浮窗透明度失败：%s", e)

    def _toggle_overlay(self, on: bool) -> None:
        if on:
            self.overlay.cards.clear_all()
            for c in self.engine.store.sorted_cards():
                self.overlay.cards.add_or_update(c)
            regs = self.engine.store.regions
            if regs:
                l, t, w, h = regs[0].rect
                self.overlay.move(l + w + 16, t)
            self.overlay.show()
            self.overlay.raise_()
            QTimer.singleShot(0, self.overlay._watch_children)  # noqa: SLF001
        else:
            self.overlay.hide()

    def _on_overlay_closed(self) -> None:
        """用户在悬浮窗上点了 ✕ —— 工具栏的勾选要同步取消。"""
        self.act_overlay.blockSignals(True)
        self.act_overlay.setChecked(False)
        self.act_overlay.blockSignals(False)
        self.status.showMessage("悬浮条已隐藏（工具栏可重新打开）", 4000)

    # ------------------------------------------------------------------ #
    def _on_event(self, e: dict) -> None:
        t = e.get("type")
        if t in ("card_added", "card_rearmed"):
            self._refresh_card(e["card_id"])
        elif t in ("card_ready", "card_failed", "play_started", "play_stopped"):
            cid = e.get("card_id")
            if cid:
                self._refresh_card(cid)
            if t == "card_failed":
                self._note_synth_failure(e.get("error", ""))
        elif t == "play_finished":
            self._refresh_cards()
        elif t == "card_died":
            self._remove_card(e["card_id"])
        elif t == "region_added":
            self._refresh_regions()
        elif t == "region_removed":
            self._refresh_cards()
            self._sync_marks()
        elif t == "cleared":
            for v in self.views:
                v.clear_all()
        elif t == "error":
            log.warning("引擎报错：%s", e)
            self.status.showMessage(f"出错：{e.get('where')} {str(e.get('error', ''))[:90]}", 8000)

    def _note_synth_failure(self, err: str) -> None:
        """合成失败时别只把原因塞进 tooltip —— 用户根本看不到。

        连续失败到一定次数就弹一次说清楚，并给下一步建议。
        """
        self._synth_fails = getattr(self, "_synth_fails", 0) + 1
        self._last_synth_err = err or self._last_synth_err if hasattr(self, "_last_synth_err") else err
        self._last_synth_err = err or getattr(self, "_last_synth_err", "")

        if err:
            self.status.showMessage(f"合成失败：{err[:140]}", 12000)
            log.warning("合成失败：%s", err)

        if self._synth_fails < 2 or getattr(self, "_warned_synth", False):
            return
        self._warned_synth = True

        from .. import runtime as R

        port = R.service_port(self.cfg)
        running = R.service_ready(port)
        box = QMessageBox(self)
        box.setWindowTitle("语音合成失败")
        box.setIcon(QMessageBox.Warning)
        box.setText(f"已经有 {self._synth_fails} 条合成失败。\n\n最后一条的原因：\n{(err or '')[:400]}")
        info = []
        if running:
            info.append(f"TTS 服务在端口 {port} 上是**在跑**的，所以问题可能出在声库上。")
            info.append("可以试试：工具栏「语音运行时…」→ 检查声库，或用下面的按钮重新拉起服务。")
        else:
            info.append(f"TTS 服务（端口 {port}）**没有在跑**——这多半就是原因。")
            info.append("点下面的「重新拉起服务」即可（首次要几十秒加载模型）。")
        info.append("\n常见原因：")
        info.append("  · 没有安装声库（voices\\ 下为空）")
        info.append("  · 声库缺少 tts_infer.yaml（用「语音运行时…」→ 重新生成）")
        info.append("  · 声库权重文件缺失或版本不匹配")
        box.setInformativeText("\n".join(info))
        b_restart = box.addButton("重新拉起服务", QMessageBox.AcceptRole)
        b_detail = box.addButton("查看运行时…", QMessageBox.ActionRole)
        box.addButton("知道了", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is b_restart:
            self._warned_synth = False
            self._synth_fails = 0
            self._ensure_service()
        elif box.clickedButton() is b_detail:
            self._open_runtime_dialog()

    def _set_tts_light(self, state: str, text: str = "") -> None:
        """设置 TTS 服务状态灯。state: ok / starting / down / wait"""
        color = {
            "ok": "#3fb950",
            "starting": "#d29922",
            "down": "#f85149",
            "wait": "#8b949e",
        }.get(state, "#8b949e")
        label = text or {
            "ok": "语音服务 就绪",
            "starting": "语音服务 启动中…",
            "down": "语音服务 未运行",
            "wait": "语音服务 检测中…",
        }.get(state, "")
        self.lbl_tts.setText(f'<span style="color:{color};font-size:16px;">●</span> {label}')

    def _refresh_tts_light(self) -> None:
        """每 1.5 秒探测一次服务端口，刷新状态灯。

        探测只是本地回环 TCP 连接，很便宜；但仍然节流，避免每 500ms 一次。
        """
        now = time.time()
        if now < self._tts_light_next:
            return
        self._tts_light_next = now + 1.5
        try:
            from .. import runtime as R

            port = R.service_port(self.cfg)
            busy = (getattr(self, "_svc_worker", None) is not None
                    and self._svc_worker.isRunning())
            if R.service_ready(port):
                self._set_tts_light("ok")
            elif busy:
                self._set_tts_light("starting")
            else:
                self._set_tts_light("down")
        except Exception:
            self._set_tts_light("wait")

    def _refresh_stats(self) -> None:
        s = self.engine.stats_snapshot()
        cache = self.engine.cache.stats()
        steps = self.engine.sample_steps
        self.lbl_stats.setText(
            f"区域 {s['regions']}   卡片 {s['cards']}"
            f"（待合成 {s['pending']} / 就绪 {s['ready']} / 失败 {s['failed']}）"
            f"   轮询 {s['polls']} 次 / {s['ocr_ms']}ms   采样步数 {steps}"
            f"   缓存命中 {cache['hits']}（{cache['hit_rate'] * 100:.0f}%）"
        )
        self._check_detect_health(s["polls"], s["regions"])
        self._refresh_tts_light()

    def _check_detect_health(self, polls: int, regions: int) -> None:
        """识别健康看门狗：线程死了补一个，长时间不推进就记一笔。

        用户实测："截图朗读之后识别一直卡住，需要手动停止再开始"。
        根因是检测线程可能静默退出而 `running` 还是 True —— 界面看着在识别，
        实际永远不会有新卡片。这里每 0.5 秒检查一次，能自愈。
        """
        try:
            if not self.engine.running or regions <= 0:
                self._detect_seen_polls = -1
                self._detect_stall_ticks = 0
                return
            # 先看线程是不是真的没了 —— 这种可以直接补
            if self.engine.ensure_threads():
                self.status.showMessage("识别/合成线程已自动重启（自愈）", 6000)
                self._detect_stall_ticks = 0
                self._detect_seen_polls = polls
            # 把卡在"合成中"太久的卡片救回来（TTS 服务没起来时会走到这里）
            try:
                if self.engine.recover_stuck_synth(45.0):
                    self.status.showMessage(
                        "有句子合成卡住了，已尝试重新合成。"
                        "若提示失败，多半是 TTS 服务没在运行。", 12000)
            except Exception:
                pass
            # 再看轮询计数有没有推进（线程活着但被卡住）
            if polls != self._detect_seen_polls:
                self._detect_seen_polls = polls
                self._detect_stall_ticks = 0
                return
            self._detect_stall_ticks += 1
            # 6 秒（12 个 tick）没有任何推进
            if self._detect_stall_ticks == 12:
                log.warning("识别轮询 6 秒没有推进（区域 %d 个），可能被 OCR 卡住", regions)
                self.status.showMessage(
                    "识别好像停住了（6 秒没有进展）。已尝试自愈；"
                    "如果一直不动，可以点「停止识别」再「开始识别」。",
                    12000,
                )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def closeEvent(self, ev) -> None:
        """关闭流程：确认 -> **总是显示清理窗口** -> 复核 -> 退出。

        用户要求：
          * 不管当时是否在识别，都要显示清理 TTS 的窗口
          * 并且**再次确认**后台进程是否真的被清掉
        """
        busy = False
        try:
            busy = bool(self.engine.running) or self.engine.store.playing_id is not None
        except Exception:
            busy = False
        # **先拆掉可能存在的全屏框选层**：它是置顶窗口，留着会挡住
        # 关机确认框，甚至让程序看起来"关不掉"（用户实测只能重启电脑）。
        try:
            if getattr(self, "_snap_picker", None) is not None or self._snap_guard.isActive():
                self._snap_hard_cancel()
        except Exception:
            pass
        # **先卸掉全局钩子**：钩子线程有自己的消息泵，
        # 不主动 UnhookWindowsHookEx 的话进程会拖着不退。
        try:
            self._hotkey_timer.stop()
            self._hotkeys.stop()
        except Exception:
            pass
        try:
            self._snap_timer.stop()
        except Exception:
            pass
        # 只有"正在忙"时才问一句，避免每次关都弹
        if busy and not self._confirm_close():
            ev.ignore()
            return

        res = self._shutdown_all(include_service=True, show_ui=True)
        log.info("关闭前完整停机结果：%s", res)

        svc = res.get("service") or {}
        left = svc.get("left") or []
        if left:
            QMessageBox.warning(
                self, "语音服务未完全停止",
                "以下进程仍占用语音端口，可能需要在任务管理器里手动结束：\n\n"
                f"  PID {left}\n\n"
                f"日志：{self.cfg.cache_dir().parent / 'screenreader.log'}",
            )

        for w in (self.region_overlay, self.overlay):
            try:
                w.close()
            except Exception:
                pass
        super().closeEvent(ev)

    def _confirm_close(self) -> bool:
        """识别中或正在朗读时，关闭要先确认。"""
        busy = False
        try:
            busy = bool(self.engine.running)
        except Exception:
            busy = False
        playing = False
        try:
            playing = self.engine.store.playing_id is not None
        except Exception:
            playing = False
        if not (busy or playing):
            return True

        what = []
        if busy:
            what.append("正在识别")
        if playing:
            what.append("正在朗读")
        box = QMessageBox(self)
        box.setWindowTitle("确认关闭")
        box.setIcon(QMessageBox.Question)
        box.setText("读屏台" + "、".join(what) + "，确定要关闭吗？")
        box.setInformativeText(
            "关闭会同时停止语音服务（下次启动需要重新加载，约 10~40 秒）。"
        )
        yes = box.addButton("关闭", QMessageBox.AcceptRole)
        box.addButton("继续用", QMessageBox.RejectRole)
        box.setDefaultButton(box.buttons()[-1])       # 默认停在「继续用」，防误触
        box.exec()
        return box.clickedButton() is yes

    def _stop_tts_service(self, step=None) -> dict:
        """清掉 TTS 服务进程，返回 {'ok':..., 'killed':[...], 'left':[...]}。

        用 `kill_service_processes()` 的三重保险：
          1. 干掉我们记下的进程树（/T 连子进程）
          2. 反复按端口找持有者并干掉
          3. 还不行就按**可执行文件路径**清扫 —— 运行时目录下的 python.exe
             只可能是我们的服务，跟端口/父子关系都无关

        第 3 步是用户反馈"关了还是有 python.exe"之后加的：
        前两步都依赖端口检测，而 `api_v2.py` 派生子进程时父子 PID 对不上，
        光靠端口不一定认得出来。
        """
        # **这个 import 不能少**：R 在别的函数里是各自局部导入的，
        # 这里漏了就会 NameError —— 而关闭流程把它 try/except 吞掉，
        # 表现就是"确认关闭了但 python.exe 还在"，查了很久。
        from .. import runtime as R

        def say(m: str) -> None:
            if step is not None:
                step(m)

        try:
            port = R.service_port(self.cfg)
        except Exception as e:
            log.warning("取服务端口失败：%s", e)
            say(f"取端口失败：{e}")
            return {"ok": True, "killed": [], "left": [], "note": "取端口失败"}

        if not R.service_ready(port):
            log.info("端口 %s 上没有服务，无需清理", port)
            say(f"端口 {port} 上没有服务，无需清理")
            return {"ok": True, "killed": [], "left": []}

        try:
            runtime = R.runtime_root()
        except Exception:
            runtime = None

        before = R.service_pids(port)
        ours = getattr(self, "_tts_pid", None)
        log.info("准备清 TTS 服务：端口 %s 记录 PID=%s 实际监听=%s 运行时=%s",
                 port, ours, before, runtime)
        say(f"端口 {port} 上的服务 PID={before}，正在结束…")

        res = R.kill_service_processes(port, only_pid=ours, runtime=runtime)
        res["ok"] = not res.get("left")
        if res["ok"]:
            log.info("TTS 服务已停止，端口 %s 已释放（清了 %s）", port, res.get("killed"))
            say(f"已结束进程 {res.get('killed')}")
            self.status.showMessage("已停止语音服务", 4000)
        else:
            log.warning("TTS 服务仍有残留：%s", res.get("left"))
            say(f"仍有残留：{res.get('left')}")
            self.status.showMessage(
                f"语音服务可能没停干净（PID {res.get('left')}），可在任务管理器结束", 10000
            )
        return res
