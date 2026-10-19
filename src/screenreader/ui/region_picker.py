"""全屏拖拽框选区域。

交互：
    * 覆盖整个虚拟桌面（多屏也算一张大画布）
    * 按住左键拖拽画矩形，松手即确定一个区域，可连续框多个
    * **正在拖的那块不压暗**（能看清下面的内容，方便对齐）
    * **单击（没有拖动）= 确认完成**，等同 Enter
    * Backspace/右键 撤销上一个
    * Enter 或双击 完成；Esc 取消
    * 已框好的区域会一直显示，并标上序号

为什么必须做成原生覆盖层：浏览器无法覆盖桌面，所以这一块只能用桌面 UI 框架。
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)

Rect = Tuple[int, int, int, int]

# 鼠标移动超过这个像素才算"拖动"，否则当成单击（=确认）
DRAG_THRESHOLD = 6


def virtual_desktop_rect() -> QRect:
    """所有屏幕的并集（可能是负坐标，比如副屏在主屏左侧）。"""
    rect = QRect()
    for s in QGuiApplication.screens():
        rect = rect.united(s.geometry())
    return rect


def monitor_of(global_rect: QRect) -> int:
    """判断矩形落在哪个显示器上（取交集面积最大的那个）。"""
    best, best_area = 0, -1
    screens = QGuiApplication.screens()
    for i, s in enumerate(screens):
        inter = s.geometry().intersected(global_rect)
        area = max(0, inter.width()) * max(0, inter.height())
        if area > best_area:
            best, best_area = i, area
    return best


class RegionPicker(QWidget):
    """全屏半透明覆盖层，拖拽框选。完成后发出 picked(list[(rect, monitor_index)])。"""

    picked = Signal(list)   # [(left, top, w, h), ...]  相对各自显示器
    cancelled = Signal()

    HINT = ("拖拽框选区域（可框多个）    单击=确认    Enter 完成    "
            "右键=撤销上一个（没有则退出）    Backspace 撤销    Esc 取消")

    def __init__(self, parent=None, max_rects: int = 0) -> None:
        """`max_rects`：最多能框几个，0 = 不限。

        「截图朗读」只允许框一个，传 1 —— 再拖一次会**替换**掉上一个，
        方便框不准时直接重来，不用先撤销。
        """
        super().__init__(parent)
        self.max_rects = int(max_rects or 0)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self._press_pos: QPoint | None = None
        self._dragged = False
        self._rects: List[QRect] = []      # 全局坐标
        self._desktop = virtual_desktop_rect()
        self.setGeometry(self._desktop)

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._rects.clear()
        self._origin = self._current = None
        self._press_pos = None
        self._dragged = False
        self.setGeometry(virtual_desktop_rect())
        self.showFullScreen()
        self.raise_()
        self.activateWindow()

    # ------------------------------------------------------------------ #
    def _band(self) -> QRect | None:
        if self._origin is None or self._current is None:
            return None
        return QRect(self._origin, self._current).normalized()

    def _punch(self, p: QPainter, local: QRect) -> None:
        """把一块区域"挖亮"（去掉压暗），下面的内容就看得清了。"""
        p.setCompositionMode(QPainter.CompositionMode_Clear)
        p.fillRect(local, Qt.transparent)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

    def paintEvent(self, ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        to_local = self.geometry().topLeft()

        # 压暗整个桌面
        p.fillRect(self.rect(), QColor(0, 0, 0, 110))

        font = QFont()
        font.setPixelSize(15)
        p.setFont(font)

        # 已确定的区域：挖亮 + 画边框 + 序号
        palette = ["#4ea1ff", "#ffb454", "#7ee787", "#ff7b72", "#d2a8ff", "#79c0ff"]
        for i, r in enumerate(self._rects):
            local = r.translated(-to_local)
            color = QColor(palette[i % len(palette)])
            self._punch(p, local)
            pen = QPen(color, 2)
            p.setPen(pen)
            p.drawRect(local)
            tag = f"{i + 1}"
            p.setPen(QColor("#ffffff"))
            p.fillRect(QRect(local.left(), local.top() - 20, 22, 20), color)
            p.drawText(QRect(local.left(), local.top() - 20, 22, 20), Qt.AlignCenter, tag)

        # 正在拖的橡皮筋：**同样挖亮**，这样能看清正在框的内容
        band = self._band()
        if band is not None and band.width() > 1 and band.height() > 1:
            local = band.translated(-to_local)
            self._punch(p, local)
            p.setPen(QPen(QColor("#4ea1ff"), 2, Qt.DashLine))
            p.drawRect(local)
            # 尺寸标注放在框外，免得挡住内容
            label = f"{band.width()} × {band.height()}"
            lw = p.fontMetrics().horizontalAdvance(label) + 12
            ly = local.top() - 22 if local.top() > 24 else local.bottom() + 2
            p.fillRect(QRect(local.left(), ly, lw, 20), QColor(20, 20, 24, 210))
            p.setPen(QColor("#e8e8ea"))
            p.drawText(QRect(local.left(), ly, lw, 20), Qt.AlignCenter, label)

        # 顶部提示条
        hint_h = 40
        p.fillRect(QRect(0, 0, self.width(), hint_h), QColor(20, 20, 24, 220))
        p.setPen(QColor("#e8e8ea"))
        p.drawText(QRect(16, 0, self.width() - 32, hint_h), Qt.AlignVCenter | Qt.AlignLeft, self.HINT)
        p.drawText(
            QRect(16, 0, self.width() - 32, hint_h),
            Qt.AlignVCenter | Qt.AlignRight,
            f"已选 {len(self._rects)} 个",
        )
        p.end()

    # ------------------------------------------------------------------ #
    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            self._origin = ev.position().toPoint()
            self._current = self._origin
            self._press_pos = self._origin
            self._dragged = False
            self.update()
        elif ev.button() == Qt.RightButton:
            # **右键 = 撤销上一次框选；没有可撤销的就退出框选**（用户要求）。
            # 以前无论有没有框过都直接退出，于是框了三个想退掉一个时，
            # 得反复右键好几次才退出，手感很怪。
            if self._rects:
                self._undo()
            else:
                self.hide()
                self.cancelled.emit()
                self.close()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        if self._origin is not None:
            self._current = ev.position().toPoint()
            # 只有真的移动了才算拖动；否则保持"单击"语义
            if self._press_pos is not None and not self._dragged:
                d = (self._current - self._press_pos).manhattanLength()
                if d >= DRAG_THRESHOLD:
                    self._dragged = True
            self.update()
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton and self._origin is not None:
            band = self._band()
            moved = self._dragged
            self._origin = self._current = None
            self._press_pos = None
            self._dragged = False

            if not moved:
                # **单击 = 确认**（等同 Enter）。手抖一下不算拖动。
                self.update()
                self._finish()
                return
            if band is not None and band.width() >= 8 and band.height() >= 8:
                # 只允许框一个时，新的拖拽**替换**旧的（方便框不准直接重画）
                if self.max_rects and len(self._rects) >= self.max_rects:
                    self._rects.clear()
                self._rects.append(band.translated(self.geometry().topLeft()))
                log.info("框选区域 %d: %s", len(self._rects), band)
            self.update()
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev) -> None:
        self._finish()

    # ------------------------------------------------------------------ #
    def _undo(self) -> None:
        if self._rects:
            self._rects.pop()
            self.update()

    def keyPressEvent(self, ev) -> None:
        k = ev.key()
        if k in (Qt.Key_Return, Qt.Key_Enter):
            self._finish()
        elif k == Qt.Key_Escape:
            self.hide()
            self.cancelled.emit()
            self.close()
        elif k in (Qt.Key_Backspace, Qt.Key_Delete):
            self._undo()
        else:
            super().keyPressEvent(ev)

    def _finish(self) -> None:
        out: List[Tuple[Rect, int]] = []
        for r in self._rects:
            mon = monitor_of(r)
            geo = QGuiApplication.screens()[mon].geometry()
            local = r.translated(-geo.topLeft())
            out.append(
                ((int(local.left()), int(local.top()), int(local.width()), int(local.height())), mon)
            )
        self.hide()
        self.picked.emit(out)
        self.close()
