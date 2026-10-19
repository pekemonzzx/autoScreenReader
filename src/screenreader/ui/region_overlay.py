"""区域角标覆盖层：把框选的区域画在屏幕上，随时看得见边界。

要点：
  * **点击穿透**（`WindowTransparentForInput` + `WA_TransparentForMouseEvents`），
    绝不挡住下面的游戏或聊天窗口。
  * 覆盖整个虚拟桌面（多屏并集），各区域按自己显示器的原点换算。
  * 画四个角的 L 形角标 + 序号徽标 + 区域名，中间完全透明——
    这样既能看清范围，又几乎不遮挡内容。
  * 可以在工具栏一键开关。
"""

from __future__ import annotations

import logging
from typing import Dict, List

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..board import Region
from .region_picker import virtual_desktop_rect

log = logging.getLogger(__name__)


class RegionOverlay(QWidget):
    CORNER = 22          # 角标臂长
    THICK = 3            # 角标线宽
    BADGE = 20           # 序号徽标尺寸

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowTransparentForInput
            | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self._regions: List[Region] = []
        self.setGeometry(virtual_desktop_rect())

    # ------------------------------------------------------------------ #
    def set_regions(self, regions: List[Region]) -> None:
        self._regions = list(regions)
        self.setGeometry(virtual_desktop_rect())
        if self.isVisible():
            self.update()

    def show_overlay(self) -> None:
        self.setGeometry(virtual_desktop_rect())
        self.show()
        self.raise_()

    # ------------------------------------------------------------------ #
    def _local_rect(self, r: Region) -> QRect:
        """把「相对显示器」的坐标换算成「相对整个虚拟桌面」的屏幕坐标。"""
        screens = QGuiApplication.screens()
        origin = screens[0].geometry().topLeft() if screens else virtual_desktop_rect().topLeft()
        if 0 <= r.monitor_index < len(screens):
            origin = screens[r.monitor_index].geometry().topLeft()
        l, t, w, h = r.rect
        return QRect(origin.x() + l, origin.y() + t, w, h)

    def paintEvent(self, ev) -> None:
        if not self._regions:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        to_local = self.geometry().topLeft()

        font = QFont()
        font.setPixelSize(12)
        font.setBold(True)
        p.setFont(font)

        for i, r in enumerate(self._regions):
            rect = self._local_rect(r).translated(-to_local)
            color = QColor(r.color)
            if not r.enabled:
                color.setAlpha(90)
            pen = QPen(color, self.THICK)
            pen.setCapStyle(Qt.FlatCap)
            p.setPen(pen)

            c = self.CORNER
            L, T, R, B = rect.left(), rect.top(), rect.right(), rect.bottom()
            # 四个角的 L 形角标（只画角，中间留空，尽量不遮内容）
            for (x, y, dx, dy) in (
                (L, T, 1, 1),
                (R, T, -1, 1),
                (L, B, 1, -1),
                (R, B, -1, -1),
            ):
                p.drawLine(x, y, x + dx * c, y)
                p.drawLine(x, y, x, y + dy * c)

            # 序号徽标
            badge = QRect(L, T - self.BADGE - 3, self.BADGE, self.BADGE)
            p.fillRect(badge, color)
            p.setPen(QColor("#0d1117"))
            p.drawText(badge, Qt.AlignCenter, str(i + 1))
            p.setPen(pen)

            # 区域名（徽标右边）
            name = r.name + ("" if r.enabled else "（已停用）")
            tw = p.fontMetrics().horizontalAdvance(name) + 12
            tag = QRect(L + self.BADGE + 4, T - self.BADGE - 3, tw, self.BADGE)
            bg = QColor(13, 17, 23, 210)
            p.fillRect(tag, bg)
            p.setPen(QColor("#e6edf3"))
            p.drawText(tag.adjusted(6, 0, -6, 0), Qt.AlignVCenter | Qt.AlignLeft, name)
            p.setPen(pen)

        p.end()
