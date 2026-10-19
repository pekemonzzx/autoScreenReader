"""「截图朗读」用的**冻结帧**框选层。

和普通的 `RegionPicker` 的本质区别：
    普通框选  —— 半透明盖在**实时桌面**上，框选过程中画面还在动
    冻结框选  —— 先把当前屏幕拍成一张静态图，让用户**在这张图上**裁切，
                  所以"框到的"就是"读到的"，画面绝不会变

为什么必须冻结：游戏/视频/滚动列表在框选的那一两秒里会变，用户框完之后程序
再去截一次屏，拿到的可能已经不是同一帧了 —— 实测表现就是"框了却没读对"，
或者整段模式因为内容一直在变、永远等不到"稳定"而降级成不读。

只覆盖**鼠标所在的那一个屏幕**（用户的选择）：多屏拼接要处理跨屏 DPI 和
负坐标，收益不大。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

HINT = ("画面已冻结 —— 拖拽框选要朗读的那一段    单击=确定    "
        "Enter 完成    右键 / Esc 取消")


class SnapPicker(QWidget):
    """在冻结的屏幕截图上框选一块。

    信号：
        picked(QRect, QImage)   所选矩形（**相对该屏幕**）+ 裁好的图
        cancelled()             右键 / Esc 取消
    """

    picked = Signal(QRect, QImage)
    cancelled = Signal()

    def __init__(self, frozen: QPixmap, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
            | Qt.NoDropShadowWindowHint
        )
        # **和 RegionPicker 保持一致**。差别就在这里：普通框选层有
        # WA_TranslucentBackground，截图朗读层没有 —— Windows 对"不透明
        # 全屏无边框窗口"的合成路径不同，配合每次鼠标移动的整屏重绘，
        # 会让系统光标被反复擦掉又来不及重画，表现就是"有时看不到指针"。
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

        self._frozen = frozen
        # 屏幕截图的逻辑尺寸（可能和像素尺寸因 DPI 不同而不等）
        dpr = float(frozen.devicePixelRatio() or 1.0)
        self._dpr = dpr
        self._logical_w = int(round(frozen.width() / dpr))
        self._logical_h = int(round(frozen.height() / dpr))

        # **预缩放 + 预压暗两份缓存**。
        # 原来每次 paintEvent 都 `drawPixmap(self.rect(), self._frozen)` ——
        # 那是**带缩放**的整屏绘制（2560x1440），而鼠标每动一下就要重绘一次。
        # 重绘太慢，光标就会闪/不显示。现在 paintEvent 只做两次 1:1 贴图。
        self._view: Optional[QPixmap] = None
        self._dim: Optional[QPixmap] = None

        self._origin: Optional[QPoint] = None
        self._current: Optional[QPoint] = None
        self._press_pos: Optional[QPoint] = None
        self._dragged = False
        self._rect: Optional[QRect] = None      # 局部坐标
        # 自己画准星：即使系统光标因为别的原因不显示，用户也永远看得见
        self._cursor_pos: Optional[QPoint] = None

    # ------------------------------------------------------------------ #
    def _build_cache(self) -> None:
        """按当前窗口大小生成"原图"和"压暗图"两份缓存。"""
        w = max(1, self.width())
        h = max(1, self.height())
        self._view = self._frozen.scaled(
            w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        dim = QPixmap(self._view)
        dp = QPainter(dim)
        dp.fillRect(0, 0, w, h, QColor(0, 0, 0, 120))
        dp.end()
        self._dim = dim

    def start(self, screen_rect: QRect) -> None:
        self.setGeometry(screen_rect)
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self.setFocus()
        # showFullScreen() 之后尺寸才最终确定，缓存要这时建
        self._build_cache()
        # 光标锁成十字准星，退出时恢复（配合自绘准星双保险）
        try:
            QApplication.setOverrideCursor(Qt.CrossCursor)
        except Exception:
            pass

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if self._frozen is not None and not self._frozen.isNull():
            self._build_cache()

    # ------------------------------------------------------------------ #
    def _band(self) -> Optional[QRect]:
        if self._origin is None or self._current is None:
            return None
        return QRect(self._origin, self._current).normalized()

    def paintEvent(self, ev) -> None:
        p = QPainter(self)

        # 1) 压暗后的整屏图（**1:1 贴图，不缩放**）+ 2) 框内贴回亮的原图。
        #    这是"冻结"的关键：画的是抓下来的那张图，不是活的桌面。
        if self._dim is not None:
            p.drawPixmap(0, 0, self._dim)
        elif self._frozen is not None:
            p.drawPixmap(self.rect(), self._frozen)

        font = QFont()
        font.setPixelSize(15)
        p.setFont(font)

        def brighten(local: QRect) -> None:
            """把这一块**重新画出不压暗的原图**。

            注意：不能用 CompositionMode_Clear —— 那会把窗口挖透明、
            露出底下活的桌面，就又不"冻结"了。
            现在从预缩放缓存里取，是 1:1 贴图，很快。
            """
            if local.width() < 1 or local.height() < 1:
                return
            if self._view is not None:
                p.drawPixmap(local, self._view, local)
            else:
                src = QRect(
                    int(local.left() * self._dpr),
                    int(local.top() * self._dpr),
                    max(1, int(local.width() * self._dpr)),
                    max(1, int(local.height() * self._dpr)),
                )
                p.drawPixmap(local, self._frozen, src)

        # 3) 已确定的框：挖亮 + 边框
        if self._rect is not None:
            brighten(self._rect)
            p.setPen(QPen(QColor("#4ea1ff"), 2))
            p.drawRect(self._rect)

        # 4) 正在拖的橡皮筋
        band = self._band()
        if band is not None and band.width() > 1 and band.height() > 1:
            brighten(band)
            p.setPen(QPen(QColor("#4ea1ff"), 2, Qt.DashLine))
            p.drawRect(band)
            label = f"{band.width()} × {band.height()}"
            lw = p.fontMetrics().horizontalAdvance(label) + 12
            ly = band.top() - 22 if band.top() > 24 else band.bottom() + 2
            p.fillRect(QRect(band.left(), ly, lw, 20), QColor(20, 20, 24, 210))
            p.setPen(QColor("#e8e8ea"))
            p.drawText(QRect(band.left(), ly, lw, 20), Qt.AlignCenter, label)

        # 5) 顶部提示
        hint_h = 40
        p.fillRect(QRect(0, 0, self.width(), hint_h), QColor(20, 20, 24, 225))
        p.setPen(QColor("#e8e8ea"))
        p.drawText(QRect(16, 0, self.width() - 32, hint_h),
                   Qt.AlignVCenter | Qt.AlignLeft, HINT)
        p.drawText(QRect(16, 0, self.width() - 32, hint_h),
                   Qt.AlignVCenter | Qt.AlignRight,
                   "已框选" if self._rect is not None else "未框选")

        # （准星线已按用户要求去掉 —— 看着太乱。
        #   光标可见性现在靠：WA_TranslucentBackground + 1:1 快速贴图
        #   + overrideCursor 锁成十字，已经不需要自绘参考线了。）
        p.end()

    # ------------------------------------------------------------------ #
    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton:
            self._origin = self._current = ev.position().toPoint()
            self._press_pos = self._origin
            self._dragged = False
            self.update()
        elif ev.button() == Qt.RightButton:
            self._cancel()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        # **不管有没有在拖动，都要记下光标位置**（画准星用）。
        # 用户反馈"鼠标移出框选范围时有时看不到指针" —— 自绘准星就是兜底。
        self._cursor_pos = ev.position().toPoint()
        if self._origin is not None:
            self._current = self._cursor_pos
            if (self._press_pos is not None
                    and (self._current - self._press_pos).manhattanLength() > 4):
                self._dragged = True
        self.update()
        super().mouseMoveEvent(ev)

    def enterEvent(self, ev) -> None:
        # 重新进窗口时把光标再声明一次，防止被别的控件改掉
        self.setCursor(Qt.CrossCursor)
        try:
            QApplication.setOverrideCursor(Qt.CrossCursor)
        except Exception:
            pass
        super().enterEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() == Qt.LeftButton and self._origin is not None:
            band = self._band()
            moved = self._dragged
            self._origin = self._current = None
            self._press_pos = None
            self._dragged = False

            if not moved:
                # **单击 = 确定**（和普通框选一致，手抖不算拖）
                self.update()
                self._finish()
                return
            if band is not None and band.width() >= 8 and band.height() >= 8:
                # 只允许一个框：新的拖拽**替换**旧的
                self._rect = band
            self.update()
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev) -> None:
        self._finish()

    def keyPressEvent(self, ev) -> None:
        k = ev.key()
        if k in (Qt.Key_Return, Qt.Key_Enter):
            self._finish()
        elif k == Qt.Key_Escape:
            self._cancel()
        elif k in (Qt.Key_Backspace, Qt.Key_Delete):
            self._rect = None
            self.update()
        else:
            super().keyPressEvent(ev)

    # ------------------------------------------------------------------ #
    def _cancel(self) -> None:
        """取消。**无论发信号时出什么事，窗口一定要关掉。**

        这是全屏置顶窗口 —— 关不掉就等于把用户困在桌面上，
        实测严重到只能重启电脑。所以 emit 用 try/finally 兜住。
        """
        try:
            self.hide()
        except Exception:
            pass
        try:
            self.cancelled.emit()
        except Exception:
            pass
        finally:
            self._force_close()

    def _force_close(self) -> None:
        # **一定要恢复光标**：setOverrideCursor 不还原会把整个程序的光标
        # 卡在十字准星上（退出框选后连主窗口都是准星）。
        try:
            QApplication.restoreOverrideCursor()
        except Exception:
            pass
        try:
            self.releaseKeyboard()
        except Exception:
            pass
        try:
            self.close()
        except Exception:
            pass
        try:
            self.deleteLater()
        except Exception:
            pass

    def closeEvent(self, ev) -> None:
        """任何关闭路径都要放开键盘抓取和光标覆盖。"""
        try:
            QApplication.restoreOverrideCursor()
        except Exception:
            pass
        try:
            self.releaseKeyboard()
        except Exception:
            pass
        super().closeEvent(ev)

    def _finish(self) -> None:
        rect = self._rect
        if rect is None or rect.width() < 8 or rect.height() < 8:
            # 没框出东西就当取消，别送一个空区域去 OCR
            self._cancel()
            return
        # 裁图：局部逻辑坐标 -> 像素坐标
        src = QRect(
            int(rect.left() * self._dpr),
            int(rect.top() * self._dpr),
            max(1, int(rect.width() * self._dpr)),
            max(1, int(rect.height() * self._dpr)),
        )
        try:
            img = self._frozen.toImage().copy(src)
        except Exception:
            self._cancel()
            return
        try:
            self.hide()
        except Exception:
            pass
        try:
            self.picked.emit(QRect(rect), img)
        except Exception:
            pass
        finally:
            self._force_close()


# --------------------------------------------------------------------------- #
def qimage_to_rgb_array(img: QImage) -> np.ndarray:
    """QImage -> (H, W, 3) uint8 RGB 数组，给 OCR 用。

    两个必须注意的点，都是实测踩出来的：

    1. **要按 `bytesPerLine` 取行**：QImage 每行可能按 4 字节对齐，
       直接用 `bits()` 整体 reshape 会在有行内填充时错位，图是花的。

    2. **必须真的复制一份**：`np.frombuffer(conv.constBits())` 拿到的是
       **视图**，而 `np.ascontiguousarray()` 在数组本来就连续时**不复制** ——
       于是返回的数组指向 `conv` 这块 QImage 的内存。函数一返回，`conv`
       被析构，数组就指向已释放的内存，OCR 读到的是垃圾像素
       （表现就是"识别出来全是乱的"）。所以最后一定要 `.copy()`。
    """
    if img.isNull():
        return np.zeros((1, 1, 3), dtype=np.uint8)
    conv = img.convertToFormat(QImage.Format_RGB888)
    h, w = conv.height(), conv.width()
    bpl = conv.bytesPerLine()
    mv = conv.constBits()
    buf = np.frombuffer(mv, dtype=np.uint8)
    need = h * bpl
    if buf.size < need:
        buf = np.concatenate([buf, np.zeros(need - buf.size, dtype=np.uint8)])
    arr = buf[:need].reshape(h, bpl)[:, : w * 3].reshape(h, w, 3)
    # **copy() 不能省**（见上面第 2 点）
    return arr.copy()
