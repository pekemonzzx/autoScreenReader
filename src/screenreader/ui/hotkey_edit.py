"""快捷键捕获框：点一下，然后**直接按**组合键或鼠标侧键。

为什么要单独写一个控件（原来的做法错在哪）：
    原来对话框用「两个按钮 + 一个 QLabel」设置侧键，保存时靠
    **读标签文字**判断用户选了哪个：

        if lbl.text().startswith("当前：mouse4"): spec = "mouse4"

    于是打开对话框时只要当前值是 mouse4，标签就已经是"当前：mouse4"，
    之后无论怎么按键盘，保存都会退回 mouse4 —— **改不回来了**。
    本质是"把显示文字当成了状态"。

现在只有**一个状态变量** `self.spec`，键盘和鼠标都写它，
显示只是它的投影，不存在两条路径打架。

两种输入共用同一个交互（用户要求的"点调整后直接按"）：
    键盘     keyPressEvent  ->  Ctrl+Alt+S
    鼠标侧键 mousePressEvent ->  Qt.BackButton    = 侧键1（后退）
                                 Qt.ForwardButton = 侧键2（前进）

注意：侧键只能在**这个框有焦点时**捕获（Qt 拿不到别的程序的鼠标事件）。
但这正是设置时的交互，没问题 —— 运行时触发仍然是全局钩子。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QLineEdit

from ..hotkey import parse_binding

# 只按下修饰键时不该当成长度 1 的"按键"，要等真正的那个键
_MODIFIER_KEYS = {
    Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta,
    Qt.Key_AltGr, Qt.Key_CapsLock,
}


def _iv(x) -> int:
    """PySide6 的枚举不能直接 int()，要走 .value。"""
    return x.value if hasattr(x, "value") else int(x)


class HotkeyCapture(QLineEdit):
    """点一下然后按键。`spec` 是唯一的状态。"""

    def __init__(self, spec: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText("点这里，然后按组合键或鼠标侧键")
        self.setToolTip(
            "点一下这个框，然后直接按：\n"
            "  · 键盘组合键 —— 例如 Ctrl+Alt+S\n"
            "  · 鼠标侧键   —— 直接按鼠标侧面的前进/后退键\n"
            "按 Esc 清空"
        )
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumWidth(220)
        self._spec = ""
        self.spec = spec or ""

    # ------------------------------------------------------------------ #
    @property
    def spec(self) -> str:
        return self._spec

    @spec.setter
    def spec(self, value: str) -> None:
        v = (value or "").strip()
        # 存进来的值先校验一遍，看不懂的一律当没设置
        self._spec = v if (v and parse_binding(v) is not None) else (v if not v else "")
        self.setText(self._spec)

    # ------------------------------------------------------------------ #
    def _set_spec(self, spec: str) -> None:
        self._spec = spec
        self.setText(spec)

    def keyPressEvent(self, ev) -> None:
        k = ev.key()
        if k == Qt.Key_Escape:
            self._set_spec("")
            ev.accept()
            return
        if k in _MODIFIER_KEYS or k == Qt.Key_unknown:
            # 只按了修饰键：等真正的按键，别把 "Ctrl" 本身存成快捷键
            ev.accept()
            return

        mods = ev.modifiers()
        try:
            # 组合键 -> "Ctrl+Alt+S" -> 归一化成 "ctrl+alt+s"
            # 注意枚举必须走 .value：PySide6 的 KeyboardModifier/Key 不能 int()
            text = QKeySequence(_iv(mods) | _iv(k)).toString()
        except Exception:
            text = ""
        spec = (text or "").lower().replace(" ", "").replace("\u00a0", "")
        if spec and parse_binding(spec) is not None:
            self._set_spec(spec)
        ev.accept()

    def mousePressEvent(self, ev) -> None:
        b = ev.button()
        if b == Qt.BackButton:
            self._set_spec("mouse4")
            ev.accept()
            return
        if b == Qt.ForwardButton:
            self._set_spec("mouse5")
            ev.accept()
            return
        # 左键点进来要给焦点，方便接着按键
        self.setFocus(Qt.MouseFocusReason)
        super().mousePressEvent(ev)
