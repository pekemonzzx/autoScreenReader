"""卡片列表：每句一行，行首一个图标按钮切换播放/停止。

行结构：
    [区域色条]  [▶/⏹]  文本（可选中、自动换行）  [状态]

按钮规则（只有一个按钮，按状态切换图标）：
    合成中…  ▶ 可点（会等合成好自动播）
    就绪     ▶ 可点
    播放中   ⏹ 可点
    失败     ↻ 可点（重试）

用「按卡片 id 增量更新」而不是整表重建：既保留滚动位置，也避免每轮闪一下。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..board import Card

log = logging.getLogger(__name__)

STATE_TEXT = {
    "pending": "合成中…",
    "ready": "就绪",
    "playing": "播放中",
    "failed": "失败",
}
STATE_COLOR = {
    "pending": "#8b949e",
    "ready": "#3fb950",
    "playing": "#58a6ff",
    "failed": "#f85149",
}


class CardRow(QFrame):
    """一行卡片。行首播放/停止按钮，行尾一个「重新生成」按钮。"""

    toggle_clicked = Signal(str)
    retry_clicked = Signal(str)

    def __init__(self, card: Card, color: str = "#4ea1ff", parent=None) -> None:
        super().__init__(parent)
        self.card_id = card.id
        self.setObjectName("cardRow")
        self.setFrameShape(QFrame.StyledPanel)

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 5, 8, 5)
        root.setSpacing(8)

        self.bar = QFrame()
        self.bar.setFixedWidth(3)
        root.addWidget(self.bar)

        self.btn = QPushButton("▶")
        self.btn.setFixedSize(32, 28)
        self.btn.clicked.connect(lambda: self.toggle_clicked.emit(self.card_id))
        root.addWidget(self.btn)

        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        root.addWidget(self.text, 1)

        # **重新生成**：对同一段文字重跑一次 TTS（绕过缓存）。
        #
        # 用户反馈"个别语音会少半句话"，怀疑是合成偶发截断。GPT-SoVITS 是采样
        # 生成的，重跑有机会拿到完整的一份；绕过缓存才有意义（否则拿回同一份）。
        self.btn_retry = QPushButton("⟳")
        self.btn_retry.setFixedSize(26, 28)
        self.btn_retry.setCursor(Qt.PointingHandCursor)
        self.btn_retry.setToolTip("重新生成这句的语音（会绕过缓存，重新合成一次）")
        self.btn_retry.setStyleSheet(
            "QPushButton{background:#21262d;border:1px solid #30363d;"
            "border-radius:4px;color:#8b949e;}"
            "QPushButton:hover{background:#30363d;color:#e6edf3;}"
        )
        self.btn_retry.clicked.connect(lambda: self.retry_clicked.emit(self.card_id))
        root.addWidget(self.btn_retry)

        self.state = QLabel()
        f = QFont()
        f.setPixelSize(12)
        self.state.setFont(f)
        self.state.setFixedWidth(58)
        self.state.setAlignment(Qt.AlignCenter)
        root.addWidget(self.state)

        self._color = color
        self.bar.setStyleSheet(f"background:{color}; border-radius:1px;")
        self.update_from(card)

    def set_color(self, color: str) -> None:
        self._color = color
        self.bar.setStyleSheet(f"background:{color}; border-radius:1px;")

    def update_from(self, c: Card) -> None:
        self.text.setText(c.display(120))
        self.text.setToolTip(c.text)

        playing = c.play_state == "playing"
        state = "playing" if playing else c.audio_state
        label = STATE_TEXT.get(state, state)
        if c.from_cache and state == "ready":
            label = "就绪·缓存"
        self.state.setText(label)
        self.state.setStyleSheet(f"color:{STATE_COLOR.get(state, '#8b949e')};")

        # 只有一个按钮：图标随状态变
        if playing:
            self.btn.setText("⏹")
            self.btn.setToolTip("停止")
            self.btn.setEnabled(True)
        elif c.audio_state == "failed":
            self.btn.setText("↻")
            self.btn.setToolTip(f"重试（{c.audio_error[:60]}）")
            self.btn.setEnabled(True)
        elif c.audio_state == "ready":
            self.btn.setText("▶")
            self.btn.setToolTip("播放这一句")
            self.btn.setEnabled(True)
        else:  # pending / none
            self.btn.setText("▶")
            self.btn.setToolTip("还没合成好，点了会等合成完自动播放")
            self.btn.setEnabled(True)

        self.setStyleSheet(
            "#cardRow { border: 1px solid %s; border-radius: 6px; background: %s; }"
            % ("#1f6feb" if playing else "#21262d",
               "#132033" if playing else "transparent")
        )


class CardList(QScrollArea):
    toggle_requested = Signal(str)
    retry_requested = Signal(str)

    def __init__(self, parent=None, compact: bool = False) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._host = QWidget()
        self._layout = QVBoxLayout(self._host)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(4)
        self._layout.addStretch(1)
        self.setWidget(self._host)
        self._rows: Dict[str, CardRow] = {}
        self._colors: Dict[str, str] = {}
        self._empty = QLabel(
            "等待识别…" if compact else "识别到文字后会在这里按句列出，行首按钮可单独播放。"
        )
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setStyleSheet("color:#8b949e; padding:24px;")
        self._layout.insertWidget(0, self._empty)

    def set_region_colors(self, colors: Dict[str, str]) -> None:
        self._colors = dict(colors)
        for row in self._rows.values():
            row.set_color(colors.get(row.card_id, "#4ea1ff"))

    def add_or_update(self, c: Card) -> None:
        row = self._rows.get(c.id)
        if row is None:
            row = CardRow(c, self._colors.get(c.region_id, "#4ea1ff"))
            row.toggle_clicked.connect(self.toggle_requested)
            row.retry_clicked.connect(self.retry_requested)
            self._rows[c.id] = row
            self._layout.insertWidget(self._layout.count() - 1, row)
            if not self._empty.isHidden():
                self._empty.hide()
        else:
            row.set_color(self._colors.get(c.region_id, "#4ea1ff"))
        row.update_from(c)

    def remove(self, cid: str) -> None:
        row = self._rows.pop(cid, None)
        if row is not None:
            self._layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        # 注意用 isHidden() 而不是 isVisible()：
        # 父窗口还没 show 的时候 isVisible() 恒为 False，判断会失真。
        if not self._rows and self._empty.isHidden():
            self._empty.show()

    def clear_all(self) -> None:
        for cid in list(self._rows):
            self.remove(cid)

    @property
    def count(self) -> int:
        return len(self._rows)

    @property
    def showing_empty_hint(self) -> bool:
        return not self._empty.isHidden()

