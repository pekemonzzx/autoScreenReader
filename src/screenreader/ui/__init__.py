"""PySide6 界面层。

只做「把 board.py / worker.py 的状态画出来，把用户操作转回去」，
所有生灭规则都在 board.py，所有线程都在 worker.py。
"""

from .app import main
from .cards import CardList, CardRow
from .region_overlay import RegionOverlay
from .region_picker import RegionPicker, virtual_desktop_rect

__all__ = [
    "main",
    "CardList",
    "CardRow",
    "RegionOverlay",
    "RegionPicker",
    "virtual_desktop_rect",
]
