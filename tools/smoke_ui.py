"""最终冒烟：界面能起来、两种模式可切换。"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["SCREENREADER_CONFIG"] = r"D:\work\autoScreenReader\data\_smoke_cfg.json"
sys.path.insert(0, "src")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication([])
from screenreader.config import AppConfig  # noqa: E402
from screenreader.ui.main_window import MainWindow  # noqa: E402

cfg = AppConfig.load()
cfg.apply_voice()
w = MainWindow(cfg)
print("  声库:", cfg.tts.voice)
print("  模式下拉:", [w.cmb_mode.itemData(i) for i in range(w.cmb_mode.count())])

r = w.engine.add_region("正文区", (0, 0, 600, 400), mode="paragraph")
w._refresh_regions()
w._save_regions()
row = next(i for i in range(w.region_list.count())
           if w.region_list.item(i).data(Qt.UserRole) == r.id)
w.region_list.setCurrentRow(row)
app.processEvents()
print("  下拉显示:", w.cmb_mode.currentData())
print("  区域标签:", w.region_list.item(row).text())
print("  提示:", w.lbl_mode_hint.text())
print("  区域已落盘:", bool(cfg.load_regions()))
w.close()
print("  冒烟通过 ✅")
