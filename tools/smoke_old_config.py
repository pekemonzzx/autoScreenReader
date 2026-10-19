"""验证：老配置（没有 audio.volume 键）也能正常开界面。"""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

td = Path(tempfile.mkdtemp())
cfgfile = td / "config.json"
# 故意做成老格式：没有 volume 键
cfgfile.write_text(json.dumps({"audio": {"device": "", "latency": "low"}}), encoding="utf-8")
os.environ["SCREENREADER_CONFIG"] = str(cfgfile)

from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication([])
from screenreader.config import AppConfig  # noqa: E402
from screenreader.ui.main_window import MainWindow, _safe_int  # noqa: E402

c = AppConfig.load()
print("  载入后 volume =", getattr(c.audio, "volume", None))
w = MainWindow(c)
print("  滑动条值:", w.sld_volume.value(),
      "范围:", w.sld_volume.minimum(), "-", w.sld_volume.maximum())
print("  拖动条常显:", not w.overlay.header.isHidden())
print("  全部按钮可用:", w.overlay.btn_play_all.isEnabled())
print("  _safe_int(None,100,0,300) =", _safe_int(None, 100, 0, 300))
print("  _safe_int('乱',100,0,300) =", _safe_int("乱", 100, 0, 300))
print("  _safe_int(9999,100,0,300) =", _safe_int(9999, 100, 0, 300))
w.close()
app.processEvents()
print("  OK")
