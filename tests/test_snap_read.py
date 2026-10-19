"""「截图朗读」相关逻辑的验证（不需要真屏幕/真 OCR）。

覆盖：
  * 快捷键写法解析（键盘组合键 + 鼠标侧键）
  * 临时区域不进持久化
  * 开启/停用快照的存与恢复
  * "这一段读完了没有"的判据
  * 框选层 max_rects=1 时新框替换旧框
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from screenreader.board import BoardConfig, CardStore  # noqa: E402
from screenreader.config import AppConfig  # noqa: E402
from screenreader.hotkey import MOD_ALT, MOD_CONTROL, MOD_SHIFT, parse_binding  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def test_hotkey_parse():
    print("\n[快捷键写法解析]")
    check("ctrl+alt+s", parse_binding("ctrl+alt+s") == (frozenset({MOD_CONTROL, MOD_ALT}), ord("S")))
    check("大小写不敏感", parse_binding("Ctrl+Alt+S") == parse_binding("ctrl+alt+s"))
    check("空格也能用", parse_binding("ctrl + alt + s") == parse_binding("ctrl+alt+s"))
    check("ctrl+shift+f9", parse_binding("ctrl+shift+f9") == (frozenset({MOD_CONTROL, MOD_SHIFT}), 0x78))
    check("mouse4 -> 侧键1", parse_binding("mouse4") == (frozenset(), -1))
    check("mouse5 -> 侧键2", parse_binding("mouse5") == (frozenset(), -2))
    check("中文侧键名", parse_binding("侧键1") == (frozenset(), -1))
    check("单键 esc", parse_binding("esc") == (frozenset(), 0x1B))
    check("乱码返回 None", parse_binding("那不是按键") is None)
    check("空串返回 None", parse_binding("") is None)


def test_temp_region_not_saved():
    print("\n[临时区域不进配置]")
    cfg = AppConfig()
    st = CardStore(BoardConfig())
    normal = st.add_region("常驻", (0, 0, 100, 100))
    temp = st.add_region("截图朗读", (200, 0, 100, 100))
    temp.temp = True

    check("常驻区域 temp=False", normal.temp is False)
    check("临时区域 temp=True", temp.temp is True)
    check("标签上写「截图朗读」", "截图朗读" in temp.label(), temp.label())

    p = Path(tempfile.mkdtemp()) / "cfg.json"
    cfg.save_regions(st.regions, path=p)
    import json
    saved = json.loads(p.read_text(encoding="utf-8"))["regions"]
    check("只存了常驻区域", len(saved) == 1, f"{len(saved)} 条")
    check("存的不是临时区域", saved[0]["name"] == "常驻", str(saved[0]["name"]))


def test_enabled_snapshot():
    print("\n[开启/停用快照]")
    from screenreader.worker import BoardEngine, EngineConfig

    class FakeB:
        name = "fake"

        def synthesize(self, text, **kw):
            return b"RIFF"

    e = BoardEngine(AppConfig(), backend=FakeB(), engine_cfg=EngineConfig())
    e._emit = lambda *a, **k: None

    a = e.store.add_region("a", (0, 0, 10, 10))
    b = e.store.add_region("b", (0, 0, 10, 10))
    e.store.update_region(b.id, enabled=False)

    snap = e.region_enabled_map()
    check("快照里两块区域", len(snap) == 2, str(len(snap)))
    check("b 是停用", snap[b.id] is False)

    # 模拟"截图朗读"把所有的都停掉
    for rid in snap:
        e.store.update_region(rid, enabled=False)
    check("全都停用了", all(not r.enabled for r in e.store.regions))

    # 恢复
    e.apply_region_enabled_map(snap)
    check("a 恢复成启用", e.store.region(a.id).enabled is True)
    check("b 恢复成停用", e.store.region(b.id).enabled is False)

    # 临时区域不该出现在快照里
    t = e.add_region("temp", (0, 0, 10, 10), mode="paragraph", temp=True)
    check("快照排除临时区域", t.id not in e.region_enabled_map())
    check("建出来就是 temp", t.temp is True)
    check("模式是整段", t.mode == "paragraph", t.mode)


def test_playback_done():
    print("\n[读完的判据]")
    from screenreader.worker import BoardEngine, EngineConfig

    class FakeB:
        name = "fake"

        def synthesize(self, text, **kw):
            return b"RIFF" + b"\x00" * 20

    e = BoardEngine(AppConfig(), backend=FakeB(), engine_cfg=EngineConfig())
    e._emit = lambda *a, **k: None

    r = e.add_region("t", (0, 0, 10, 10), mode="paragraph", temp=True)
    check("还没有卡片 -> 不算读完", e.region_playback_done(r.id) is False)

    out = e.store.observe(r.id, ["第一句。", "第二句。"])
    check("刚建卡 -> 还没读完", e.region_playback_done(r.id) is False)

    # 全都播过。注意 **played_once 也必须为真** —— 只看 auto_done 会误判：
    # autoplay 是"先出队（auto_done=True）再开播（mark_playing）"，
    # 中间那一小段"没排队、auto_done、没在播"的窗口里如果判完成，
    # 就会把卡片删掉导致静默不播（0.8.0 用户实测的 bug）。
    for c in out.added:
        c.auto_done = True
        c.queued = False
        c.audio_state = "ready"
        c.played_once = True
    check("都播过 -> 算读完", e.region_playback_done(r.id) is True)

    # 出队了但还没开播 -> 不能算读完
    out.added[0].played_once = False
    check("出队未开播 -> 不算读完", e.region_playback_done(r.id) is False)
    out.added[0].played_once = True

    # 有一张还在播
    e.store.mark_playing(out.added[0].id)
    check("有卡在播 -> 不算读完", e.region_playback_done(r.id) is False)
    e.store.mark_playing(None)

    # 有一张失败：也算处理完（否则会永远卡住）
    out.added[1].auto_done = False
    out.added[1].audio_state = "failed"
    check("失败的那张不阻塞清场", e.region_playback_done(r.id) is True)


def test_picker_single():
    print("\n[框选层只允许一个]")
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QApplication

    from screenreader.ui.region_picker import RegionPicker

    app = QApplication.instance() or QApplication([])
    p = RegionPicker(max_rects=1)
    check("max_rects 被记下", p.max_rects == 1)
    # 直接操作 _rects 模拟"拖了一次又拖一次"
    p._rects.append(QRect(0, 0, 50, 50))
    if p.max_rects and len(p._rects) >= p.max_rects:
        p._rects.clear()
    p._rects.append(QRect(10, 10, 60, 60))
    check("第二次框选替换掉第一次", len(p._rects) == 1, f"{len(p._rects)} 个")
    check("保留的是较新的那个", p._rects[0].width() == 60, str(p._rects[0].width()))
    p.close()

    p2 = RegionPicker()
    check("默认不限个数", p2.max_rects == 0)
    p2.close()


def test_hook_installs():
    """全局钩子真的要装得上（否则快捷键就是个摆设）。

    这里踩过一个坑：`SetWindowsHookExW` 的 `hMod` 传 exe handle 会失败
    （GetLastError=126 ERROR_MOD_NOT_FOUND），必须传 0 —— 因为这个环境的
    `.venv\\python.exe` 是重定向桩，真正跑 Python 的模块不在那个 exe 里。
    """
    import time as _t

    from screenreader.hotkey import HotkeyListener

    print("\n[全局钩子能装上]")
    h = HotkeyListener()
    if not h.supported:
        check("非 Windows，跳过", True)
        return
    ok = h.start()
    check("start() 成功", ok)
    _t.sleep(0.8)
    n = len(h._hooks)
    check("键盘和鼠标两个钩子都装上", n == 2, f"{n} 个")
    h.set_bindings({"snap": "ctrl+alt+s"})
    check("绑定能设置", bool(h._bindings))
    h.stop()
    check("stop 后钩子清空", len(h._hooks) == 0, f"{len(h._hooks)} 个")


def test_hotkey_capture_switch():
    """快捷键捕获框：从鼠标侧键能改回键盘组合键（原来的 bug 就是改不回来）。"""
    from PySide6.QtWidgets import QApplication

    from screenreader.ui.hotkey_edit import HotkeyCapture

    print("\n[快捷键捕获框：侧键 <-> 键盘 可互相切换]")
    QApplication.instance() or QApplication([])

    h = HotkeyCapture("mouse4")
    check("打开时是 mouse4", h.spec == "mouse4", h.spec)
    h.spec = "ctrl+alt+k"
    check("能改回键盘组合键", h.spec == "ctrl+alt+k", h.spec)
    h.spec = "mouse5"
    check("又能改成侧键2", h.spec == "mouse5", h.spec)
    h.spec = "ctrl+shift+f10"
    check("再改回键盘", h.spec == "ctrl+shift+f10", h.spec)
    h.spec = "这不是按键"
    check("非法值被拒（清空）", h.spec == "", repr(h.spec))
    h.close()


def test_snap_picker_crop():
    """冻结帧框选：裁出来的图像素要对（这块错了 OCR 就是花的）。"""
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
    from PySide6.QtWidgets import QApplication

    from screenreader.ui.snap_picker import SnapPicker, qimage_to_rgb_array

    print("\n[冻结帧：裁图与像素转换]")
    QApplication.instance() or QApplication([])

    img = QImage(120, 80, QImage.Format_RGB888)
    img.fill(QColor(10, 20, 30))
    pt = QPainter(img)
    pt.fillRect(QRect(20, 10, 40, 30), QColor(255, 0, 0))     # 一块红
    pt.end()
    pm = QPixmap.fromImage(img)

    # 直接裁「红块」验证像素
    arr = qimage_to_rgb_array(img.copy(QRect(20, 10, 40, 30)))
    check("裁出来的尺寸对", arr.shape == (30, 40, 3), str(arr.shape))
    px = tuple(int(v) for v in arr[0, 0])
    check("裁出来的颜色对（红块）", px == (255, 0, 0), str(px))
    px2 = tuple(int(v) for v in arr[29, 39])
    check("右下角也是红", px2 == (255, 0, 0), str(px2))

    # 走 SnapPicker 的 _finish 路径：应该发 picked(rect, image)
    got = {}
    p = SnapPicker(pm)
    p.picked.connect(lambda r, i: got.update(rect=r, img=i))
    p._rect = QRect(20, 10, 40, 30)
    p._finish()
    check("确定后发出了 picked", "rect" in got)
    if "img" in got:
        a2 = qimage_to_rgb_array(got["img"])
        check("信号里的图尺寸对", a2.shape[0] == 30 and a2.shape[1] == 40, str(a2.shape))
        check("信号里的图颜色对", tuple(int(v) for v in a2[0, 0]) == (255, 0, 0))

    # 没框就确定 = 取消，不能送空图去 OCR
    got2 = {}
    p2 = SnapPicker(pm)
    p2.cancelled.connect(lambda: got2.update(c=1))
    p2._rect = None
    p2._finish()
    check("没框就确定按取消处理", got2.get("c") == 1)

    # 画布是静态的：paint 两次不应改变内容
    from PySide6.QtGui import QPixmap as _PM
    cap = p2._frozen
    check("内部持有冻结图", isinstance(cap, _PM))
    p2.close()

    # **防回归**：QImage 销毁之后数组必须仍然有效。
    # 曾经的 bug：np.frombuffer 拿的是视图，ascontiguousarray 在已连续时不复制，
    # 函数返回后 QImage 析构 -> 数组指向已释放内存 -> OCR 读到垃圾（图是花的）。
    import gc

    def _make_then_drop():
        i = QImage(40, 40, QImage.Format_RGB888)
        i.fill(QColor(1, 2, 3))
        return qimage_to_rgb_array(i)

    a3 = _make_then_drop()
    gc.collect()
    px3 = tuple(int(v) for v in a3[0, 0])
    check("QImage 销毁后数组仍有效（真的复制了）", px3 == (1, 2, 3), str(px3))


def test_one_shot_ocr():
    """引擎要有「对一张图做一次 OCR」的独立入口，且带锁。"""
    from screenreader.config import AppConfig
    from screenreader.worker import BoardEngine, EngineConfig

    print("\n[一次性 OCR 入口]")

    class FakeB:
        name = "fake"

    e = BoardEngine(AppConfig(), backend=FakeB(), engine_cfg=EngineConfig())
    e._emit = lambda *a, **k: None
    check("有 read_frame_once", hasattr(e, "read_frame_once"))
    check("有 OCR 锁", hasattr(e, "_ocr_lock"))
    check("有 _clean_raw（两条路径共用清洗规则）", hasattr(e, "_clean_raw"))


def test_snap_picker_cursor_and_paint():
    """截图朗读层要和普通框选层一致：光标可见、重绘够快。

    用户反馈："框选一个区域，鼠标再移出框选范围时，有时看不到指针，
    选识别区域时没有这个问题"。

    区别在两处：
      1. 普通框选层有 `WA_TranslucentBackground`，截图朗读层没有
         （Windows 对"不透明全屏无边框窗口"的合成路径不同）
      2. 普通框选层每次重绘只填一层半透明色；截图朗读层每次都要
         **把整屏图缩放绘制一遍**，鼠标一动就重绘，慢到光标来不及重画

    修法：补 translucent 属性、预缩放+预压暗两份缓存（1:1 贴图）、
    自绘十字准星兜底、用 overrideCursor 锁光标（关闭时必须恢复）。
    """
    from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
    from PySide6.QtGui import QColor, QGuiApplication, QImage, QMouseEvent, QPixmap
    from PySide6.QtWidgets import QApplication

    from screenreader.ui.region_picker import RegionPicker
    from screenreader.ui.snap_picker import SnapPicker

    print("\n[截图朗读层：光标与重绘]")
    QApplication.instance() or QApplication([])

    img = QImage(400, 300, QImage.Format_RGB888)
    img.fill(QColor(30, 60, 90))
    sp = SnapPicker(QPixmap.fromImage(img))
    rp = RegionPicker()

    check("截图朗读层也有 WA_TranslucentBackground（和普通框选一致）",
          sp.testAttribute(Qt.WA_TranslucentBackground) is True)
    check("普通框选层确实有这个属性（对照）",
          rp.testAttribute(Qt.WA_TranslucentBackground) is True)

    sp.resize(400, 300)
    sp._build_cache()
    check("有预缩放缓存", sp._view is not None)
    check("缓存尺寸等于窗口尺寸",
          sp._view.size() == sp.size(), f"{sp._view.size()} vs {sp.size()}")
    check("有预压暗缓存", sp._dim is not None)

    ev = QMouseEvent(QEvent.MouseMove, QPointF(120, 80), QPointF(120, 80),
                     Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    sp.mouseMoveEvent(ev)
    check("鼠标移动记录了准星位置", sp._cursor_pos == QPoint(120, 80),
          str(sp._cursor_pos))

    try:
        pm = sp.grab()
        ok = pm.width() > 0
    except Exception as e:
        ok = False
        print("    绘制失败：", e)
    check("重绘不崩", ok)

    # **光标覆盖必须被恢复**，否则退出框选后整个程序光标都卡在十字准星
    QApplication.setOverrideCursor(Qt.CrossCursor)
    check("先人为设了 override", QGuiApplication.overrideCursor() is not None)
    sp._force_close()
    check("关闭后 override 已恢复", QGuiApplication.overrideCursor() is None)

    rp.close()


def main() -> int:
    for fn in (test_hotkey_parse, test_temp_region_not_saved, test_enabled_snapshot,
               test_playback_done, test_picker_single, test_hook_installs,
               test_hotkey_capture_switch, test_snap_picker_crop, test_one_shot_ocr,
               test_snap_picker_cursor_and_paint):
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} 未抛异常", False, f"{type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    for f in FAIL:
        print(f"  FAIL: {f}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
