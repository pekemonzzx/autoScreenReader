"""界面冒烟测试：离屏创建 QApplication + 主窗口，验证信号/槽接线不炸。

不弹窗、不识别、不发声。运行：
    .\\.venv\\Scripts\\python tests\\test_ui_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # 离屏，不弹窗

# **必须在建窗口之前**把配置路径指到临时文件。
# MainWindow 保存识别区域时会写 default_config_path()，不隔离的话
# 会把真的 data/config.json（声库、参考音频）冲掉 —— 这个坑真踩过。
_TMP_CFG_DIR = tempfile.mkdtemp(prefix="sr_uismoke_")
os.environ["SCREENREADER_CONFIG"] = str(Path(_TMP_CFG_DIR) / "config.json")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from screenreader.board import BoardConfig, Card  # noqa: E402
from screenreader.config import AppConfig  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def main() -> int:
    app = QApplication.instance() or QApplication([])

    print("[模块导入]")
    from screenreader.ui import main as ui_main  # noqa: F401
    from screenreader.ui.cards import CardList, CardRow  # noqa: F401
    from screenreader.ui.main_window import EngineBridge, MainWindow, OverlayWindow  # noqa: F401
    from screenreader.ui.region_picker import RegionPicker, virtual_desktop_rect  # noqa: F401
    check("界面模块可导入", True)

    print("\n[虚拟桌面]")
    geo = virtual_desktop_rect()
    check("能取到桌面尺寸", geo.width() > 0 and geo.height() > 0, f"{geo.width()}x{geo.height()}")

    print("\n[主窗口构建]")
    cfg = AppConfig()
    cfg.ocr.autotune = False
    win = MainWindow(cfg)
    check("主窗口创建成功", win is not None)
    check("卡片列表初始为空", win.card_list.count == 0)

    print("\n[卡片增删改 / 单按钮切换]")
    r = win.engine.add_region("测试", (0, 0, 300, 200))
    win._refresh_regions()  # noqa: SLF001
    check("区域列表已刷新", win.region_list.count() == 1, f"{win.region_list.count()} 项")
    check("区域角标已同步", len(win.region_overlay._regions) == 1)  # noqa: SLF001

    out = win.engine.store.observe(r.id, ["第一句测试", "第二句测试"])
    for c in out.added:
        win.card_list.add_or_update(c)
    check("卡片已加入界面", win.card_list.count == 2, f"{win.card_list.count} 张")

    c0 = out.added[0]
    row = win.card_list._rows[c0.id]  # noqa: SLF001
    # 卡片行现在有两个按钮：行首的播放/停止，和行尾的「重新生成」
    btns = row.findChildren(type(row.btn))
    check("卡片有两个按钮（播放 + 重新生成）", len(btns) == 2, f"{len(btns)} 个")
    check("行首是播放按钮", row.btn.text() == "▶", row.btn.text())
    check("未就绪时显示 ▶", row.btn.text() == "▶", row.btn.text())
    check("按钮始终可点（会等合成）", row.btn.isEnabled())

    c0.audio = b"\x00\x00" * 100
    c0.audio_state = "ready"
    win.card_list.add_or_update(c0)
    check("就绪后仍是 ▶", row.btn.text() == "▶", row.btn.text())
    check("就绪状态文案", row.state.text() == "就绪", row.state.text())

    win.engine.store.mark_playing(c0.id)
    win.card_list.add_or_update(c0)
    check("播放中变 ⏹", row.btn.text() == "⏹", row.btn.text())
    check("播放中状态文案", row.state.text() == "播放中", row.state.text())

    c0.audio_state = "failed"
    c0.audio_error = "测试错误"
    c0.play_state = "idle"
    win.card_list.add_or_update(c0)
    check("失败变 ↻", row.btn.text() == "↻", row.btn.text())
    c0.audio_state = "ready"

    print("\n[两个视图同步]")
    win.overlay.show()
    win._refresh_cards()  # noqa: SLF001
    check("主列表有卡片", win.card_list.count == 2, f"{win.card_list.count}")
    check("悬浮条也被填充", win.overlay.cards.count == 2, f"{win.overlay.cards.count}")
    win.overlay.hide()

    print("\n[两个区域同时开：悬浮窗要显示全部]")
    # 用户反馈：单行 + 整段两个区域一起开，悬浮窗里的卡片会消失，像"卡住"。
    # 根因是 _refresh_cards 把"非当前选中区域"的卡片也从悬浮窗删掉了。
    r2 = win.engine.add_region("整段区", (320, 0, 300, 200), mode="paragraph")
    win.engine.store.observe(r2.id, ["整段区域的一句话。"])
    win._refresh_regions()  # noqa: SLF001

    def select_region(rid):
        for i in range(win.region_list.count()):
            if win.region_list.item(i).data(Qt.UserRole) == rid:
                win.region_list.setCurrentRow(i)
                return True
        return False

    select_region(r.id)          # 选中「单行」那个区域
    app.processEvents()
    win._refresh_cards()  # noqa: SLF001
    total = len(win.engine.store.sorted_cards())
    sel_count = win.card_list.count
    ov_count = win.overlay.cards.count
    check("主列表只显示选中区域", sel_count < total, f"列表 {sel_count} / 全部 {total}")
    check("悬浮窗显示**所有**区域", ov_count == total, f"悬浮窗 {ov_count} / 全部 {total}")

    select_region(r2.id)         # 切到「整段」那个区域
    app.processEvents()
    win._refresh_cards()  # noqa: SLF001
    check("切区域后悬浮窗仍是全部", win.overlay.cards.count == total,
          f"{win.overlay.cards.count} / {total}")

    # **必须清理**：后面的用例都假设只有一个区域、且没有选中项
    win.engine.remove_region(r2.id)
    win._refresh_regions()  # noqa: SLF001
    select_region(r.id)
    app.processEvents()
    win._refresh_cards()  # noqa: SLF001
    check("清理后回到单区域状态",
          len(win.engine.store.regions) == 1, f"{len(win.engine.store.regions)} 个区域")
    # 恢复"无选中"的初始状态（后面的用例依赖这一点）
    win.region_list.clearSelection()
    win.region_list.setCurrentRow(-1)
    app.processEvents()

    print("\n[事件桥接]")
    got = []
    win.bridge.event.connect(lambda e: got.append(e))
    win.engine._emit("card_added", card_id=c0.id, region_id=r.id, text="x")  # noqa: SLF001
    app.processEvents()
    check("引擎事件能投递到主线程", any(e.get("type") == "card_added" for e in got), f"{len(got)} 个事件")

    print("\n[卡片销毁联动画]")
    # 先让"播放中"状态收尾：正在播的卡片现在**不会被自动销毁**
    # （否则音频会被掐断），这里不模拟播放，直接清干净
    win.engine.store.mark_playing(None)
    for _ in range(3):
        out2 = win.engine.store.observe(r.id, [""])
    for c in out2.died:
        win._remove_card(c.id)  # noqa: SLF001
    check("消亡卡片已从主列表移除", win.card_list.count == 0, f"{win.card_list.count} 张")
    check("消亡卡片已从悬浮条移除", win.overlay.cards.count == 0, f"{win.overlay.cards.count} 张")
    check("空列表提示重新出现", win.card_list.showing_empty_hint)
    check("播放状态已复位", win.engine.store.playing_id is None)

    print("\n[悬浮条]")
    ov = OverlayWindow()
    ov.cards.add_or_update(out.added[0])
    check("悬浮条可加卡片", ov.cards.count == 1)
    check("悬浮条默认隐藏", not ov.isVisible())

    print("\n[悬浮条：拖动条 / 关闭 / 调整大小]")
    # 拖动条现在**常显**（用户要求）：既不能整体 hide()（会让列表跳），
    # 也不能只藏内部控件（hover 过滤器装在它们身上，一藏就再也收不到鼠标移动）
    check("拖动条常显", not ov.header.isHidden())
    check("标题常显", not ov.lbl_title.isHidden())
    check("「全部」按钮常显", not ov.btn_play_all.isHidden())
    check("「全部」按钮可用", ov.btn_play_all.isEnabled())
    check("收起时高度仍占位", ov.header.height() > 0 or ov.header.minimumHeight() > 0,
          f"{ov.header.height()}")
    check("右上角有关闭按钮", ov.btn_close.text() == "✕", ov.btn_close.text())
    from PySide6.QtWidgets import QSizeGrip

    check("右下角有尺寸手柄", ov.findChild(QSizeGrip) is not None)
    check("设了最小尺寸（防止拖成看不见）",
          ov.minimumWidth() >= 200 and ov.minimumHeight() >= 100,
          f"{ov.minimumWidth()}x{ov.minimumHeight()}")
    # 拖动条只在鼠标移进来时出现
    ov._poke_header()  # noqa: SLF001
    check("鼠标碰一下拖动条就出现", not ov.header.isHidden())
    # 关闭按钮要通知主窗口把工具栏勾选去掉
    got_closed = []
    ov.closed.connect(lambda: got_closed.append(1))
    ov.btn_close.click()
    check("点 ✕ 会发 closed 信号", bool(got_closed))

    print("\n[安装对话框：本地优先 / 提示按需显示]")
    from screenreader.ui.runtime_dialog import RuntimeInstallDialog

    dlg = RuntimeInstallDialog()
    check("默认选中「本地文件」", dlg.rb_local.isChecked())
    check("默认没选「在线下载」", not dlg.rb_online.isChecked())
    check("本地来源时在线警示是收起的", dlg.lbl_online_warn.isHidden())
    dlg.rb_online.setChecked(True)
    app.processEvents()
    check("切到在线后警示才出现", not dlg.lbl_online_warn.isHidden())
    check("在线时本地输入框禁用", not dlg.ed_local.isEnabled())
    dlg.rb_local.setChecked(True)
    app.processEvents()
    check("切回本地警示又收起", dlg.lbl_online_warn.isHidden())
    check("对话框里有「声库（可选）」来源", hasattr(dlg, "ed_voice"))
    check("声库可只选文件或文件夹", dlg.ed_voice.placeholderText() != "")
    dlg.close()

    print("\n[识别模式：单行 / 整段]")
    check("工具栏下方有识别模式下拉", hasattr(win, "cmb_mode"))
    modes = [win.cmb_mode.itemData(i) for i in range(win.cmb_mode.count())]
    check("两种模式都在", modes == ["line", "paragraph"], str(modes))
    check("默认没有选中区域时禁用", not win.cmb_mode.isEnabled())
    check("有模式说明文字", win.lbl_mode_hint.text() != "")

    # 建一个区域 -> 选中 -> 切成整段
    r = win.engine.add_region("整段测试", (0, 0, 300, 200))
    win._refresh_regions()  # noqa: SLF001
    check("区域列表里能看到它", win.region_list.count() >= 1)
    # 注意按 id 找行：前面的用例可能已经建过区域，第 0 行不一定是它
    row = next(
        (i for i in range(win.region_list.count())
         if win.region_list.item(i).data(Qt.UserRole) == r.id),
        -1,
    )
    check("能在列表里按 id 找到该区域", row >= 0, str(row))
    win.region_list.setCurrentRow(row)
    app.processEvents()
    check("选中后下拉可用", win.cmb_mode.isEnabled())
    check("新区域默认单行", win.engine.store.region(r.id).mode == "line")

    win.cmb_mode.setCurrentIndex(modes.index("paragraph"))
    app.processEvents()
    check("切换后区域变成整段", win.engine.store.region(r.id).mode == "paragraph")
    check("列表标签带 [整段]",
          "[整段]" in win.region_list.item(row).text(), win.region_list.item(row).text())
    check("切换时清掉了旧卡片（避免新旧混读）", win.engine.store.stats()["cards"] == 0)
    win.cmb_mode.setCurrentIndex(modes.index("line"))
    app.processEvents()
    check("还能切回单行", win.engine.store.region(r.id).mode == "line")

    print("\n[顺序朗读按钮（手动点，不自动播）]")
    check("有顺序朗读按钮", hasattr(win, "btn_play_para"), win.btn_play_para.text())
    win.cmb_mode.setCurrentIndex(modes.index("line"))
    app.processEvents()
    # 用户要求：**单行模式也要能顺序播放**，所以按钮对两种模式都可用
    check("单行区域也常亮（单行也能顺序播）", win.btn_play_para.isEnabled())
    win.cmb_mode.setCurrentIndex(modes.index("paragraph"))
    app.processEvents()
    check("整段区域同样可用", win.btn_play_para.isEnabled())

    out = win.engine.store.observe(r.id, ["第一句。", "第二句。"])
    win._refresh_cards()  # noqa: SLF001
    app.processEvents()
    check("整段区域选中时按钮可用", win.btn_play_para.isEnabled())

    # 用户要求：按钮**常亮**，不等合成好；点了就从第一项开始播
    first = next(c for c in win.engine.store.sorted_cards() if c.region_id == r.id)
    check("第一句还没就绪也照样可用（常亮）", win.btn_play_para.isEnabled())
    check("新卡不会自动进播放队列", win.engine.store.next_auto_card() is None)

    win.btn_play_para.click()
    app.processEvents()
    queued = [c.text for c in win.engine.store.sorted_cards() if c.queued]
    check("点按钮后进入队列", len(queued) == 2, str(queued))
    check("队列按从上到下顺序", queued == ["第一句。", "第二句。"], str(queued))

    print("\n[悬浮窗拖动条：开始识别 / 自动朗读 / 透明度]")
    ov = win.overlay

    # 开始/停止识别按钮
    check("拖动条有「开始识别」按钮", hasattr(ov, "btn_run"), ov.btn_run.text())
    ov.set_running(True)
    check("运行态显示「停止识别」", ov.btn_run.text() == "停止识别", ov.btn_run.text())
    ov.set_running(False)
    check("停止态显示「开始识别」", ov.btn_run.text() == "开始识别", ov.btn_run.text())
    hits = []
    # **先把主窗口的槽断开再测按钮** —— 否则真点会走 `_toggle_engine()`，
    # 它会去拉起 TTS 服务并在没有区域时弹模态框，测试会卡死。
    try:
        ov.run_toggle_requested.disconnect(win._toggle_engine)  # noqa: SLF001
    except Exception:
        pass
    ov.run_toggle_requested.connect(lambda: hits.append(1))
    ov.btn_run.click()
    app.processEvents()
    check("按钮能发出识别开关请求", bool(hits))
    ov.run_toggle_requested.connect(win._toggle_engine)  # noqa: SLF001

    # 自动朗读按钮
    check("拖动条有「自动朗读」按钮", hasattr(ov, "btn_autoread"))
    check("自动朗读是可勾选按钮", ov.btn_autoread.isCheckable())
    win.ck_autoread.setChecked(True)
    app.processEvents()
    check("左侧勾选 -> 悬浮窗按钮同步", ov.btn_autoread.isChecked())
    ov.btn_autoread.setChecked(False)
    app.processEvents()
    check("悬浮窗取消 -> 左侧勾选框同步", not win.ck_autoread.isChecked())
    check("悬浮窗取消 -> 引擎关闭", win.engine.ecfg.auto_read_all is False)

    # 透明度
    check("拖动条有透明度滑条", hasattr(ov, "sld_opacity"))
    check("透明度范围 20~100",
          (ov.sld_opacity.minimum(), ov.sld_opacity.maximum()) == (20, 100),
          f"{ov.sld_opacity.minimum()}~{ov.sld_opacity.maximum()}")
    ov.sld_opacity.setValue(55)
    app.processEvents()
    check("拖到 55% 窗口真的半透明", abs(ov.windowOpacity() - 0.55) < 0.02,
          f"{ov.windowOpacity():.2f}")
    check("透明度写进了配置", win.cfg.overlay_opacity == 55,
          str(win.cfg.overlay_opacity))
    ov.set_opacity_pct(100)
    check("set_opacity_pct 能复位", abs(ov.windowOpacity() - 1.0) < 0.02,
          f"{ov.windowOpacity():.2f}")

    print("\n[悬浮窗的「全部播放」]")
    check("悬浮窗有全部播放按钮", hasattr(win.overlay, "btn_play_all"),
          win.overlay.btn_play_all.text() if hasattr(win.overlay, "btn_play_all") else "")
    win._refresh_cards()  # noqa: SLF001
    app.processEvents()
    check("按钮常亮（不等第一句就绪）", win.overlay.btn_play_all.isEnabled())
    win.engine.store.unqueue_region(r.id)
    win.overlay.btn_play_all.click()
    app.processEvents()
    q2 = [c.text for c in win.engine.store.sorted_cards() if c.queued]
    check("悬浮窗按钮能触发整段朗读", len(q2) == 2, str(q2))

    # 清空后仍然常亮（用户要求"常亮"）
    win.engine.store.clear_region(r.id)
    win._refresh_cards()  # noqa: SLF001
    app.processEvents()
    check("没有卡片时仍然常亮", win.overlay.btn_play_all.isEnabled())

    print("\n[框选：单击=确认，拖动=框选]")
    from screenreader.ui.region_picker import DRAG_THRESHOLD, RegionPicker

    check("有拖动阈值（手抖不算拖动）", DRAG_THRESHOLD >= 3, str(DRAG_THRESHOLD))
    import inspect

    rel = inspect.getsource(RegionPicker.mouseReleaseEvent)
    check("没拖动时走确认流程", "_finish()" in rel)
    check("拖动才真的加区域", "_rects.append" in rel)
    paint = inspect.getsource(RegionPicker.paintEvent)
    check("拖动中也挖亮（不压暗）", paint.count("self._punch(p, local)") >= 2)
    check("提示里有「单击=确认」", "单击=确认" in RegionPicker.HINT, RegionPicker.HINT[:40])

    print("\n[新增控制项：自动朗读 / 语速 / 重新生成]")

    # 自动朗读勾选框
    check("有「自动朗读」勾选框", hasattr(win, "ck_autoread"))
    if hasattr(win, "ck_autoread"):
        check("自动朗读默认不勾", not win.ck_autoread.isChecked())
        win.ck_autoread.setChecked(True)
        app.processEvents()
        check("勾选后引擎开启自动朗读", win.engine.ecfg.auto_read_all is True)
        win.ck_autoread.setChecked(False)
        app.processEvents()
        check("取消后引擎关闭", win.engine.ecfg.auto_read_all is False)

    # 语速滑条
    check("有语速滑条", hasattr(win, "sld_speed"))
    if hasattr(win, "sld_speed"):
        check("语速范围 0.5~2.0", (win.sld_speed.minimum(), win.sld_speed.maximum()) == (5, 20),
              f"{win.sld_speed.minimum()}~{win.sld_speed.maximum()}")
        check("默认 1.0x", win.sld_speed.value() == 10, str(win.sld_speed.value()))
        win.sld_speed.setValue(15)
        app.processEvents()
        check("拖到 1.5x 引擎生效", abs(win.engine.cfg.tts.speed_factor - 1.5) < 1e-6,
              str(win.engine.cfg.tts.speed_factor))
        check("标签显示 1.5x", win.lbl_speed.text() == "1.5x", win.lbl_speed.text())
        win.sld_speed.setValue(10)
        app.processEvents()

    # 卡片上的「重新生成」按钮
    if win.card_list.count > 0:
        row = next(iter(win.card_list._rows.values()))  # noqa: SLF001
        check("卡片行有「重新生成」按钮", hasattr(row, "btn_retry"))
        got = []
        win.card_list.retry_requested.connect(lambda cid: got.append(cid))
        if hasattr(row, "btn_retry"):
            row.btn_retry.click()
            app.processEvents()
        check("点⟳ 会发出重试请求", bool(got), f"{got}")

    print("\n[工具箱]")
    from PySide6.QtWidgets import QToolBar

    tb = win.findChild(QToolBar)
    check("工具栏存在", tb is not None)
    actions = [a.text() for a in tb.actions() if a.text()]
    # 布局重排后的约定：**高频功能固定在工具栏**，其余收进「≡ 更多」菜单。
    # 这样默认窗口宽度下每个按钮都看得见（以前末尾按钮会被挤掉、找不到）。
    # 「全部停止」已合并进「开始识别/停止识别」，一级页面只留一个运行开关。
    for key in ("框选区域", "截图朗读", "开始识别", "删除区域", "悬浮窗", "区域角标"):
        check(f"工具栏有「{key}」", any(key in a for a in actions), str(actions))
    check("「全部停止」不再占一级位置",
          not any("全部停止" in a for a in actions), str(actions))
    check("低频功能收进「更多」菜单",
          hasattr(win, "btn_more") and win.btn_more.menu() is not None)
    if hasattr(win, "btn_more") and win.btn_more.menu() is not None:
        more = [a.text() for a in win.btn_more.menu().actions() if a.text()]
        check("「更多」里有清空/停朗读/快捷键/设置/运行时",
              all(any(k in m for m in more)
                  for k in ("清空卡片", "停止朗读", "快捷键", "高级设置", "运行时")),
              str(more))
    check("左栏可滚动（小窗口也能看到全部设置）",
          hasattr(win, "left_scroll") and win.left_scroll.widgetResizable())

    win.close()
    ov.close()
    app.processEvents()

    print("\n" + "=" * 60)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    for f in FAIL:
        print(f"  FAIL: {f}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
