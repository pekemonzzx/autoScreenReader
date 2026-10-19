"""语音运行时安装对话框。

**目标机没有 Python**，所以运行时安装不能只靠命令行——必须能在打包好的
exe 界面里完成。这个对话框就是那条路径：

    首次启动发现没装运行时 → 弹出来 → 选安装位置 → 选来源（在线/本地）
    → 自动下载（进度、速度、可取消）→ 装完自动配置

下载在 `InstallWorker`（QThread）里跑，**绝不碰控件**，只发信号；
界面线程收信号更新进度条。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from .. import runtime as R

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
class InstallWorker(QThread):
    """后台安装运行时。只发信号，不碰界面。"""

    # 注意：字节数必须用 qlonglong（64 位）！
    # Qt 的 `int` 是 **32 位**，而运行时压缩包有 9 GB = 9.6e9 > 2^31-1，
    # 走 `int` 会溢出成负数 —— 用户看到的进度条就是
    # "gsv.zip  -1.6GB / 1.1GB  -42.6MB/s" 这种鬼样子。
    progressed = Signal(str, "qlonglong", "qlonglong")   # 名称, 已完成, 总数
    status = Signal(str)                   # 一行状态文字
    finished_ok = Signal(dict)             # 结果
    failed = Signal(str)

    def __init__(self, dest: Path, source: str, manifest: Optional[str] = None,
                 voice_src: Optional[str] = None, parent=None) -> None:
        super().__init__(parent)
        self.dest = Path(dest)
        # source: "online" | "local"
        self.source = source
        self.manifest = manifest or ""
        self.voice_src = (voice_src or "").strip()
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def _should_stop(self) -> bool:
        return self._stop

    def run(self) -> None:
        try:
            if self.source == "voice":
                # 只装声库：运行时已经装好了，这轮只是加音色。
                # 以前这种情形会被 _start() 的"还没填来源"挡掉，声库根本装不上。
                res = {"ok": True, "skipped_runtime": True}
            elif self.source == "online":
                self.status.emit(f"读取清单：{self.manifest}")
                res = R.install(
                    self.manifest, self.dest,
                    progress=lambda n, d, t: self.progressed.emit(n, d, t),
                    say=lambda m: self.status.emit(str(m).strip()),
                    should_stop=self._should_stop,
                )
            else:
                res = self._install_local()
            if res.get("cancelled"):
                self.status.emit("已取消")
                self.finished_ok.emit({"ok": False, "cancelled": True})
                return

            # ---- 声库：**不管运行时成不成都要装** ----
            #
            # 以前这里在运行时失败时直接 return，声库安装被整个跳过。
            # 但两者本来就独立：运行时是 9GB 的推理引擎，声库只是一个音色包，
            # 装到 voices\ 下不需要运行时在位。用户明明勾了声库却没解压，
            # 原因就在这里（只要运行时校验有任何一条不过，声库就悄悄不装了）。
            voice_res = None
            if self.voice_src:
                voice_res = self._install_voice()
            res["voice_requested"] = bool(self.voice_src)

            if not res.get("ok"):
                msg = f"安装失败：{res}"
                if voice_res is not None:
                    msg += f"\n（声库单独安装结果：{voice_res}）"
                log.error("运行时安装失败：%s  声库结果：%s", res, voice_res)
                self.failed.emit(msg)
                return

            if voice_res is not None:
                res["voice"] = voice_res

            self.finished_ok.emit(res)
        except R.InstallCancelled:
            self.finished_ok.emit({"ok": False, "cancelled": True})
        except Exception as e:
            log.exception("运行时安装失败")
            self.failed.emit(str(e))

    def _install_voice(self) -> dict:
        """装声库，并**自动选中**它。

        装到哪很讲究：必须和 `AppConfig.voices_root()` 的判断一致。
        以前这里写死 `cfg.voices_root()`，而它当年只会返回 `<程序目录>/voices`——
        用户在向导里把运行时换到 D 盘之后，声库却跑回程序目录，
        用户看到的就是「声库压缩包没有正确解压到安装目录/voices/ 下」。

        现在跟着运行时位置走：运行时在 D:\\X\\gsv，声库就装到 D:\\X\\voices。
        """
        try:
            from ..config import AppConfig, default_config_path
            from ..voice import VoiceStore

            src = Path(self.voice_src)
            log.info("开始安装声库：%s", src)
            self.status.emit(f"安装声库：{src.name if src.exists() else src}")
            if not src.exists():
                msg = f"声库文件不存在：{src}"
                log.error(msg)
                return {"ok": False, "message": msg}

            cfg = AppConfig.load(default_config_path())
            # 运行时刚装完的话，configure() 已经把它的位置写进配置了；
            # 万一运行时没装成（或用户跳过了），就用向导里选的那个目录。
            rt = (cfg.tts.runtime_root or "").strip() or str(self.dest)

            # **声库优先装到 `<程序目录>\voices`**。
            #
            # 用户反馈过两次，期望都是这个位置：
            #   "解压到 voices\ 下" / "安装目录/voices/"
            # 而且程序默认就是从这儿找声库的，装这儿最不容易出意外。
            # 只有程序目录写不进去（比如装在 Program Files）才退到运行时旁边。
            base = cfg.voices_roots()[0]
            try:
                from ..runtime import check_writable
                if check_writable(base).get("ok"):
                    target = base
                else:
                    target = cfg.voices_dir_for_runtime(rt)
            except Exception:
                target = base
            self.status.emit(f"声库目录：{target}")
            log.info("声库目标目录：%s（运行时 %s）", target, rt or "(未设置)")
            target.mkdir(parents=True, exist_ok=True)

            store = VoiceStore(target)
            names = store.install_any(
                src,
                overwrite=True,
                say=lambda m: self.status.emit(str(m).strip()),
            )
            if not names:
                log.error("install_any 返回空：%s", src)
                return {"ok": False, "message": "没装到任何声库", "dir": str(target)}

            cfg.tts.voice = names[0]
            cfg.apply_voice(target)
            cfg.save(default_config_path())
            # 让推理配置指向真正的运行时（声库包里带的可能是打包者的路径）
            try:
                R.ensure_voice_configs(cfg, say=lambda m: self.status.emit(str(m).strip()))
            except Exception as e:
                log.warning("重建声库推理配置失败：%s", e)

            # 装完必须**核对程序找得到** —— 否则装对了也白搭
            found = cfg.voices_root()
            ok_find = (found / names[0] / "voice.json").exists()
            self.status.emit(f"已安装并选中声库：{names[0]}")
            log.info("声库装到 %s；voices_root() 解析为 %s；能找到=%s",
                     target, found, ok_find)
            if not ok_find:
                log.warning("声库装到了 %s，但程序会去 %s 找 —— 可能找不到", target, found)
            return {"ok": True, "names": names, "active": names[0],
                    "dir": str(target), "found": str(found)}
        except Exception as e:
            log.exception("声库安装失败")
            self.status.emit(f"声库安装失败：{e}")
            try:
                return {"ok": False, "message": str(e),
                        "dir": str(locals().get("target", ""))}
            except Exception:
                return {"ok": False, "message": str(e)}

    def _install_local(self) -> dict:
        """本地安装。**直接交给 runtime.install() 处理**。

        以前这里自己建了个 `<目标>/_local_extract` 暂存目录再解压，
        结果多出一个失败点（目录不可写就整个挂掉），还白占一份磁盘。
        现在 install() 会按类型自己分派：zip 直接解压到目标、目录直接复制。
        """
        src = Path(self.manifest)
        return R.install(
            src, self.dest,
            progress=lambda n, d, t: self.progressed.emit(n, d, t),
            say=lambda m: self.status.emit(str(m).strip()),
            should_stop=self._should_stop,
        )


# --------------------------------------------------------------------------- #
class RuntimeInstallDialog(QDialog):
    """运行时安装向导。可以在首次启动时自动弹，也可以从工具栏再打开。"""

    def __init__(self, parent=None, first_run: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("语音运行时安装")
        self.setMinimumWidth(680)
        self.worker: Optional[InstallWorker] = None
        self.first_run = first_run
        self._t0 = 0.0
        self._build()
        self._refresh_status()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        intro = QLabel(
            "语音（TTS）推理运行时约 <b>9&nbsp;GB</b>，因为体积大、平时又用不到，"
            "没有放进安装包，改为<b>按需下载</b>。<br>"
            "没装运行时程序照样能识别文字，只是念不出声音。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#8b949e;")
        root.addWidget(intro)

        # ---- 当前状态 ----
        self.lbl_status = QLabel()
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        # ---- 安装位置 ----
        box_dest = QGroupBox("安装位置")
        f1 = QFormLayout(box_dest)
        row = QHBoxLayout()
        self.ed_dest = QLineEdit(str(R.suggest_dest()))
        self.ed_dest.textChanged.connect(self._update_space)
        row.addWidget(self.ed_dest, 1)
        btn_browse_dest = QPushButton("浏览…")
        btn_browse_dest.clicked.connect(self._pick_dest)
        row.addWidget(btn_browse_dest)
        w = QWidget()
        w.setLayout(row)
        f1.addRow("装到：", w)
        self.lbl_space = QLabel()
        self.lbl_space.setWordWrap(True)
        self.lbl_space.setStyleSheet("color:#8b949e;")
        f1.addRow("", self.lbl_space)
        root.addWidget(box_dest)

        # ---- 来源（本地在前：多数人是网盘下载后本地安装）----
        box_src = QGroupBox("安装来源")
        v = QVBoxLayout(box_src)

        self.rb_local = QRadioButton("本地文件（网盘下载好的压缩包，或已解压的文件夹）")
        self.rb_local.setChecked(True)          # 默认本地，最常用也最省事
        self.rb_local.toggled.connect(self._on_source_changed)
        v.addWidget(self.rb_local)

        row3 = QHBoxLayout()
        row3.addSpacing(22)
        self.ed_local = QLineEdit()
        self.ed_local.setPlaceholderText(
            "gsv.zip / gsv_part*.zip，或已解压出来的运行时文件夹"
        )
        row3.addWidget(self.ed_local, 1)
        self.btn_browse_local = QPushButton("浏览…")
        self.btn_browse_local.clicked.connect(self._pick_local)
        row3.addWidget(self.btn_browse_local)
        v.addLayout(row3)

        self.rb_online = QRadioButton("在线下载（清单地址，或单个压缩包的直链）")
        self.rb_online.toggled.connect(self._on_source_changed)
        v.addWidget(self.rb_online)

        row2 = QHBoxLayout()
        row2.addSpacing(22)
        self.ed_manifest = QLineEdit()
        self.ed_manifest.setPlaceholderText(
            "…/runtime_manifest.json （分卷）  或  …/gsv.zip （单个压缩包直链）"
        )
        row2.addWidget(self.ed_manifest, 1)
        self.btn_browse_manifest = QPushButton("浏览…")
        self.btn_browse_manifest.clicked.connect(self._pick_manifest)
        row2.addWidget(self.btn_browse_manifest)
        v.addLayout(row2)

        # 这条警示只对"在线"有意义，选本地时藏起来，免得满屏黄字
        self.lbl_online_warn = QLabel(
            "⚠ 必须填<b>直链</b>：点开就直接下载文件的地址。<br>"
            "网盘<b>分享页面的网址不是直链</b>（返回的是网页，有的还要提取码）——"
            "那种情况请用上面的「本地文件」。<br>"
            "支持需要认证的直链（WebDAV / S3）：把账号密码写进网址即可，"
            "形如 <code>https://用户名:密码@主机/路径/gsv.zip</code>"
        )
        self.lbl_online_warn.setWordWrap(True)
        self.lbl_online_warn.setStyleSheet("color:#d29922; font-size:11px;")
        v.addWidget(self.lbl_online_warn)

        hint = QLabel(
            "本地文件夹里应该有 <code>runtime\\python.exe</code>；"
            "多套了一层目录也没关系，会自动找到，zip 也会自动解压。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#8b949e; font-size:11px;")
        v.addWidget(hint)
        root.addWidget(box_src)

        # ---- 声库（可选，顺便装）----
        box_voice = QGroupBox("声库（音色，可选）")
        vv = QVBoxLayout(box_voice)
        vv.addWidget(QLabel("想现在就装上音色的话，在这里选一个："))
        row4 = QHBoxLayout()
        self.ed_voice = QLineEdit()
        self.ed_voice.setPlaceholderText(
            "xxx.voicepack.zip，或已解压的声库文件夹（也可以选装着多个声库的上级目录）"
        )
        row4.addWidget(self.ed_voice, 1)
        self.btn_browse_voice = QPushButton("浏览…")
        self.btn_browse_voice.clicked.connect(self._pick_voice)
        row4.addWidget(self.btn_browse_voice)
        vv.addLayout(row4)
        vhint = QLabel()
        vhint.setWordWrap(True)
        vhint.setStyleSheet("color:#8b949e; font-size:11px;")
        self.lbl_voice_dest = vhint
        vv.addWidget(vhint)
        root.addWidget(box_voice)

        # ---- 进度 ----
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self.lbl_progress = QLabel("")
        self.lbl_progress.setWordWrap(True)
        self.lbl_progress.setStyleSheet("color:#8b949e;")
        self.lbl_progress.setVisible(False)
        root.addWidget(self.lbl_progress)

        # ---- 按钮 ----
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_install = QPushButton("开始安装")
        self.btn_install.setDefault(True)
        self.btn_install.clicked.connect(self._start)
        btns.addWidget(self.btn_install)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_cancel.setVisible(False)
        btns.addWidget(self.btn_cancel)

        self.btn_close = QPushButton("稍后再说" if self.first_run else "关闭")
        self.btn_close.clicked.connect(self.reject)
        btns.addWidget(self.btn_close)
        root.addLayout(btns)

        self._on_source_changed()

    # ------------------------------------------------------------------ #
    def _refresh_status(self) -> None:
        st = R.status(R.runtime_root())
        if st["installed"]:
            self.lbl_status.setText(
                f'<span style="color:#3fb950">✔ 已安装</span>　'
                f'版本 {st.get("version", "?")}　大小 {R.human(st["size"])}<br>'
                f'位置：{st["root"]}'
            )
            self.btn_install.setText("重新安装")
        else:
            extra = ""
            if st.get("missing"):
                extra = f'<br><span style="color:#f85149">缺失文件：{st["missing"][:3]}</span>'
            self.lbl_status.setText(
                f'<span style="color:#f0883e">✘ 未安装</span>　'
                f"装好后才能把识别到的文字念出来{extra}"
            )
        self._update_space()

    def _update_space(self) -> None:
        dest = Path(self.ed_dest.text().strip() or ".")
        self._update_voice_dest()      # 安装位置一改，声库落点也跟着改
        free = R.free_space(dest)
        wr = R.check_writable(dest)

        parts = []
        if free >= 0:
            need = 11 * 1024 ** 3
            color = "#f85149" if free < need else "#8b949e"
            parts.append(
                f'<span style="color:{color}">所在磁盘剩余 {R.human(free)}'
                f"（安装约需 9 GB）</span>"
            )
        if wr["ok"]:
            parts.append('<span style="color:#3fb950">该位置可写 ✔</span>')
        else:
            # 提前把问题暴露出来，别等解压到一半才炸；并说清楚"为什么"和"怎么办"
            why = R.describe_location(dest).replace("\n", "<br>")
            parts.append(f'<span style="color:#f85149">✘ 不能装到这里：{wr["reason"]}</span>')
            parts.append(f'<span style="color:#d29922">{why}</span>')
            self.btn_install.setEnabled(False)
            self.lbl_space.setText("<br>".join(parts))
            return
        self.btn_install.setEnabled(True)
        self.lbl_space.setText("<br>".join(parts))

    def _on_source_changed(self) -> None:
        online = self.rb_online.isChecked()
        self.ed_manifest.setEnabled(online)
        self.btn_browse_manifest.setEnabled(online)
        self.ed_local.setEnabled(not online)
        self.btn_browse_local.setEnabled(not online)
        # 那条"必须是直链"的警示只对在线方式有意义
        self.lbl_online_warn.setVisible(online)

    def _update_voice_dest(self) -> None:
        """把「声库会装到哪」明明白白写出来。

        用户反馈过两次对不上，所以这里说死：**优先装在程序目录的 voices\\ 下**，
        只有那个位置写不进去才退到运行时旁边。
        """
        try:
            from ..config import AppConfig

            cfg = AppConfig()
            base = cfg.voices_roots()[0]
            try:
                from ..runtime import check_writable
                writable = check_writable(base).get("ok")
            except Exception:
                writable = True
            if writable:
                target = base
                why = "程序目录下的 voices\\"
            else:
                target = cfg.voices_dir_for_runtime(self.ed_dest.text().strip())
                why = "程序目录写不进去，改放到运行时旁边"
            self.lbl_voice_dest.setText(
                f"会装到：<code>{target}</code><br>（{why}。留空也行，之后再装。）"
            )
        except Exception as e:
            log.warning("计算声库目录失败：%s", e)
            self.lbl_voice_dest.setText("留空也行，之后再装。")

    # ------------------------------------------------------------------ #
    def _pick_dest(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择运行时的安装位置", self.ed_dest.text())
        if d:
            self.ed_dest.setText(d)
            self._update_space()

    def _pick_manifest(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "选择清单或压缩包（也可以直接在上面的框里填网址）",
            "", "清单或压缩包 (*.json *.zip);;所有文件 (*)",
        )
        if f:
            self.ed_manifest.setText(f.replace("\\", "/"))

    def _pick_local(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "选择运行时压缩包（取消则改为选文件夹）", "", "压缩包 (*.zip);;所有文件 (*)",
        )
        if f:
            self.ed_local.setText(f)
            return
        d = QFileDialog.getExistingDirectory(self, "选择已解压的运行时文件夹")
        if d:
            self.ed_local.setText(d)

    def _pick_voice(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "选择声库包（取消则改为选文件夹）", "",
            "声库包 (*.zip);;所有文件 (*)",
        )
        if f:
            self.ed_voice.setText(f)
            return
        d = QFileDialog.getExistingDirectory(self, "选择声库文件夹（也可以选装着多个声库的上级目录）")
        if d:
            self.ed_voice.setText(d)

    # ------------------------------------------------------------------ #
    def _start(self) -> None:
        dest = Path(self.ed_dest.text().strip() or str(R.runtime_root()))
        online = self.rb_online.isChecked()
        src = self.ed_manifest.text().strip() if online else self.ed_local.text().strip()
        voice_src = self.ed_voice.text().strip()

        # 只装声库：运行时早就装好了，用户这次只想加个音色包。
        #
        # 以前这里一律要求填运行时来源，否则弹「还没填来源」直接返回 ——
        # 结果用户选好了声库包也装不上，看起来就是「声库没有自动解压」。
        if not src:
            # 判定"运行时在位"要比 R.status() 宽松：
            # status() 在任一关键文件缺失时会把 installed 置 False，
            # 但裁剪版/老版本运行时恰恰会缺几项 —— 那种情况下用户更该能装声库。
            # 只要解释器在，就认为运行时在位。
            try:
                rt_dir = Path(R.runtime_root())
                rt_present = (rt_dir / "runtime" / "python.exe").exists()
            except Exception:
                rt_dir, rt_present = Path(str(dest)), False
            if rt_present and voice_src:
                if not Path(voice_src).exists():
                    QMessageBox.warning(self, "声库路径不存在", f"找不到：\n{voice_src}")
                    return
                self._begin(rt_dir, "voice", "", voice_src)
                return
            QMessageBox.warning(
                self, "还没填来源",
                "请填写清单/压缩包的网址，或选择一个本地压缩包/文件夹。\n\n"
                "（如果运行时已经装好了、这次只想加声库："
                "把上面的运行时来源留空，只选「声库（音色，可选）」即可。）",
            )
            return
        if online and not src.startswith(("http://", "https://")) and not Path(src).exists():
            QMessageBox.warning(self, "地址不对",
                                "在线安装需要 http(s) 直链。\n\n"
                                "如果你手上是网盘的分享页面网址，那个程序下不了——\n"
                                "请先把文件下载到本地，然后改用「本地文件」。")
            return
        if not online and not Path(src).exists():
            QMessageBox.warning(self, "路径不存在", f"找不到：{src}")
            return

        # 装之前先确认目标能写。9 GB 装到一半才发现写不了，用户会看到一句
        # 莫名其妙的 [WinError 5]，磁盘上还留一堆残渣。
        wr = R.check_writable(dest)
        if not wr["ok"]:
            alt = R.suggest_dest()
            msg = QMessageBox(self)
            msg.setWindowTitle("这个位置装不了")
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"无法写入：\n{dest}\n\n{wr['reason']}")
            msg.setInformativeText(
                R.describe_location(dest)
                + f"\n\n建议装到：\n  {alt}"
                + f"\n（该盘剩余 {R.human(R.free_space(alt))}）"
            )
            b_alt = msg.addButton("用建议位置", QMessageBox.AcceptRole)
            msg.addButton("我自己改", QMessageBox.RejectRole)
            msg.exec()
            if msg.clickedButton() is b_alt:
                self.ed_dest.setText(str(alt))
                self._update_space()
            return
        if not online and Path(src).is_file() and Path(src).suffix.lower() != ".zip":
            QMessageBox.warning(self, "格式不对",
                                "本地文件只能是 .zip 压缩包，或者一个已解压的文件夹。")
            return

        self._begin(dest, "online" if online else "local", src, voice_src)

    def _begin(self, dest: Path, source: str, src: str, voice_src: str) -> None:
        """真正开始安装（运行时 + 可选声库）。"""
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.lbl_progress.setVisible(True)
        self.lbl_progress.setText("准备中…")
        self.btn_install.setEnabled(False)
        self.btn_close.setEnabled(False)
        self.btn_cancel.setVisible(True)
        self._t0 = time.time()

        self.worker = InstallWorker(
            dest, source, src, voice_src=voice_src, parent=self,
        )
        self.worker.progressed.connect(self._on_progress)
        self.worker.status.connect(self._on_status)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.lbl_progress.setText("正在取消…（已下载的部分会保留，下次断点续传）")

    def _on_status(self, msg: str) -> None:
        # 状态文字可能很长，只留最后一行有信息量的
        line = msg.strip().splitlines()[-1] if msg.strip() else ""
        if line:
            self.lbl_progress.setText(line)

    def _on_progress(self, name: str, done: int, total: int) -> None:
        # 防御：万一还有别的地方传了溢出的负数，不要显示成 "-1.6GB"
        done = max(0, int(done))
        total = max(0, int(total))
        if total > 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(min(100, int(done * 100 / total)))
        else:
            self.progress.setRange(0, 0)   # 未知总量，走不确定进度
        el = max(0.1, time.time() - self._t0)
        sp = done / el
        eta = ""
        if total > done and sp > 0:
            e = R.human_eta((total - done) / sp)
            if e:
                eta = f"　{e}"
        self.lbl_progress.setText(
            f"{name}　{R.human(done)} / {R.human(total)}　{R.human(sp)}/s{eta}"
        )

    def _on_done(self, res: dict) -> None:
        self.btn_install.setEnabled(True)
        self.btn_close.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.progress.setVisible(False)
        if res.get("cancelled"):
            self.lbl_progress.setText("已取消。已下载的部分保留着，下次会断点续传。")
            return
        self._refresh_status()
        only_voice = bool(res.get("skipped_runtime"))
        lines = ["声库安装完成 ✅" if only_voice else "语音运行时安装完成 ✅", ""]
        v = res.get("voice")
        requested = res.get("voice_requested")
        if v and v.get("ok"):
            lines.append(f"声库已装好并选中：{v.get('active')}")
            if v.get("dir"):
                lines.append(f"存放位置：{v['dir']}")
            lines.append("")
            lines.append("现在重启本程序就能用了。")
        elif v and not v.get("ok"):
            lines.append("声库没装成：" if only_voice else "运行时装好了，但声库没装成：")
            lines.append(f"  {v.get('message', '未知原因')}")
            if v.get("dir"):
                lines.append(f"  尝试装到：{v['dir']}")
            lines.append("")
            lines.append("可以手动装：把 .voicepack.zip 里的声库文件夹解压到")
            fallback_dir = v.get("dir") or "程序目录的 voices 目录下"
            lines.append(f"  {fallback_dir}")
        elif requested:
            # 用户明明选了声库，结果连结果都没有 —— 这是 bug，别让他自己猜
            lines.append("声库没有装上（程序内部没返回结果）。")
            lines.append("请把 data\\screenreader.log 发给开发者。")
        elif only_voice:
            lines.append("没有选择声库文件。")
        else:
            lines.append("还差一个声库（音色）才能出声：")
            lines.append("把 xxx.voicepack.zip 解压到程序目录的 voices\\ 目录下，")
            lines.append("重开程序后在工具栏「声库」下拉框里选一个。")
        lines.append("")
        lines.append("然后重启本程序。")

        # 装完直接给一个「现在重启」按钮（用户要求），省得自己去找 exe 再点一次
        box = QMessageBox(self)
        box.setWindowTitle("安装完成")
        box.setIcon(QMessageBox.Information)
        box.setText("\n".join(lines))
        b_restart = box.addButton("现在重启", QMessageBox.AcceptRole)
        box.addButton("稍后自己重启", QMessageBox.RejectRole)
        box.setDefaultButton(b_restart)
        box.exec()
        restart = box.clickedButton() is b_restart
        self.accept()
        if restart:
            self._restart_app()

    def _restart_app(self) -> None:
        """重启本程序。

        打包后 `sys.executable` 就是 读屏台.exe，重新拉一个自己再退出即可。
        用 `proc.relaunch()` —— 它会用分离进程拉起，并**确认新进程没立刻退出**
        （用户反馈过"点了重启，当前关了但新的没起来"）。
        """
        import sys

        from ..proc import relaunch

        try:
            if getattr(sys, "frozen", False):
                cmd = [sys.executable] + list(sys.argv[1:])
                workdir = str(Path(sys.executable).resolve().parent)
            else:
                cmd = [sys.executable] + list(sys.argv)
                workdir = str(Path.cwd())
            log.info("准备重启：%s（cwd=%s）", cmd, workdir)
            res = relaunch(cmd, cwd=workdir)
            log.info("重启结果：%s", res)
            if not res.get("ok"):
                QMessageBox.warning(
                    self, "重启失败",
                    "没能自动重新打开程序，请手动双击 读屏台.exe。\n\n"
                    f"（{res.get('detail')}）",
                )
                return
        except Exception as e:
            log.exception("重启失败")
            QMessageBox.warning(self, "重启失败", f"请手动重新打开程序。\n\n{e}")
            return
        # 新进程已经确认起来了，退出当前实例。
        # 服务和引擎由主窗口的关闭流程收尾。
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _on_fail(self, msg: str) -> None:
        self.btn_install.setEnabled(True)
        self.btn_close.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.progress.setVisible(False)
        self.lbl_progress.setText("安装失败")
        box = QMessageBox(self)
        box.setWindowTitle("安装失败")
        box.setIcon(QMessageBox.Critical)
        box.setText(msg[:600])
        extra = []
        low = msg.lower()
        if "winerror 5" in low or "拒绝访问" in msg or "permission" in low:
            extra.append(
                "这看起来是**权限问题**：目标目录不可写，或被安全软件拦了。\n"
                f"建议换一个位置，例如：{R.suggest_dest()}"
            )
        if "不是 zip" in msg or "网页" in msg:
            extra.append(
                "如果来源是网盘分享链接，请先把文件下载到本地，"
                "再改用「本地文件」选那个 zip。"
            )
        if extra:
            box.setInformativeText("\n\n".join(extra))
        box.exec()

    def closeEvent(self, ev) -> None:
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(
                self, "还在安装", "安装还没完成，确定要关闭吗？\n（已下载的部分会保留）"
            ) != QMessageBox.Yes:
                ev.ignore()
                return
            self.worker.cancel()
            self.worker.wait(3000)
        super().closeEvent(ev)
