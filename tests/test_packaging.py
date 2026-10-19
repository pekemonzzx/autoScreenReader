"""声库包与运行时的测试（纯逻辑，不需要 GPU / 网络 / 真实运行时）。

运行：
    .\\.venv\\Scripts\\python tests\\test_packaging.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screenreader import runtime as R  # noqa: E402
from screenreader.voice import (  # noqa: E402
    PACK_SUFFIX,
    PACK_VERSION,
    VoiceStore,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def expect_eq(name, got, want):
    check(name, got == want, f"得到 {got!r}，期望 {want!r}")


def _mkvoice(td: Path, name="v1"):
    """造一个自包含的声库（权重是假的小文件，测试不关心内容）。"""
    store = VoiceStore(Path(td) / "voices")
    p = store.create(name, display_name="测试音色", version="v2ProPlus",
                     gsv_root=str(Path(td) / "gsv"))
    d = store.dir_of(name)
    # 假权重放在整合包目录里，模拟"训练产物"的原始位置
    gsv = Path(td) / "gsv"
    (gsv / "GPT_weights_v2ProPlus").mkdir(parents=True, exist_ok=True)
    (gsv / "SoVITS_weights_v2ProPlus").mkdir(parents=True, exist_ok=True)
    g = gsv / "GPT_weights_v2ProPlus" / f"{name}-e12.ckpt"
    s = gsv / "SoVITS_weights_v2ProPlus" / f"{name}_e8_s248.pth"
    g.write_bytes(b"gpt" * 100)
    s.write_bytes(b"sovits" * 100)
    p.gpt_weights = f"GPT_weights_v2ProPlus/{g.name}"
    p.sovits_weights = f"SoVITS_weights_v2ProPlus/{s.name}"
    (d / "ref.wav").write_bytes(b"RIFF" + b"\x00" * 200)
    p.ref_audio = "ref.wav"
    p.prompt_text = "这是参考文本"
    store.save(p)
    return store, p


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 全局护栏：**任何测试都不许碰真实的 data/config.json 和 voices/**
#
# 踩过的坑：有的测试直接调 R.install()/R.install_from_dir()，它们内部会调
# configure()，而 configure() 默认写真实的 config.json，
# 还会把真实声库的 tts_infer.yaml 改成指向临时目录 —— 把开发机的配置写脏了。
# 所以这里统一把 configure 换成写临时配置的版本，测试里不用各自记得包装。
# --------------------------------------------------------------------------- #
_SAFE_CFG_ROOT = Path(tempfile.mkdtemp(prefix="sr_testcfg_"))
_real_configure = R.configure


def _safe_configure(root, say=print, config_path=None):
    # 显式传了 config_path 就尊重它（个别测试要检查写进去的内容）；
    # 没传才落到这个临时配置上，绝不碰真实的 data/config.json。
    cfg = Path(config_path) if config_path else (_SAFE_CFG_ROOT / "data" / "config.json")
    cfg.parent.mkdir(parents=True, exist_ok=True)
    (cfg.parent / "voices").mkdir(exist_ok=True)
    return _real_configure(root, say=say, config_path=cfg)


R.configure = _safe_configure


def test_pack_and_install():
    print("\n[声库打包与安装]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        store, p = _mkvoice(td)
        d = store.dir_of("v1")

        check("打包前不是自包含（权重在整合包目录）", not p.is_self_contained(d))

        out = store.pack("v1", td / "packs", say=lambda *a: None)
        check("生成了声库包", out.exists() and out.name.endswith(PACK_SUFFIX), out.name)
        check("打包后变自包含", store.load("v1").is_self_contained(d))

        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            man = json.loads(z.read("manifest.json").decode("utf-8"))
        check("包里有 manifest", "manifest.json" in names)
        check("manifest 版本正确", man["pack_version"] == PACK_VERSION)
        expect_eq("manifest 记了声库名", man["name"], "v1")
        check("权重进包", any("weights/" in n and n.endswith(".ckpt") for n in names))
        check("参考音频进包", "voice/ref.wav" in names)
        check("默认不含训练标注", not any(n.endswith("train.list") for n in names))
        check("不含生成的推理配置", not any(n.endswith("tts_infer.yaml") for n in names))

        # 装到另一个全新的 store，模拟别人的机器
        store2 = VoiceStore(td / "other" / "voices")
        expect_eq("目标为空", len(store2.list_names()), 0)
        prof = store2.install(out, say=lambda *a: None)
        expect_eq("安装后能列出", store2.list_names(), ["v1"])
        d2 = store2.dir_of("v1")
        check("安装后自包含", prof.is_self_contained(d2))
        g, s = prof.weight_paths(d2)
        check("权重指向档案内", d2 in g.parents and d2 in s.parents, str(g.parent))
        expect_eq("参考文本保留", prof.prompt_text, "这是参考文本")

        # 重复安装要报错，除非 --overwrite
        try:
            store2.install(out, say=lambda *a: None)
            check("重名拒绝", False)
        except FileExistsError:
            check("重名拒绝", True)
        prof2 = store2.install(out, overwrite=True, say=lambda *a: None)
        check("--overwrite 可覆盖", prof2.name == "v1")


def test_pack_rejects_incomplete():
    print("\n[打包拒绝不完整的声库]")
    with tempfile.TemporaryDirectory() as td:
        store, p = _mkvoice(Path(td))
        (Path(td) / "gsv" / "GPT_weights_v2ProPlus" / "v1-e12.ckpt").unlink()
        try:
            store.pack("v1", Path(td) / "packs", say=lambda *a: None)
            check("缺 GPT 权重时拒绝打包", False)
        except FileNotFoundError as e:
            check("缺 GPT 权重时拒绝打包", "GPT" in str(e))


def test_install_rejects_bad_zip():
    print("\n[安装拒绝非法包]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        store = VoiceStore(td / "voices")
        bad = td / "bad.voicepack.zip"
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("hello.txt", "not a voice pack")
        try:
            store.install(bad, say=lambda *a: None)
            check("缺 manifest 时拒绝", False)
        except ValueError as e:
            check("缺 manifest 时拒绝", "manifest" in str(e))

        # 版本不匹配
        bad2 = td / "old.voicepack.zip"
        with zipfile.ZipFile(bad2, "w") as z:
            z.writestr("manifest.json", json.dumps({"pack_version": 999, "name": "x"}))
        try:
            store.install(bad2, say=lambda *a: None)
            check("版本不匹配时拒绝", False)
        except ValueError as e:
            check("版本不匹配时拒绝", "版本" in str(e))


def test_install_blocks_path_traversal():
    print("\n[安装阻断目录穿越]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        store = VoiceStore(td / "voices")
        evil = td / "evil.voicepack.zip"
        with zipfile.ZipFile(evil, "w") as z:
            z.writestr("manifest.json", json.dumps(
                {"pack_version": PACK_VERSION, "name": "evil"}))
            z.writestr("voice/voice.json", json.dumps({"name": "evil"}))
            z.writestr("voice/ref.wav", b"RIFF")
            # 试图写到包外
            z.writestr("voice/../../../../pwned.txt", "should not escape")
            z.writestr("../../pwned2.txt", "should not escape either")
        store.install(evil, say=lambda *a: None)
        escaped = [p for p in td.parent.rglob("pwned*.txt")] + [p for p in td.rglob("pwned*.txt")]
        check("没有文件被写到包外", not escaped, str(escaped[:3]))
        check("声库本身装进来了", store.exists("evil"))


def test_install_rejects_incomplete_pack():
    print("\n[安装拒绝缺少 voice.json 的包]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        store = VoiceStore(td / "voices")
        bad = td / "bad.voicepack.zip"
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("manifest.json", json.dumps(
                {"pack_version": PACK_VERSION, "name": "bad"}))
            z.writestr("voice/ref.wav", b"RIFF")
        try:
            store.install(bad, say=lambda *a: None)
            check("缺少 voice.json 时拒绝", False)
        except ValueError as e:
            check("缺少 voice.json 时拒绝", "voice.json" in str(e))
        check("失败的安装不留残渣", not store.exists("bad"))


# --------------------------------------------------------------------------- #
def test_manifest_roundtrip():
    print("\n[运行时清单]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        man = R.RuntimeManifest(
            runtime_version="1", base_url="https://example.com/gsv",
            total_size=123,
            parts=[R.RuntimePart(name="gsv_part01.zip", sha256="a" * 64, size=100),
                   R.RuntimePart(name="gsv_part02.zip", sha256="b" * 64, size=23)],
        )
        p = td / R.MANIFEST_NAME
        man.save(p)
        back = R.RuntimeManifest.load(p)
        expect_eq("版本", back.runtime_version, "1")
        expect_eq("分卷数", len(back.parts), 2)
        expect_eq("base_url", back.base_url, "https://example.com/gsv")
        expect_eq("URL 拼接", back.resolve_url(back.parts[0]),
                  "https://example.com/gsv/gsv_part01.zip")
        # 显式 url 优先
        back.parts[1].url = "https://cdn.example.com/x.zip"
        expect_eq("显式 url 优先", back.resolve_url(back.parts[1]),
                  "https://cdn.example.com/x.zip")
        # 从 dict 加载
        d = R.RuntimeManifest.load({"parts": [{"name": "a.zip"}]})
        expect_eq("dict 加载", d.parts[0].name, "a.zip")


def test_runtime_status():
    print("\n[运行时状态]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        st = R.status(td / "gsv")
        check("空目录未安装", not st["installed"])

        # 造一个假的运行时目录
        root = td / "gsv"
        for rel in R.REQUIRED:
            p = root / rel.replace("/", "\\")
            if rel.endswith("chinese-hubert-base") or rel.endswith("chinese-roberta-wwm-ext-large") \
                    or rel.endswith("AP_BWE_main"):
                p.mkdir(parents=True, exist_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
        check("文件齐了", R.check_files(root) == [])
        (root / R.MARKER_NAME).write_text(json.dumps(
            {"version": "1", "installed_at": "now"}), encoding="utf-8")
        st = R.status(root)
        check("标记后视为已安装", st["installed"])
        check("无缺失文件", st["missing"] == [])

        # 删一个关键文件 -> 应该报未安装
        (root / "api_v2.py").unlink()
        st = R.status(root)
        check("缺文件时判定为未安装", not st["installed"])
        check("列出缺失文件", "api_v2.py" in st["missing"])


def test_repair_users_pth():
    print("\n[users.pth 隐形引用修复]")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "gsv"
        sp = root / "runtime" / "Lib" / "site-packages"
        sp.mkdir(parents=True)
        pth = sp / "users.pth"
        pth.write_text("D:\\GPT-SoVITS\\原打包者路径\nD:\\GPT-SoVITS\\原打包者路径/tools\n",
                       encoding="utf-8")
        (root / "tools").mkdir()
        (root / "GPT_SoVITS").mkdir()

        R.repair_users_pth(root, say=lambda *a: None)
        txt = pth.read_text(encoding="utf-8")
        check("已指向新位置", str(root) in txt)
        check("不再含原路径", "原打包者路径" not in txt)
        check("只保留存在的子目录", "tools" in txt and "tools/asr" not in txt)


def test_install_from_dir_detects_nested():
    print("\n[离线安装：识别多余的一层目录]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 模拟"下载后解压出来多套了一层"
        nested = td / "下载" / "gsv-runtime" / "runtime"
        nested.mkdir(parents=True)
        (nested / "python.exe").write_bytes(b"x")
        (nested.parent / "api_v2.py").write_bytes(b"x")
        for rel in ("GPT_SoVITS/TTS_infer_pack/TTS.py",
                    "GPT_SoVITS/pretrained_models/chinese-hubert-base",
                    "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                    "tools/audio_sr.py", "tools/AP_BWE_main"):
            p = nested.parent / rel.replace("/", "\\")
            if rel.endswith("chinese-hubert-base") or rel.endswith("chinese-roberta-wwm-ext-large") \
                    or rel.endswith("AP_BWE_main"):
                p.mkdir(parents=True, exist_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")

        dst = td / "installed" / "gsv"
        res = R.install_from_dir(td / "下载", dst, say=lambda *a: None)
        check("自动找到内层目录并安装成功", res.get("ok"), str(res))
        check("目标目录里有 python.exe", (dst / "runtime" / "python.exe").exists())


def test_runtime_manifest_from_dict():
    print("\n[清单容错]")
    m = R.RuntimeManifest.load({"runtime_version": "2", "parts": []})
    expect_eq("空分卷", len(m.parts), 0)
    check("human 格式化", R.human(1536) == "1.5KB", R.human(1536))


def _fake_runtime(root: Path) -> Path:
    """造一个"结构齐全"的假运行时（内容是占位字节，测试不关心真伪）。"""
    for rel in R.REQUIRED:
        p = root / rel.replace("/", "\\")
        if rel.endswith(("chinese-hubert-base", "chinese-roberta-wwm-ext-large", "AP_BWE_main")):
            p.mkdir(parents=True, exist_ok=True)
            (p / "placeholder.bin").write_bytes(b"x")
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x" * 1024)
    # 来点够分卷的数据
    (root / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base" / "big.bin").write_bytes(
        b"y" * 3_000_000
    )
    (root / "tools" / "AP_BWE_main" / "big2.bin").write_bytes(b"z" * 2_000_000)
    # 一个带原包路径的 users.pth，用于验证安装时会修掉
    sp = root / "runtime" / "Lib" / "site-packages"
    sp.mkdir(parents=True, exist_ok=True)
    (sp / "users.pth").write_text("D:\\原打包者\\路径\nD:\\原打包者\\路径/tools\n", encoding="utf-8")
    return root


def test_end_to_end_download():
    """真起一个 HTTP 服务，走完 打包 -> 下载 -> 校验 -> 解压 -> 配置 全链路。"""
    print("\n[端到端：HTTP 下载安装运行时]")
    import functools
    import http.server
    import socketserver
    import threading

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _fake_runtime(td / "gsv-src")
        out = td / "release"

        # part_mb=1 -> 强制分成多卷，验证多卷拼接
        man_path = R.pack(src, out, base_url="", part_mb=1, say=lambda *a: None)
        man = json.loads(man_path.read_text(encoding="utf-8"))
        check("分成了多个卷", len(man["parts"]) >= 2, f"{len(man['parts'])} 卷")
        check("每卷都有 sha256", all(p.get("sha256") for p in man["parts"]))

        # 起本地 HTTP 服务
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
        httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            man["base_url"] = f"http://127.0.0.1:{port}"
            man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

            dest = td / "installed" / "gsv"
            cfg_file = td / "installed" / "data" / "config.json"
            cfg_file.parent.mkdir(parents=True, exist_ok=True)
            (cfg_file.parent / "voices").mkdir(exist_ok=True)

            seen = []
            real_configure = R.configure

            def fake_configure(root, say=print, config_path=None):
                return real_configure(root, say=say, config_path=cfg_file)

            R.configure = fake_configure          # noqa: 只为把配置写到临时目录
            try:
                res = R.install(
                    f"http://127.0.0.1:{port}/{R.MANIFEST_NAME}",
                    dest,
                    progress=lambda n, d, tot: seen.append((n, d, tot)),
                    say=lambda *a: None,
                )
            finally:
                R.configure = real_configure

            check("安装成功", res.get("ok"), str(res))
            check("下载过程有进度回调", len(seen) > 0, f"{len(seen)} 次回调")
            check("解压后文件齐", R.check_files(dest) == [], str(R.check_files(dest)))
            check("状态为已安装", R.status(dest)["installed"])
            check("sha256 校验通过（否则不会有标记）", (dest / R.MARKER_NAME).exists())

            pth = (dest / "runtime" / "Lib" / "site-packages" / "users.pth").read_text(encoding="utf-8")
            check("users.pth 的隐形引用被修掉", "原打包者" not in pth and str(dest) in pth)

            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            check("配置写入了 runtime_root", cfg["tts"]["runtime_root"] == str(dest),
                  cfg["tts"]["runtime_root"])

            # 分卷临时文件应该被清掉
            check("临时分卷已清理", not (dest / "_parts").exists()
                  or not any((dest / "_parts").iterdir()))

            # 篡改一个分卷，重新安装应该 sha256 校验失败
            (out / man["parts"][0]["name"]).write_bytes(b"corrupted")
            dest2 = td / "installed2" / "gsv"
            R.configure = fake_configure
            try:
                res2 = R.install(
                    f"http://127.0.0.1:{port}/{R.MANIFEST_NAME}",
                    dest2, say=lambda *a: None,
                )
            finally:
                R.configure = real_configure
            check("分卷被篡改时拒绝安装", not res2.get("ok"), str(res2))
        finally:
            httpd.shutdown()
            httpd.server_close()


def test_install_cancellation():
    print("\n[取消安装]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _fake_runtime(td / "gsv-src")
        out = td / "release"
        R.pack(src, out, part_mb=1, say=lambda *a: None)
        man = json.loads((out / R.MANIFEST_NAME).read_text(encoding="utf-8"))
        # 不真正下载，只验证 should_stop 在循环一开始就被尊重
        res = R.install(
            out / R.MANIFEST_NAME, td / "dest",
            say=lambda *a: None, should_stop=lambda: True,
        )
        check("立即取消返回 cancelled", res.get("cancelled"), str(res))


def test_launcher_bat_is_ascii_safe():
    """防止"start.bat 找不到 '???.exe'"这类事故再发生。

    事故经过：写 .bat 时用了 encoding="ascii", errors="replace"，
    把 `读屏台.exe` 里的中文静默替换成了 `?`，用户双击直接报
    「Windows 找不到文件 '???.exe'」。
    """
    print("\n[启动脚本必须纯 ASCII 且不硬编码文件名]")
    sys.path.insert(0, str(ROOT / "tools"))
    import build_installer as B

    check("START_BAT 本身是纯 ASCII", all(ord(c) < 128 for c in B.START_BAT))
    check("RUN_CONSOLE_BAT 本身是纯 ASCII", all(ord(c) < 128 for c in B.RUN_CONSOLE_BAT))
    check("start.bat 用通配符定位 exe（不硬编码文件名）", "%%~ff" in B.START_BAT)
    check("run_console.bat 也用通配符", "%%~ff" in B.RUN_CONSOLE_BAT)
    # 不能出现"引用某个具体 exe 文件名"的写法
    import re as _re

    for label, body in (("start.bat", B.START_BAT), ("run_console.bat", B.RUN_CONSOLE_BAT)):
        refs = _re.findall(r'"([^"%]*\.exe)"', body)
        check(f"{label} 只引用通配符/变量，不写死 exe 名", not refs, str(refs))

    # _ascii_bat 必须"报错"而不是"静默替换"
    try:
        B._ascii_bat("@echo off\nstart 读屏台.exe\n", "t.bat")
        check("写 .bat 遇到中文时报错（而不是替换成 ?）", False)
    except SystemExit as e:
        check("写 .bat 遇到中文时报错（而不是替换成 ?）", "ASCII" in str(e))
    # 纯 ASCII 的正常内容要能过
    try:
        B._ascii_bat("@echo off\nset APP=x\n", "t.bat")
        check("纯 ASCII 内容正常通过", True)
    except SystemExit:
        check("纯 ASCII 内容正常通过", False)


def test_single_zip_install():
    """单包直链：发布者只把整个 gsv 目录压成一个 zip 传网盘也能装。"""
    print("\n[单包安装（一条直链）]")
    import functools
    import http.server
    import socketserver
    import threading

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _fake_runtime(td / "gsv-src")
        out = td / "release"

        z = R.pack_single(src, out, say=lambda *a: None)
        check("生成了单个 zip", z.exists() and z.name == "gsv.zip")
        check("生成了 sha256 旁注文件", z.with_suffix(".zip.sha256").exists())

        cfg_file = td / "inst" / "data" / "config.json"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        (cfg_file.parent / "voices").mkdir(exist_ok=True)
        real_configure = R.configure

        def fake_configure(root, say=print, config_path=None):
            return real_configure(root, say=say, config_path=cfg_file)

        # --- 本地 zip ---
        R.configure = fake_configure
        try:
            res = R.install(z, td / "inst" / "gsv", say=lambda *a: None)
        finally:
            R.configure = real_configure
        check("从本地 zip 安装成功", res.get("ok"), str(res))
        check("文件齐全", R.check_files(td / "inst" / "gsv") == [])

        # --- 走 HTTP 直链 ---
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
        httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            seen = []
            R.configure = fake_configure
            try:
                res2 = R.install(
                    f"http://127.0.0.1:{port}/gsv.zip",
                    td / "inst2" / "gsv",
                    progress=lambda n, d, t: seen.append(d),
                    say=lambda *a: None,
                )
            finally:
                R.configure = real_configure
            check("从直链安装成功", res2.get("ok"), str(res2))
            check("有进度回调", len(seen) > 0, f"{len(seen)} 次")
            check("装完删掉了下载的 zip 腾空间",
                  not (td / "inst2" / "gsv" / "_parts" / "gsv.zip").exists())
            mark = json.loads((td / "inst2" / "gsv" / R.MARKER_NAME).read_text(encoding="utf-8"))
            check("标记里记了来源与 sha256",
                  mark.get("sha256") and "http" in mark.get("source", ""))

            # --- 拿到的是网页而不是 zip（模拟网盘分享链接）---
            (out / "fake.html").write_text("<html>网盘分享页面，需要提取码</html>", encoding="utf-8")
            res3 = R.install(f"http://127.0.0.1:{port}/fake.html",
                             td / "inst3" / "gsv", say=lambda *a: None)
            check("非 zip 时明确报错（而不是装出个坏运行时）", not res3.get("ok"))
        finally:
            httpd.shutdown()
            httpd.server_close()


def test_install_autodetect():
    print("\n[install() 自动判断来源类型]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        check("目录 -> 离线安装",
              R._looks_like_manifest(td) is False)   # 间接验证：目录不按 manifest 走
        check(".json -> 分卷", R._looks_like_manifest("https://x/runtime_manifest.json"))
        check(".json?query -> 分卷",
              R._looks_like_manifest("https://x/runtime_manifest.json?token=abc"))
        check(".zip -> 单包", not R._looks_like_manifest("https://x/gsv.zip"))
        check("无扩展名 -> 单包", not R._looks_like_manifest("https://x/download?id=1"))

        # 本地 .json 走分卷、本地 .zip 走单包
        src = _fake_runtime(td / "gsv-src")
        z = R.pack_single(src, td / "rel", say=lambda *a: None)
        cfg_file = td / "d" / "data" / "config.json"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        (cfg_file.parent / "voices").mkdir(exist_ok=True)
        real = R.configure
        R.configure = lambda root, say=print, config_path=None: real(root, say=say, config_path=cfg_file)
        try:
            r = R.install(z, td / "d" / "gsv", say=lambda *a: None)
        finally:
            R.configure = real
        check("本地 .zip 走单包并成功", r.get("ok"), str(r))


def test_unnest():
    print("\n[多套一层目录时自动提上来]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        inner = td / "gsv"
        (inner / "runtime").mkdir(parents=True)
        (inner / "runtime" / "python.exe").write_bytes(b"x")
        (inner / "api_v2.py").write_bytes(b"x")

        check("初始未识别", not (td / "runtime" / "python.exe").exists())
        ok = R._unnest(td, say=lambda *a: None)
        check("识别并提取成功", ok)
        check("python.exe 到了上层", (td / "runtime" / "python.exe").exists())
        check("api_v2.py 也上来了", (td / "api_v2.py").exists())
        check("空的内层目录被删掉", not inner.exists())


def test_url_auth_parsing():
    print("\n[直链里的账号密码解析]")
    c, a = R._split_auth("https://h/p.zip")
    check("无认证时原样返回", c == "https://h/p.zip" and a is None)
    c, a = R._split_auth("https://user:pass@h/p.zip")
    check("拆出 Basic 认证头", a == "Basic dXNlcjpwYXNz", str(a))
    check("URL 去掉了 userinfo", c == "https://h/p.zip", c)
    # 密码里可能有 @ 的 URL 编码、端口、query
    c, a = R._split_auth("https://u:p%40ss@host:8443/a/b.zip?x=1")
    check("保留端口与 query", c == "https://host:8443/a/b.zip?x=1", c)
    check("URL 编码的密码被解码", a == "Basic dTpwQHNz", str(a))
    # 没有冒号（只有用户名）不算认证
    c, a = R._split_auth("https://user@h/p.zip")
    check("只有用户名时不加认证头", a is None)


def test_authenticated_download():
    """WebDAV / 需要认证的 S3：直链里带账号密码也要能下。"""
    print("\n[带认证的直链下载]")
    import base64
    import functools
    import http.server
    import socketserver
    import threading

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out = td / "rel"
        out.mkdir()
        payload = b"Z" * 200_000
        (out / "gsv.zip").write_bytes(payload)

        want = "Basic " + base64.b64encode(b"alice:s3cret").decode()

        class AuthHandler(http.server.SimpleHTTPRequestHandler):
            """带 Basic 认证 + 支持 Range 的处理函数（模拟 WebDAV/S3）。"""

            def __init__(self, *a, **kw):
                super().__init__(*a, directory=str(out), **kw)

            def do_GET(self):
                if self.headers.get("Authorization") != want:
                    self.send_response(401)
                    self.send_header("WWW-Authenticate", 'Basic realm="x"')
                    self.end_headers()
                    return
                rng = self.headers.get("Range")
                if not rng or not rng.startswith("bytes="):
                    return super().do_GET()
                # 只支持 bytes=N- 这一种形式，够测了
                start = int(rng.split("=", 1)[1].split("-")[0])
                body = payload[start:]
                self.send_response(206)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        httpd = socketserver.TCPServer(("127.0.0.1", 0), AuthHandler)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            # 不带认证 -> 401，要有清楚提示
            dest1 = td / "a.zip"
            msgs = []
            ok = R.download_file(f"http://127.0.0.1:{port}/gsv.zip", dest1,
                                 say=lambda m: msgs.append(str(m)))
            check("无认证时失败", not ok)
            check("401 时提示怎么填账号密码",
                  any("401" in m and "user:pass" in m for m in msgs),
                  " / ".join(msgs[-2:])[:120])

            # 带认证 -> 成功
            dest2 = td / "b.zip"
            ok2 = R.download_file(
                f"http://alice:s3cret@127.0.0.1:{port}/gsv.zip", dest2,
                expect_size=len(payload), say=lambda *a: None,
            )
            check("带认证时下载成功", ok2)
            check("内容完整", dest2.exists() and dest2.read_bytes() == payload)

            # 断点续传也要带上认证头
            dest3 = td / "c.zip"
            (dest3.with_suffix(".zip.part")).write_bytes(payload[:50_000])
            ok3 = R.download_file(
                f"http://alice:s3cret@127.0.0.1:{port}/gsv.zip", dest3,
                expect_size=len(payload), say=lambda *a: None,
            )
            check("断点续传也带认证", ok3 and dest3.read_bytes() == payload)
        finally:
            httpd.shutdown()
            httpd.server_close()


def test_resume_when_server_ignores_range():
    """服务器不支持断点续传（忽略 Range 返回 200）时不能把文件追加坏。

    这是真踩到的坑：Python 自带的 SimpleHTTPRequestHandler 就不支持 Range，
    导致追加写入后得到一个长度翻倍、内容错乱的坏文件。
    """
    print("\n[服务器忽略 Range 时必须重下而不是追加]")
    import functools
    import http.server
    import socketserver
    import threading

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out = td / "rel"
        out.mkdir()
        payload = b"Q" * 120_000
        (out / "gsv.zip").write_bytes(payload)

        handler = functools.partial(  # 默认实现：完全不理会 Range
            http.server.SimpleHTTPRequestHandler, directory=str(out)
        )
        httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            dest = td / "d.zip"
            (dest.with_suffix(".zip.part")).write_bytes(payload[:40_000])   # 假装上次下了一半
            ok = R.download_file(
                f"http://127.0.0.1:{port}/gsv.zip", dest,
                expect_size=len(payload), say=lambda *a: None,
            )
            check("下载成功", ok)
            check("文件长度正确（没有被追加成两倍）",
                  dest.stat().st_size == len(payload), f"{dest.stat().st_size} vs {len(payload)}")
            check("内容正确", dest.read_bytes() == payload)
        finally:
            httpd.shutdown()
            httpd.server_close()


def test_extract_zip_robustness():
    """解压要能报进度、要抗长路径、跳过缓存、单个文件坏了不拖垮整体。"""
    print("\n[解压健壮性]")
    import inspect

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        z = td / "t.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("api_v2.py", b"x" * 1000)
            zf.writestr("runtime/python.exe", b"x" * 2000)
            zf.writestr("GPT_SoVITS/a/b/c.py", b"x" * 500)
            zf.writestr("dir/", b"")
            zf.writestr("__pycache__/junk.pyc", b"x" * 999)
            zf.writestr("a/__pycache__/junk2.pyc", b"x" * 999)
            zf.writestr("good/b.py", b"x" * 300)
            # 恶意条目
            zf.writestr("../../evil.txt", b"nope")
            zf.writestr("C:/abs.txt", b"nope")

        dest = td / "out"
        got = []
        ok = R.extract_zip(z, dest, say=lambda *a: None,
                           progress=lambda n, d, t: got.append((d, t)))
        check("解压成功", ok)
        check("有进度回调", len(got) > 0, f"{len(got)} 次")
        check("最终进度到达 100%", got and got[-1][0] == got[-1][1])

        check("正常文件解出来了", (dest / "api_v2.py").exists())
        check("多级目录解出来了", (dest / "GPT_SoVITS" / "a" / "b" / "c.py").exists())
        check("空目录建出来了", (dest / "dir").is_dir())
        check("__pycache__ 被跳过", not (dest / "__pycache__").exists())
        check("嵌套的 __pycache__ 也跳过", not (dest / "a" / "__pycache__").exists())
        check("其它文件不受影响", (dest / "good" / "b.py").exists())
        check("穿越条目被丢弃", not (td / "evil.txt").exists())
        check("绝对路径条目被丢弃", not Path("C:/abs.txt").exists())

        # 一个文件都没解出来 -> 判定失败（避免"装成功"其实空的）
        empty = td / "empty.zip"
        with zipfile.ZipFile(empty, "w") as zf:
            zf.writestr("__pycache__/only.pyc", b"x")
        check("全是缓存条目时判定为失败",
              not R.extract_zip(empty, td / "o2", say=lambda *a: None))

        # 目标不可写 -> 明确失败不抛异常
        f = td / "afile"
        f.write_bytes(b"x")
        check("目标不可写时返回 False 而不是抛异常",
              not R.extract_zip(z, f / "sub", say=lambda *a: None))

    # 长路径前缀
    check("_long_path 加了 \\\\?\\ 前缀",
          R._long_path(Path(r"D:\a\b")).startswith("\\\\?\\"))
    if sys.platform == "win32":
        long_p = Path(r"D:\x") / ("y" * 200) / ("z" * 100)
        check("超长路径也能生成合法字符串", len(R._long_path(long_p)) < 32767)

    # 解压不能再直接调用 extractall（那样没有进度、没有长路径支持）
    src_txt = inspect.getsource(R.extract_zip)
    check("解压已改为自实现循环（有进度）", "progress(" in src_txt and "infolist()" in src_txt)
    check("解压使用长路径辅助", "_long_path(" in src_txt)


def test_writable_preflight():
    """安装前必须先确认目标可写。

    真踩到的坑：用户把安装位置选到一个只读目录，程序解压到一半才抛
    `[WinError 5] 拒绝访问`，磁盘上还留了一堆残渣，错误信息也完全看不出原因。
    """
    print("\n[安装位置预检]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ok = R.check_writable(td / "new" / "deep")
        check("可写目录判定为 ok", ok["ok"], str(ok))

        # 拿一个普通文件当父目录 -> 任何系统都建不出子目录
        f = td / "afile"
        f.write_bytes(b"x")
        bad = R.check_writable(f / "sub")
        check("父路径是文件时判定为不可用", not bad["ok"], str(bad))
        check("给出原因", bool(bad.get("reason")))

        # suggest_dest 必须返回一个真的能写的位置
        s = R.suggest_dest()
        check("suggest_dest 返回可写位置", R.check_writable(s)["ok"], str(s))

        check("free_space 能取到值", R.free_space(td) > 0)

        # 装到不可写的位置：要明确失败，而不是抛异常或留残渣
        src = _fake_runtime(td / "src")
        z = R.pack_single(src, td / "rel", say=lambda *a: None)
        res = R.install(z, f / "sub", say=lambda *a: None)
        check("装到不可写位置时明确失败", not res.get("ok"), str(res))
        check("失败原因标为权限问题", res.get("kind") == "permission", str(res.get("kind")))
        check("没有留下半拉子目录", not (f / "sub").exists())


def test_local_zip_goes_through_install():
    """对话框的「本地文件」必须走 runtime.install()，不能再自建 _local_extract。

    旧实现会在目标目录下建 `<目标>/_local_extract` 暂存再解压：
    多一个失败点，还白占一份磁盘。
    """
    print("\n[本地 zip 直接解压到目标（无暂存目录）]")
    import inspect

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = _fake_runtime(td / "src")
        z = R.pack_single(src, td / "rel", say=lambda *a: None)

        dest = td / "target"
        cfg_file = td / "d" / "data" / "config.json"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        (cfg_file.parent / "voices").mkdir(exist_ok=True)
        real = R.configure
        R.configure = lambda root, say=print, config_path=None: real(root, say=say, config_path=cfg_file)
        try:
            res = R.install(z, dest, say=lambda *a: None)
        finally:
            R.configure = real
        check("本地 zip 安装成功", res.get("ok"), str(res))
        check("文件齐全", R.check_files(dest) == [])
        check("目标下没有 _local_extract 残渣", not (dest / "_local_extract").exists())

    # 静态检查：本地安装那条路径不能再自己搞暂存目录
    from screenreader.ui import runtime_dialog as rd

    fn_src = inspect.getsource(rd.InstallWorker._install_local)
    check("_install_local 委托给 runtime.install", "R.install(" in fn_src)
    check("_install_local 不再自己建暂存目录/删目录",
          "rmtree(" not in fn_src and "mkdir(" not in fn_src)
    check("对话框不再 import 用不到的 zipfile 处理",
          "zipfile" not in inspect.getsource(rd).split("class InstallWorker")[0])


def test_location_diagnostics():
    """不能只回一句"拒绝访问"——要说清楚为什么、怎么办。"""
    import os

    print("\n[安装位置诊断]")
    check("is_admin 返回布尔", isinstance(R.is_admin(), bool))

    # Program Files 要有专门的解释
    # 注意用 classify_location：describe_location 会先看进程完整性级别，
    # 在沙箱/CI 里那一段总是先命中，就测不到这里的分支了。
    pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "X" / "gsv"
    why = R.classify_location(pf)
    check("Program Files 的解释提到管理员", "管理员" in why or "Program Files" in why, why[:60])

    # 系统目录
    win = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "gsv"
    why = R.classify_location(win)
    check("系统目录被识别出来", "系统目录" in why, why[:60])

    # 盘根
    why = R.classify_location(Path("D:/"))
    check("盘根有专门解释", "根目录" in why, why[:60])

    # 普通目录也要给可操作的建议
    with tempfile.TemporaryDirectory() as td:
        why = R.classify_location(Path(td) / "sub")
        check("普通位置给出可操作建议",
              "新建一个文件夹" in why or "权限" in why, why[:80])

    # describe_location 在低完整性下要优先说完整性，而不是甩 ACL 的锅
    if R.integrity_is_low():
        check("低完整性时优先提示完整性",
              "完整性" in R.describe_location(Path("D:/whatever")),
              R.describe_location(Path("D:/whatever"))[:60])
        check("integrity_is_low 与 token_info 一致", R.integrity_is_low() is True)
    else:
        check("非低完整性时走位置分类",
              R.describe_location(Path(td) / "x") == R.classify_location(Path(td) / "x"))

    # suggest_dest 必须能写，且优先选空间够的
    s = R.suggest_dest()
    check("suggest_dest 返回可写位置", R.check_writable(s)["ok"], str(s))
    check("suggest_dest 不是盘根", s.parent != s, str(s))
    # 要求很大空间时，也应返回一个可写的（哪怕不够大）
    s2 = R.suggest_dest(min_free_gb=99999)
    check("空间都不够时仍返回可写位置", R.check_writable(s2)["ok"], str(s2))


def test_proc_hidden():
    """GUI 程序里 spawn 子进程必须隐藏窗口，否则会闪黑窗。

    实测症状：读屏台点「开始识别」后**每 10 秒闪一次黑窗**——
    显存看门狗每 10 秒调一次 nvidia-smi，而它是控制台程序。
    """
    print("\n[子进程必须隐藏窗口]")
    from screenreader import proc as P

    r = P.run_hidden([sys.executable, "-c", "print('hello-hidden')"], timeout=60)
    check("run_hidden 能跑通", r.returncode == 0, str(r.returncode))
    check("能拿到 stdout", "hello-hidden" in (r.stdout or ""), repr(r.stdout)[:60])

    if sys.platform == "win32":
        kw = P._hidden_kwargs()
        check("带上了 CREATE_NO_WINDOW",
              kw.get("creationflags") == P.CREATE_NO_WINDOW, str(kw.get("creationflags")))
        check("带上了 STARTUPINFO 隐藏窗口",
              "startupinfo" in kw and kw["startupinfo"].wShowWindow == 0)
        # 看门狗与 token 检测都必须走隐藏路径
        import inspect

        from screenreader import watchdog

        check("watchdog 用 run_hidden 调 nvidia-smi",
              "run_hidden" in inspect.getsource(watchdog.query_gpus))
        check("runtime.token_info 用 run_hidden 调 whoami",
              "run_hidden" in inspect.getsource(R.token_info))
    else:
        check("非 Windows 平台跳过创建标志检查", True)


def test_tts_service_helpers():
    print("\n[TTS 服务辅助函数]")
    from screenreader.config import AppConfig

    cfg = AppConfig()
    cfg.tts.api_url = "http://127.0.0.1:19880"
    check("service_port 解析正确", R.service_port(cfg) == 19880, str(R.service_port(cfg)))
    cfg.tts.api_url = "http://127.0.0.1:9880"
    check("默认端口", R.service_port(cfg) == 9880)

    # 一个没人监听的端口
    check("未监听时 service_ready 为 False", not R.service_ready(19999, timeout=0.5))

    # 起一个真实的监听端口
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        check("监听后 service_ready 为 True", R.service_ready(port, timeout=1.0))
    finally:
        srv.close()

    # find_infer_config：没有声库时返回 None
    with tempfile.TemporaryDirectory() as td:
        check("没有声库时返回 None", R.find_infer_config(cfg, Path(td)) is None)

    # start_tts_service 在运行时缺失时要优雅失败（不能抛异常）
    cfg2 = AppConfig()
    cfg2.tts.runtime_root = str(Path(tempfile.gettempdir()) / "__no_such_runtime__")
    cfg2.tts.api_url = "http://127.0.0.1:19999"
    res = R.start_tts_service(cfg2, say=lambda *a: None)
    check("运行时缺失时优雅失败", not res.get("ok"), str(res))
    check("给出原因", bool(res.get("message")))


def test_voice_install_generates_infer_config():
    """装完声库包必须有 tts_infer.yaml，否则 TTS 服务起不来。

    真踩到的坑：声库包**不带** tts_infer.yaml（它是本机生成的），
    而 install() 也没补 —— 结果目标机上"装好了声库，但每句合成都失败"，
    状态栏只丢一句"还没有安装任何声库（voices/ 下没有 tts_infer.yaml）"。
    """
    print("\n[声库包安装后必须生成推理配置]")
    import inspect

    from screenreader.config import AppConfig
    from screenreader.voice import VoiceStore

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        store, _ = _mkvoice(td)
        packz = store.pack("v1", td / "packs", say=lambda *a: None)

        # 包本身不该带 tts_infer.yaml（带的是打包者机器的路径，没意义）
        with zipfile.ZipFile(packz) as z:
            check("包里不含 tts_infer.yaml",
                  not any(n.endswith("tts_infer.yaml") for n in z.namelist()))

        store2 = VoiceStore(td / "target" / "voices")
        prof = store2.install(packz, say=lambda *a: None)
        d2 = store2.dir_of("v1")
        check("安装后生成了 tts_infer.yaml", (d2 / "tts_infer.yaml").exists())
        if (d2 / "tts_infer.yaml").exists():
            txt = (d2 / "tts_infer.yaml").read_text(encoding="utf-8")
            check("配置里有 custom 段", "custom:" in txt)
            check("配置指向包内权重", "weights" in txt)
            check("配置里有 version", "version:" in txt)

        # ensure_voice_configs 必须能把配置指到真正的运行时
        rt = td / "gsv-runtime"
        (rt / "runtime").mkdir(parents=True)
        (rt / "runtime" / "python.exe").write_bytes(b"x")
        cfg = AppConfig()
        cfg.tts.runtime_root = str(rt)
        cfg.tts.voice = "v1"
        n = R.ensure_voice_configs(cfg, voices_root=td / "target" / "voices", say=lambda *a: None)
        check("ensure_voice_configs 处理了声库", n == 1, str(n))
        txt2 = (d2 / "tts_infer.yaml").read_text(encoding="utf-8")
        check("配置已指向新运行时", str(rt) in txt2, txt2.splitlines()[2] if len(txt2.splitlines()) > 2 else "")

        # 没有声库时不能炸
        empty = td / "empty-voices"
        check("没有声库时返回 0",
              R.ensure_voice_configs(cfg, voices_root=empty, say=lambda *a: None) == 0)

    # 静态检查：install() 里必须调用 write_api_config
    src = inspect.getsource(VoiceStore.install)
    check("install() 会生成推理配置", "write_api_config" in src)


def test_start_is_not_blocking():
    """点「开始识别」不能在界面线程上做耗时初始化。

    实测症状：点击启动后窗口"未响应"约 10 秒 ——
    因为 start() 同步调了 _ensure_ocr()，而它会跑 OCR 线程数标定。
    """
    print("\n[引擎启动不能阻塞界面线程]")
    import inspect

    from screenreader.worker import BoardEngine

    src = inspect.getsource(BoardEngine.start)
    # 只看代码行（注释里可能会引用这个函数名做说明）
    code_lines = [l.strip() for l in src.splitlines()
                  if l.strip() and not l.strip().startswith("#")]
    check("start() 不再同步初始化 OCR",
          not any(l.startswith("self._ensure_ocr(") for l in code_lines),
          str([l for l in code_lines if "_ensure_ocr" in l])[:80])
    check("start() 仍有说明为何不能同步初始化",
          "未响应" in src or "界面线程" in src)

    # 检测线程里必须仍然会懒加载 OCR，否则永远不会初始化
    poll = inspect.getsource(BoardEngine._poll_region)
    check("_poll_region 懒加载 OCR", "_ensure_ocr" in poll)


def test_human_eta():
    """剩余时间不足 1 分钟要按秒显示。

    用户反馈：进度条写"剩余约 0 分钟"，根本不知道还要等 5 秒还是 55 秒。
    """
    print("\n[剩余时间显示]")
    check("3 秒", R.human_eta(3) == "剩余约 3 秒", R.human_eta(3))
    check("45 秒", R.human_eta(45) == "剩余约 45 秒", R.human_eta(45))
    check("59 秒仍是秒", R.human_eta(59).endswith("秒"), R.human_eta(59))
    check("60 秒变分钟", R.human_eta(60) == "剩余约 1 分钟", R.human_eta(60))
    check("10 分钟", R.human_eta(600) == "剩余约 10 分钟", R.human_eta(600))
    check("超过 1 小时", "小时" in R.human_eta(7200), R.human_eta(7200))
    check("超过一天不显示", R.human_eta(200000) == "", R.human_eta(200000))
    check("负数不显示", R.human_eta(-5) == "")
    check("0 秒显示 1 秒（不显示 0）", R.human_eta(0) == "剩余约 1 秒", R.human_eta(0))


def test_install_any_shapes():
    """声库来源的三种形态都要能直接吃下去。"""
    print("\n[声库来源：zip / 声库文件夹 / 装着多个的目录]")
    import shutil

    from screenreader.voice import VoiceStore

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        store, _ = _mkvoice(td, "v1")
        packz = store.pack("v1", td / "packs", say=lambda *a: None)

        # 1. zip
        s1 = VoiceStore(td / "a" / "voices")
        check("zip 能装", s1.install_any(packz, say=lambda *a: None) == ["v1"])
        check("装完有推理配置", (s1.dir_of("v1") / "tts_infer.yaml").exists())

        # 2. 已解压的声库文件夹
        s2 = VoiceStore(td / "b" / "voices")
        check("声库文件夹能装", s2.install_any(s2.root / ".." / ".." / "a" / "voices" / "v1",
                                          say=lambda *a: None) == ["v1"])

        # 3. 一个装着多个声库的上级目录（zip + 文件夹混合）
        parent = td / "batch"
        parent.mkdir()
        shutil.copyfile(packz, parent / "one.voicepack.zip")
        shutil.copytree(td / "a" / "voices" / "v1", parent / "two")
        s3 = VoiceStore(td / "c" / "voices")
        got = s3.install_any(parent, overwrite=True, say=lambda *a: None)
        check("批量目录能装出多个", len(got) >= 1, str(got))

        # 4. 什么都不含的目录 -> 明确报错
        empty = td / "nothing"
        empty.mkdir()
        try:
            VoiceStore(td / "d" / "voices").install_any(empty, say=lambda *a: None)
            check("空目录明确报错", False)
        except ValueError as e:
            check("空目录明确报错", "没找到声库" in str(e), str(e)[:60])

        # 5. 不存在的路径
        try:
            VoiceStore(td / "e" / "voices").install_any(td / "不存在", say=lambda *a: None)
            check("路径不存在时报错", False)
        except FileNotFoundError:
            check("路径不存在时报错", True)


# --------------------------------------------------------------------------- #

def test_version_consistency():
    """版本号必须两处一致，否则用户敲 `--version` 会看到驴唇不对马嘴的数字。

    实际踩过：`packaging/installer.iss` 已经是 0.8.3，
    但 `src/screenreader/__init__.py` 还停在 0.2.0 —— 打包出来的 exe
    一敲 `--version` 打印 0.2.0。
    """
    import re
    from pathlib import Path

    print("\n[版本号一致性]")
    root = Path(__file__).resolve().parents[1]
    init = (root / "src" / "screenreader" / "__init__.py").read_text(encoding="utf-8")
    m1 = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    check("__init__.py 里有 __version__", m1 is not None)
    ver_pkg = m1.group(1) if m1 else "?"

    iss = (root / "packaging" / "installer.iss").read_text(encoding="utf-8")
    m2 = re.search(r'MyAppVersion\s+"([^"]+)"', iss)
    check("installer.iss 里有 MyAppVersion", m2 is not None)
    ver_iss = m2.group(1) if m2 else "?"

    check("[包] 与 [安装包] 版本一致（%s vs %s）" % (ver_pkg, ver_iss),
          ver_pkg == ver_iss, "%s != %s" % (ver_pkg, ver_iss))
    check("版本号形如 x.y.z", bool(re.fullmatch(r"\d+\.\d+\.\d+", ver_pkg)), ver_pkg)

def main() -> int:
    for fn in (
        test_pack_and_install,
        test_pack_rejects_incomplete,
        test_install_rejects_bad_zip,
        test_install_blocks_path_traversal,
        test_install_rejects_incomplete_pack,
        test_manifest_roundtrip,
        test_runtime_status,
        test_repair_users_pth,
        test_install_from_dir_detects_nested,
        test_runtime_manifest_from_dict,
        test_end_to_end_download,
        test_install_cancellation,
        test_launcher_bat_is_ascii_safe,
        test_single_zip_install,
        test_install_autodetect,
        test_unnest,
        test_url_auth_parsing,
        test_authenticated_download,
        test_resume_when_server_ignores_range,
        test_extract_zip_robustness,
        test_writable_preflight,
        test_location_diagnostics,
        test_local_zip_goes_through_install,
        test_proc_hidden,
        test_tts_service_helpers,
        test_voice_install_generates_infer_config,
        test_start_is_not_blocking,
        test_human_eta,
        test_install_any_shapes,
        test_version_consistency,
    ):
        fn()
    print("\n" + "=" * 60)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    for f in FAIL:
        print(f"  FAIL: {f}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
