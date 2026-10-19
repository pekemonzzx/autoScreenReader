"""测试声库的安装位置：必须跟着「安装位置」走，而且装完程序要能找到。

用户反馈：首次运行向导里选好了安装目录和声库包，声库却没解压到
`安装目录/voices/` 下。

根因：声库目录写死 `<程序目录>/voices`，用户把 9GB 运行时换到 D 盘之后，
声库还是悄悄装回程序目录。用户翻自己选的安装目录当然找不到。

这个测试守住两点：
  1. 声库装到「运行时旁边」（用户选的目录旁边）
  2. 装完之后 `voices_root()` 必须能找回来 —— 否则装对了也白搭
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"    {detail}" if detail else ""))


def _with_fake_app_dir(base: Path):
    """把 _app_data_base 指到临时目录，返回恢复函数。"""
    import screenreader.config as C

    real = C._app_data_base
    C._app_data_base = lambda: base
    return lambda: setattr(C, "_app_data_base", real)


def test_voices_dir_for_runtime() -> None:
    print("\n[声库该装到哪]")
    from screenreader.config import AppConfig

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "程序目录"
        base.mkdir(parents=True)
        restore = _with_fake_app_dir(base)
        try:
            cfg = AppConfig()

            # 1) 运行时就在程序目录里 -> 声库也在程序目录
            got = cfg.voices_dir_for_runtime(base / "gsv")
            check("运行时在程序目录里 -> 声库在程序目录",
                  got == base / "voices", str(got))

            # 2) 运行时在别的盘 -> 声库跟着过去（这就是用户要的行为）
            other = Path(td) / "D盘" / "AutoScreenReader" / "gsv"
            got = cfg.voices_dir_for_runtime(other)
            check("运行时在别处 -> 声库在运行时旁边",
                  got == other.parent / "voices", str(got))

            # 3) 传进来的直接是父目录（不叫 gsv）时也不能算错
            got = cfg.voices_dir_for_runtime(Path(td) / "别处")
            check("非 gsv 结尾也合理",
                  got == Path(td) / "别处" / "voices", str(got))

            # 4) 空值 -> 程序目录
            check("空值时回落程序目录",
                  cfg.voices_dir_for_runtime(None) == base / "voices")
        finally:
            restore()


def test_voices_root_finds_both() -> None:
    """装完之后必须能找回来 —— 装对了找不到等于没装。"""
    print("\n[装完之后能不能找到]")
    from screenreader.config import AppConfig
    from screenreader.voice import VoiceStore

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "程序目录"
        base.mkdir(parents=True)
        other = Path(td) / "D盘" / "AutoScreenReader" / "gsv"
        other.mkdir(parents=True)
        restore = _with_fake_app_dir(base)
        try:
            cfg = AppConfig()
            cfg.tts.runtime_root = str(other)

            # 还没装声库时，回落到程序目录
            check("没声库时回落程序目录",
                  cfg.voices_root() == base / "voices", str(cfg.voices_root()))

            # 向导算出来的落点
            target = cfg.voices_dir_for_runtime(cfg.tts.runtime_root)
            check("落点在运行时旁边", target == other.parent / "voices", str(target))

            # 真装一个（用最小的假声库，不碰真包）
            target.mkdir(parents=True, exist_ok=True)
            d = target / "fake"
            d.mkdir()
            (d / "voice.json").write_text(
                '{"name":"fake","display":"假","version":"v2ProPlus"}', encoding="utf-8")

            check("装完 voices_root 能找回它",
                  cfg.voices_root() == target, str(cfg.voices_root()))
            check("VoiceStore 能列出",
                  VoiceStore(cfg.voices_root()).list_names() == ["fake"],
                  str(VoiceStore(cfg.voices_root()).list_names()))

            # 两个候选目录都要在列表里（旧安装也要能用）
            roots = cfg.voices_roots()
            check("候选目录含程序目录", base / "voices" in roots, str(roots))
            check("候选目录含运行时旁边", target in roots, str(roots))
        finally:
            restore()


def test_install_then_find_roundtrip() -> None:
    """完整闭环：向导安装 -> 程序找到 -> 推理配置指向正确的运行时。"""
    print("\n[安装 -> 找到 的完整闭环]")
    from screenreader.config import AppConfig
    from screenreader.voice import VoiceStore

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "程序目录"
        base.mkdir(parents=True)
        other = Path(td) / "D盘" / "X" / "gsv"
        (other / "runtime").mkdir(parents=True)
        (other / "runtime" / "python.exe").write_bytes(b"MZ")
        restore = _with_fake_app_dir(base)
        try:
            cfg = AppConfig()
            cfg.tts.runtime_root = str(other)
            target = cfg.voices_dir_for_runtime(cfg.tts.runtime_root)
            target.mkdir(parents=True, exist_ok=True)

            d = target / "v"
            d.mkdir()
            (d / "voice.json").write_text(
                '{"name":"v","display":"V","version":"v2ProPlus",'
                '"ref_audio":"ref.wav","prompt_text":"测试"}', encoding="utf-8")
            (d / "weights").mkdir()
            (d / "weights" / "a.pth").write_bytes(b"x")
            (d / "weights" / "b.ckpt").write_bytes(b"y")

            cfg.tts.voice = "v"
            prof = VoiceStore(target).load("v")
            prof.write_api_config(d, runtime_root=other)

            yml = (d / "tts_infer.yaml").read_text(encoding="utf-8")
            check("推理配置指向了真正的运行时",
                  str(other).replace("\\", "\\\\") in yml or str(other) in yml,
                  yml.splitlines()[-1] if yml.splitlines() else "")
            check("程序能找到声库", cfg.voices_root() == target, str(cfg.voices_root()))
        finally:
            restore()


def test_voice_only_install() -> None:
    """运行时早就装好了，用户这次只想加声库 —— 必须能装成。

    用户实测踩到：第一次装了运行时；第二次想加声库，填好声库包点了安装，
    对话框却弹「还没填来源」直接返回（因为它一律要求填运行时来源），
    于是声库根本没解压 —— 看起来就是"没有自动解压"。
    """
    print("\n[只装声库（运行时已就绪）]")
    import json

    from screenreader import runtime as R
    from screenreader.voice import VoiceStore

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "程序目录"
        base.mkdir(parents=True)
        (base / "data").mkdir()
        (base / "voices").mkdir()
        gsv = base / "gsv"
        (gsv / "runtime").mkdir(parents=True)
        (gsv / "runtime" / "python.exe").write_bytes(b"MZ")
        (gsv / R.MARKER_NAME).write_text(json.dumps({"version": "0.6.2"}), encoding="utf-8")

        restore = _with_fake_app_dir(base)
        os.environ["SCREENREADER_CONFIG"] = str(base / "data" / "config.json")
        try:
            st = R.status(gsv)
            check("文件不全时 status 判为未安装（正常行为）",
                  st.get("installed") is False, str(st.get("missing"))[:60])
            # 关键：判定"运行时在位"要比 status 宽松 —— 只要解释器在就行，
            # 否则裁剪版/老版本运行时会让用户连声库都装不了
            check("但解释器在就算运行时在位",
                  (gsv / "runtime" / "python.exe").exists())

            from screenreader.ui.runtime_dialog import InstallWorker

            w = InstallWorker(gsv, "voice", "",
                              voice_src=str(ROOT / "voices" / "_packs" / "voice02hq.voicepack.zip"))
            done = {}
            w.finished_ok.connect(lambda r: done.update(r or {}))
            w.failed.connect(lambda m: done.update({"__fail__": m}))
            w.run()

            check("只装声库不报错", "__fail__" not in done, str(done.get("__fail__")))
            check("标记了跳过运行时", done.get("skipped_runtime") is True)
            vr = done.get("voice") or {}
            check("声库安装成功", vr.get("ok") is True, str(vr))

            d = base / "voices" / "voice02hq"
            check("声库落在 voices\\ 下", d.is_dir(), str(d))
            check("档案文件齐全",
                  (d / "voice.json").exists() and (d / "ref.wav").exists(),
                  str(sorted(p.name for p in d.iterdir())))
            check("权重都在",
                  len(list((d / "weights").glob("*"))) == 2,
                  str([p.name for p in (d / "weights").glob("*")]))
            check("自动生成了推理配置", (d / "tts_infer.yaml").exists())
            check("程序能找回它",
                  VoiceStore(base / "voices").list_names() == ["voice02hq"],
                  str(VoiceStore(base / "voices").list_names()))
        finally:
            restore()
            os.environ.pop("SCREENREADER_CONFIG", None)


def test_voice_installs_even_if_runtime_fails() -> None:
    """声库安装不能绑在运行时安装成功之上 —— 两者本来就独立。"""
    print("\n[运行时失败也照样装声库]")
    import zipfile

    from screenreader.ui.runtime_dialog import InstallWorker

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "程序目录"
        base.mkdir(parents=True)
        (base / "data").mkdir()
        (base / "voices").mkdir()
        restore = _with_fake_app_dir(base)
        os.environ["SCREENREADER_CONFIG"] = str(base / "data" / "config.json")
        try:
            # 故意做一个"缺文件"的假运行时包，install() 会因为校验不过返回 ok=False
            bad = Path(td) / "bad.zip"
            with zipfile.ZipFile(bad, "w") as z:
                z.writestr("runtime/python.exe", "MZ")
            dest = Path(td) / "其它盘" / "gsv"
            dest.parent.mkdir(parents=True)

            w = InstallWorker(dest, "local", str(bad),
                              voice_src=str(ROOT / "voices" / "_packs" / "voice02hq.voicepack.zip"))
            done = {}
            w.finished_ok.connect(lambda r: done.update(r or {}))
            w.failed.connect(lambda m: done.update({"__fail__": m}))
            w.run()

            check("运行时确实失败了", "__fail__" in done, str(done.get("__fail__"))[:80])
            # 关键：声库仍然要装成功
            installed = list((base / "voices").glob("*/voice.json")) + \
                        list((dest.parent / "voices").glob("*/voice.json"))
            check("运行时失败但声库照样装上了", bool(installed),
                  str([str(p) for p in installed]))
        finally:
            restore()
            os.environ.pop("SCREENREADER_CONFIG", None)


def main() -> int:
    for fn in (test_voices_dir_for_runtime, test_voices_root_finds_both,
               test_install_then_find_roundtrip, test_voice_only_install,
               test_voice_installs_even_if_runtime_fails):
        fn()
    print("\n" + "=" * 60)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    for f in FAIL:
        print(f"  FAIL: {f}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
