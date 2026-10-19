"""声库档案测试（纯逻辑，不需要 GPU/TTS 服务）。

运行：
    .\\.venv\\Scripts\\python tests\\test_voice.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screenreader.config import AppConfig  # noqa: E402
from screenreader.voice import (  # noqa: E402
    DEFAULT_GSV_ROOT,
    SUPPORTED_VERSIONS,
    TrainConfig,
    VoiceProfile,
    VoiceStore,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def expect_eq(name, got, want):
    check(name, got == want, f"得到 {got!r}，期望 {want!r}")


def test_create_and_load():
    print("\n[创建与读取]")
    with tempfile.TemporaryDirectory() as td:
        store = VoiceStore(Path(td))
        p = store.create("v1", display_name="测试音色", version="v2ProPlus", gsv_root=DEFAULT_GSV_ROOT)
        expect_eq("名字", p.name, "v1")
        expect_eq("显示名", p.display, "测试音色")
        check("目录已建", (Path(td) / "v1" / "voice.json").exists())
        check("README 已生成", (Path(td) / "v1" / "README.md").exists())
        expect_eq("list_names", store.list_names(), ["v1"])

        back = store.load("v1")
        expect_eq("回读版本", back.version, "v2ProPlus")
        expect_eq("回读显示名", back.display, "测试音色")


def test_version_validation():
    print("\n[版本校验]")
    with tempfile.TemporaryDirectory() as td:
        store = VoiceStore(Path(td))
        for v in SUPPORTED_VERSIONS:
            try:
                store.create(f"ok_{v}", version=v)
                check(f"接受 {v}", True)
            except Exception as e:
                check(f"接受 {v}", False, str(e))
        try:
            store.create("bad", version="v9")
            check("拒绝未知版本", False)
        except ValueError as e:
            check("拒绝未知版本", "v9" in str(e))
        try:
            store.create("ok_v2ProPlus")
            check("重名拒绝", False)
        except FileExistsError:
            check("重名拒绝", True)


def test_relative_paths():
    print("\n[相对路径解析]")
    with tempfile.TemporaryDirectory() as td:
        store = VoiceStore(Path(td))
        p = store.create("v1")
        d = store.dir_of("v1")
        (d / "ref.wav").write_bytes(b"RIFF")
        p.ref_audio = "ref.wav"
        p.train.list_path = "train.list"
        p.train.wav_dir = "slices"
        store.save(p)

        back = store.load("v1")
        kw = back.to_tts_kwargs(d)
        expect_eq("ref 解析为档案内绝对路径", Path(kw["ref_audio"]), d / "ref.wav")

        lst, wav = back.train_paths(d)
        expect_eq("train.list 解析到档案内", lst, d / "train.list")
        expect_eq("wav_dir 相对路径也解析", wav, d / "slices")

        # 绝对路径原样保留
        back.train.wav_dir = r"D:\some\abs\dir"
        _, wav2 = back.train_paths(d)
        expect_eq("绝对路径原样", wav2, Path(r"D:\some\abs\dir"))


def test_train_yaml_roundtrip():
    print("\n[train.yaml 读写]")
    with tempfile.TemporaryDirectory() as td:
        store = VoiceStore(Path(td))
        p = store.create("v1")
        p.train.s2_batch = 4
        p.train.s1_epoch = 20
        p.train.list_path = "train.list"
        store.save(p)
        y = store.write_train_yaml(p)
        check("train.yaml 已生成", y.exists())

        raw = store.read_train_yaml("v1")
        expect_eq("yaml 里 batch", raw["s2_batch"], 4)
        expect_eq("yaml 里 epoch", raw["s1_epoch"], 20)

        # 合并覆盖
        merged = TrainConfig.from_dict({**p.train.__dict__, **raw})
        expect_eq("合并后仍是 4", merged.s2_batch, 4)

        # 未知键容错
        m2 = TrainConfig.from_dict({"s2_batch": 3, "不存在": 1})
        expect_eq("未知键忽略", m2.s2_batch, 3)


def test_api_config_generation():
    print("\n[推理配置生成]")
    with tempfile.TemporaryDirectory() as td:
        store = VoiceStore(Path(td))
        p = store.create("v1", version="v2ProPlus")
        p.gpt_weights = "GPT_weights_v2ProPlus/v1-e12.ckpt"
        p.sovits_weights = "SoVITS_weights_v2ProPlus/v1_e8_s248.pth"
        store.save(p)
        y = p.write_api_config(store.dir_of("v1"))
        txt = y.read_text(encoding="utf-8")
        check("yaml 含 version", "version: v2ProPlus" in txt)
        check("yaml 指向 GPT 权重", "v1-e12.ckpt" in txt)
        check("yaml 指向 SoVITS 权重", "v1_e8_s248.pth" in txt)
        check("yaml 用绝对路径", str(DEFAULT_GSV_ROOT) in txt)
        cmd = p.api_launch_command(store.dir_of("v1"))
        check("启动命令含 api_v2", "api_v2.py" in cmd)
        check("启动命令含本档案的 yaml", "tts_infer.yaml" in cmd)


def test_apply_voice_to_config():
    print("\n[接入全局配置]")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        vroot = root / "voices"
        store = VoiceStore(vroot)
        p = store.create("v1")
        d = store.dir_of("v1")
        (d / "ref.wav").write_bytes(b"RIFF")
        p.ref_audio = "ref.wav"
        p.prompt_text = "这是参考文本"
        p.speed_factor = 1.05
        p.api_url = "http://127.0.0.1:9999"
        store.save(p)

        cfg = AppConfig()
        cfg.tts.ref_audio = r"D:\old\stale.wav"
        cfg.tts.prompt_text = "旧的文本"
        cfg.tts.voice = "v1"
        got = cfg.apply_voice(vroot)
        expect_eq("返回生效的声库名", got, "v1")
        expect_eq("ref_audio 被档案覆盖", Path(cfg.tts.ref_audio), d / "ref.wav")
        expect_eq("prompt_text 被档案覆盖", cfg.tts.prompt_text, "这是参考文本")
        expect_eq("speed_factor 覆盖", cfg.tts.speed_factor, 1.05)
        expect_eq("api_url 覆盖", cfg.tts.api_url, "http://127.0.0.1:9999")

        # 没指定声库时不动（关掉自动选）
        cfg2 = AppConfig()
        cfg2.tts.ref_audio = "keep.wav"
        expect_eq("关掉 auto_select 时返回 None",
                  cfg2.apply_voice(vroot, auto_select=False), None)
        expect_eq("关掉 auto_select 时不改动", cfg2.tts.ref_audio, "keep.wav")

        # **默认会自动选第一个可用声库** —— 这是修那个大坑的关键：
        # 目标机上用户把声库包解压进 voices\ 了，但 config 里 tts.voice 是空的，
        # 界面下拉框显示着音色、ref_audio 却为空，每句合成都报"没有配置参考音频"。
        cfg4 = AppConfig()
        cfg4.tts.voice = ""
        cfg4.tts.ref_audio = ""
        expect_eq("没指定时自动选第一个", cfg4.apply_voice(vroot), "v1")
        expect_eq("自动选中后 ref_audio 被填上", Path(cfg4.tts.ref_audio), d / "ref.wav")
        expect_eq("自动选中后 voice 字段也写上", cfg4.tts.voice, "v1")

        # 配置里写了个不存在的声库 -> 也要自动另选，而不是留空
        cfg5 = AppConfig()
        cfg5.tts.voice = "早就删掉的声库"
        cfg5.tts.ref_audio = ""
        expect_eq("声库名失效时自动另选", cfg5.apply_voice(vroot), "v1")
        expect_eq("另选后 ref_audio 有效", Path(cfg5.tts.ref_audio), d / "ref.wav")

        # 声库不存在时容错
        cfg3 = AppConfig()
        cfg3.tts.voice = "不存在"
        expect_eq("没有可用声库时返回 None",
                  cfg3.apply_voice(Path(td) / "空的" / "voices"), None)


def test_profile_json_schema():
    print("\n[voice.json 结构]")
    with tempfile.TemporaryDirectory() as td:
        store = VoiceStore(Path(td))
        p = store.create("v1", source="B站录播")
        raw = json.loads((store.dir_of("v1") / "voice.json").read_text(encoding="utf-8"))
        for k in ("name", "version", "ref_audio", "prompt_text", "gpt_weights",
                  "sovits_weights", "train", "created", "source"):
            check(f"含字段 {k}", k in raw)
        expect_eq("source 写入", raw["source"], "B站录播")
        check("train 是嵌套对象", isinstance(raw["train"], dict))
        check("train 里有 batch", "s2_batch" in raw["train"] or "s2_batch" in raw)


def test_delete():
    print("\n[删除]")
    with tempfile.TemporaryDirectory() as td:
        store = VoiceStore(Path(td))
        store.create("v1")
        check("删前存在", store.exists("v1"))
        check("删除返回 True", store.delete("v1"))
        check("删后不存在", not store.exists("v1"))
        check("删不存在的返回 False", not store.delete("v1"))


# --------------------------------------------------------------------------- #
def main() -> int:
    for fn in (
        test_create_and_load,
        test_version_validation,
        test_relative_paths,
        test_train_yaml_roundtrip,
        test_api_config_generation,
        test_apply_voice_to_config,
        test_profile_json_schema,
        test_delete,
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
