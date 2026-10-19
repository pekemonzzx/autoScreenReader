"""验证声库包能不能被正确解压安装（模拟安装脚本那条路径）。

用户反馈「安装脚本没有把声库文件正确解压」，这里用**真的 voicepack.zip**
走一遍 VoiceStore.install()，逐项核对产物是否完整、权重是否能对上。
"""

import hashlib
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screenreader.voice import VoiceStore  # noqa: E402

PACK = ROOT / "voices" / "_packs" / "voice02hq.voicepack.zip"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


print("=" * 74)
print(f"声库包：{PACK.name}  ({PACK.stat().st_size / 1e6:.1f} MB)")
print("=" * 74)

# 1) 先看包里有什么
with zipfile.ZipFile(PACK) as z:
    names = z.namelist()
    print(f"\n  包内 {len(names)} 个条目：")
    for n in sorted(names)[:20]:
        print(f"    {n}")
    if len(names) > 20:
        print(f"    ... 还有 {len(names) - 20} 个")
    bad = [n for n in names if n.startswith("/") or ".." in n]
    print(f"  危险路径（绝对/上跳）：{bad if bad else '无'}")

# 2) 解压安装
with tempfile.TemporaryDirectory() as td:
    voices = Path(td) / "voices"
    voices.mkdir()
    store = VoiceStore(voices)
    logs = []
    print("\n  开始安装……")
    names_out = store.install(PACK, overwrite=True, say=lambda m: logs.append(str(m)))
    print(f"  安装结果：{names_out}")

    d = store.dir_of("voice02hq")
    print(f"\n  解压到：{d}")
    got = sorted(p for p in d.rglob("*") if p.is_file())
    for p in got:
        print(f"    {p.relative_to(d)}  {p.stat().st_size / 1e6:.2f} MB")

    # 3) 逐项核对
    print("\n  核对：")
    must = ["voice.json", "ref.wav"]
    for m in must:
        print(f"    {m:<22} {'OK' if (d / m).exists() else '缺失'}")

    wdir = d / "weights"
    wfiles = sorted(wdir.iterdir()) if wdir.is_dir() else []
    print(f"    weights/ 里 {len(wfiles)} 个权重：")
    for w in wfiles:
        print(f"      {w.name}  {w.stat().st_size / 1e6:.1f} MB  sha={sha(w)}")

    # 权重要和源目录里的一致
    src = ROOT / "voices" / "voice02hq" / "weights"
    if src.is_dir():
        print("\n  与源目录对比：")
        for s in sorted(src.iterdir()):
            t = wdir / s.name
            if not t.exists():
                print(f"    {s.name}: 解压后不见了 !!")
            elif sha(s) == sha(t):
                print(f"    {s.name}: 一致 OK")
            else:
                print(f"    {s.name}: **内容不一致** !!")

    # 4) 推理配置（包里不带，安装时要自动生成）
    print("\n  推理配置：")
    infer = d / "tts_infer.yaml"
    print(f"    tts_infer.yaml 存在：{infer.exists()}")
    if infer.exists():
        txt = infer.read_text(encoding="utf-8")
        for key in ("vits_weights_path", "t2s_weights_path", "version", "device"):
            line = next((l.strip() for l in txt.splitlines() if l.strip().startswith(key)), "")
            print(f"      {line}")

    # 5) 能不能正常读出声库档案
    print("\n  档案可读性：")
    try:
        prof = store.load("voice02hq")
        print(f"    名字={prof.name}  显示={prof.display}  版本={prof.version}")
        gpt, sovits = prof.weight_paths(d)
        print(f"    GPT 权重：{gpt}")
        print(f"    SoVITS 权重：{sovits}")
        print(f"    两个权重都在：{bool(gpt and gpt.exists()) and bool(sovits and sovits.exists())}")
    except Exception as e:
        print(f"    !! 读档案失败：{e}")

    print("\n  安装日志：")
    for l in logs[-8:]:
        print(f"    {l}")
print("=" * 74)
