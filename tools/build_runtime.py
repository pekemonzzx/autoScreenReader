"""把 GPT-SoVITS 整合包裁剪成「推理最小集」，供安装包使用。

整合包原样 17.8 GB，其中大部分推理根本用不到：

    torch                       4.05 GB    ← 要用（除非走 ONNX，见 P3）
    GPT_SoVITS/pretrained_models 4.08 GB    ← 只有 1.8 GB 是 v2ProPlus 推理需要的
    tools/asr/models            1.11 GB    ← 只有**训练**标注时用
    tools/uvr5/uvr5_weights     0.70 GB    ← 只有**数据准备**时用
    site-packages 的 gradio/pandas/ctranslate2… ← api_v2 不需要

本脚本按「推理真正会加载的东西」复制出一份最小集，然后**实际启一次 api_v2.py
并合成一句话**来验证——裁剪这种东西不验证等于没做。

用法：
    $env:PYTHONPATH='src'
    .\\.venv\\Scripts\\python tools\\build_runtime.py --dest D:\\build\\gsv-runtime
    .\\.venv\\Scripts\\python tools\\build_runtime.py --dest D:\\build\\gsv-runtime --verify voice01
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screenreader.voice import VoiceProfile, VoiceStore  # noqa: E402

# --------------------------------------------------------------------------- #
# pretrained_models 里 v2ProPlus 推理真正需要的
# 其余（v1/v2/v3/v4 的底模、BigVGAN 声码器）只有训练或旧版本才用
PRETRAINED_KEEP = [
    "chinese-roberta-wwm-ext-large",   # BERT 文本前端
    "chinese-hubert-base",             # CNHuBERT 特征
    "v2Pro",                           # v2ProPlus 的底模（部分代码路径会引用）
    "sv",                              # v2ProPlus 的说话人向量模型
    "fast_langdetect",                 # 中英混排语种检测
    "s1v3.ckpt",                       # GPT 底模（部分代码路径会引用）
]
PRETRAINED_DROP = [
    "gsv-v4-pretrained", "gsv-v2final-pretrained", "models--nvidia--bigvgan_v2_24khz_100band_256x",
    "s2Gv3.pth", "s2G488k.pth", "s2D488k.pth",
    "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
]

# 推理用不到的顶层目录
TOP_DROP = ["Docker", "docs", ".git", "logs", "work", "TEMP", "__pycache__"]

# tools 里只保留推理需要的（其余是 UVR5/ASR/降噪/超分预处理，都是给数据准备和训练用的）
# audio_sr.py + AP_BWE_main 是 v2ProPlus 声码器的音频超分，**推理必需**（漏了会 400）
TOOLS_KEEP = ["i18n", "my_utils.py", "slicer2.py", "assets.py", "audio_sr.py",
              "AP_BWE_main", "__init__.py"]

# site-packages 里确认推理不需要的（按名字前缀删）。
#
# 教训：这个清单必须**保守**。实测删了下面这几类会让"服务能启动、但每个请求都 400"，
# 因为 api_v2 在请求路径上才 import 它们，而且异常被吞成空的 400 响应体：
#   * librosa / numba / llvmlite / sklearn / matplotlib  ← TTS_infer_pack 与 AR 模块要
#   * wordfreq / mecab / ipadic / pyopenjtalk / eunjeon   ← 语种检测的中日韩分词
#   * onnxruntime                                        ← G2PW 中文多音字模型（onnx_api.py）
# 所以只删那些明确与推理无关的（WebUI、ASR、训练、笔记本工具链）。
SITE_DROP = [
    "gradio", "gradio_client",           # WebUI
    "funasr", "ctranslate2", "faster_whisper", "whisper", "modelscope",  # ASR 工具
    "tensorboard",                        # 训练可视化
    "torchvision",                        # 推理用不到
    "cmake",                              # 构建工具
    "IPython", "ipython", "notebook", "jupyter", "nbformat", "nbconvert",
    "paddle", "paddleocr", "paddlex", "openvino",
]


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def copy_item(src: Path, dst: Path, say) -> int:
    if not src.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    n = dir_size(dst)
    say(f"    {src.name:<52} {human(n):>10}")
    return n


def repair_users_pth_in_dest(dest: Path, say=print) -> None:
    """把裁剪包里的 users.pth 改成指向**自己**。

    这一步至关重要：原包的 users.pth 硬编码了打包者的绝对路径，
    直接复制过来会让裁剪后的运行时**仍然去原包找模块**——
    结果是"在本机能跑、拷到目标机就崩"，而且自愈验证还会被它骗过
    （实测：tools/audio_sr 明明没复制，服务却能起来，就是靠这条路径）。
    """
    pth = dest / "runtime" / "Lib" / "site-packages" / "users.pth"
    if not pth.exists():
        return
    subdirs = [
        "GPT_SoVITS/BigVGAN", "tools", "tools/asr", "GPT_SoVITS", "tools/uvr5",
    ]
    lines = [str(dest)]
    for s in subdirs:
        # 只保留裁剪包里真实存在的子目录，避免又指向不存在的路径
        if (dest / s).exists():
            lines.append(f"{dest}/{s}")
    pth.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"    已重写 users.pth 指向裁剪目录（保留 {len(lines) - 1} 个子路径）")


def find_path_leaks(dest: Path, src_root: Path, say=print) -> list:
    """扫描裁剪包里是否还有指向原包的绝对路径。

    这类"隐形引用"是最危险的：它让包在本机测试通过、到目标机才炸。
    """
    old = str(src_root)
    leaks = []
    for f in dest.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".pth", ".py", ".json", ".yaml", ".yml", ".txt", ".cfg", ".ini", ".bat"):
            continue
        if f.stat().st_size > 2_000_000:
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if old in txt:
            n = txt.count(old)
            leaks.append((f.relative_to(dest), n))
    if leaks:
        say(f"    !! 发现 {len(leaks)} 个文件仍引用原包路径 {old}")
        for rel, n in leaks[:15]:
            say(f"       {rel}  （{n} 处）")
    else:
        say("    未发现指向原包的残留引用 ✅")
    return leaks


# --------------------------------------------------------------------------- #
def build(src_root: Path, dest: Path, voice: str, say=print) -> dict:
    if dest.exists():
        say(f"清理旧的 {dest} …")
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    stats = {"copied": 0, "skipped": 0}
    say("=" * 84)
    say(f"裁剪 GPT-SoVITS 推理最小集")
    say(f"  源  {src_root}   （{human(dir_size(src_root))}）")
    say(f"  目标 {dest}")
    say(f"  声库 {voice}")
    say("=" * 84)

    # ---- 1. runtime（Python + 依赖），先整份复制，后面再删 site-packages ----
    say("\n[1] runtime")
    copy_item(src_root / "runtime", dest / "runtime", say)

    # ---- 2. 顶层脚本与配置 ----
    say("\n[2] 顶层脚本 / 配置")
    for name in ("api_v2.py", "config.py", "weight.json", "ffmpeg.exe", "ffprobe.exe"):
        copy_item(src_root / name, dest / name, say)

    # ---- 3. GPT_SoVITS 代码 ----
    say("\n[3] GPT_SoVITS 代码（不含 pretrained_models）")
    gs = dest / "GPT_SoVITS"
    gs.mkdir(parents=True, exist_ok=True)
    for item in (src_root / "GPT_SoVITS").iterdir():
        if item.name == "pretrained_models":
            continue
        if item.name == "__pycache__":
            continue
        copy_item(item, gs / item.name, say)

    # ---- 4. pretrained_models：只留需要的 ----
    say("\n[4] pretrained_models（只留推理需要的）")
    pm_src = src_root / "GPT_SoVITS" / "pretrained_models"
    pm_dst = gs / "pretrained_models"
    pm_dst.mkdir(parents=True, exist_ok=True)
    for name in PRETRAINED_KEEP:
        copy_item(pm_src / name, pm_dst / name, say)
    kept = dir_size(pm_dst)
    dropped = sum(dir_size(pm_src / n) for n in PRETRAINED_DROP)
    say(f"    ---- 保留 {human(kept)}，丢掉 {human(dropped)}")

    # ---- 5. tools：只留 i18n / my_utils ----
    say("\n[5] tools（只留推理需要的）")
    t_src, t_dst = src_root / "tools", dest / "tools"
    t_dst.mkdir(parents=True, exist_ok=True)
    for name in TOOLS_KEEP:
        copy_item(t_src / name, t_dst / name, say)
    dropped_tools = dir_size(t_src) - dir_size(t_dst)
    say(f"    ---- tools 共 {human(dir_size(t_src))}，保留 {human(dir_size(t_dst))}，丢掉 {human(dropped_tools)}")

    # ---- 6. 声库权重 ----
    say("\n[6] 声库权重")
    prof = VoiceStore(ROOT / "voices").load(voice)
    for rel in (prof.gpt_weights, prof.sovits_weights):
        if not rel:
            continue
        s = src_root / rel
        copy_item(s, dest / rel, say)

    # ---- 7. 修剪 site-packages ----
    say("\n[7] 修剪 site-packages")
    sp = dest / "runtime" / "Lib" / "site-packages"
    before = dir_size(sp)
    removed = []
    for entry in list(sp.iterdir()):
        low = entry.name.lower()
        if any(low.startswith(d.lower()) for d in SITE_DROP):
            sz = dir_size(entry)
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
            removed.append((entry.name, sz))
    for n, sz in sorted(removed, key=lambda x: -x[1])[:12]:
        say(f"    删除 {n:<40} {human(sz):>10}")
    say(f"    ---- site-packages {human(before)} -> {human(dir_size(sp))}")

    total = dir_size(dest)

    # ---- 8. 修掉隐形引用（必须在验证之前做，否则验证是假的）----
    say("\n[8] 修掉指向原包的隐形引用")
    repair_users_pth_in_dest(dest, say)
    find_path_leaks(dest, src_root, say)

    say("\n" + "=" * 84)
    say(f"完成：{human(dir_size(src_root))} -> {human(total)}"
        f"（省下 {human(dir_size(src_root) - total)}）")
    say("=" * 84)
    return {"dest": str(dest), "size": total, "source_size": dir_size(src_root)}


# --------------------------------------------------------------------------- #
def _missing_module(log_text: str) -> str | None:
    import re as _re

    m = _re.search(r"No module named '([A-Za-z0-9_\.]+)'", log_text)
    if not m:
        return None
    return m.group(1).split(".")[0]


def _restore_module(src_root: Path, dest: Path, mod: str, say) -> bool:
    """把误删的模块从原包补回来。"""
    sp_src = src_root / "runtime" / "Lib" / "site-packages"
    sp_dst = dest / "runtime" / "Lib" / "site-packages"
    hits = list(sp_src.glob(f"{mod}*")) + list(sp_src.glob(f"*{mod}*.pth"))
    if not hits:
        # 也可能在 Lib 下（标准库/包），或者是个 .py 文件
        alt = src_root / "runtime" / "Lib" / f"{mod}.py"
        if alt.exists():
            shutil.copy2(alt, dest / "runtime" / "Lib" / alt.name)
            say(f"    补回 Lib/{alt.name}")
            return True
        return False
    for h in hits:
        tgt = sp_dst / h.name
        if h.is_dir():
            if not tgt.exists():
                shutil.copytree(h, tgt, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(h, tgt)
    n = sum(dir_size(sp_dst / h.name) for h in hits)
    say(f"    补回 {mod:<24} {human(n):>10}")
    return True


def verify(dest: Path, voice: str, port: int = 9881, say=print) -> bool:
    """真启一次 api_v2.py 并合成一句话——裁剪不验证等于没做。"""
    say("\n" + "=" * 84)
    say("验证：从裁剪后的目录启动推理服务并合成一句话")
    say("=" * 84)

    prof = VoiceStore(ROOT / "voices").load(voice)
    tmpl = ROOT / "voices" / voice / "tts_infer.yaml"
    old_root = str(Path(prof.train.gsv_root))
    cfg_dst = dest / "_verify_tts_infer.yaml"
    txt = tmpl.read_text(encoding="utf-8").replace(old_root, str(dest))
    cfg_dst.write_text(txt, encoding="utf-8")
    say(f"  推理配置 -> {cfg_dst.name}")

    py = dest / "runtime" / "python.exe"
    if not py.exists():
        say("  !! 裁剪后的目录里没有 runtime\\python.exe")
        return False

    logf = ROOT / "data" / "verify_runtime.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    say("  启动服务…")
    with logf.open("w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.Popen(
            [str(py), "-s", "api_v2.py", "-a", "127.0.0.1", "-p", str(port), "-c", str(cfg_dst)],
            cwd=str(dest), stdout=fh, stderr=subprocess.STDOUT,
        )
    try:
        import urllib.request

        ready = False
        for _ in range(90):
            time.sleep(2)
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=2)
                ready = True
                break
            except Exception:
                if p.poll() is not None:
                    break
        if not ready:
            say("  !! 服务没起来，日志末尾：")
            for line in logf.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]:
                say("     " + line)
            return False
        say("  服务已就绪，合成一句话…")

        body = json.dumps({
            "text": "裁剪验证，一切正常。", "text_lang": "zh",
            "ref_audio_path": str(ROOT / "voices" / voice / "ref.wav"),
            "prompt_text": prof.prompt_text, "prompt_lang": "zh",
            "text_split_method": "cut5", "media_type": "wav",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/tts", data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        t0 = time.time()
        data = urllib.request.urlopen(req, timeout=180).read()
        say(f"  合成成功：{len(data)} 字节，用时 {time.time() - t0:.1f}s")
        out = ROOT / "data" / "verify_runtime.wav"
        out.write_bytes(data)
        say(f"  样本 -> {out}")
        say("\n  裁剪后的运行时可用 ✅")
        return True
    except Exception as e:
        say(f"  !! 合成失败：{e}")
        for line in logf.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]:
            say("     " + line)
        return False
    finally:
        try:
            p.terminate()
            p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        cfg_dst.unlink(missing_ok=True)


def verify_with_repair(src_root: Path, dest: Path, voice: str, port: int = 9881,
                       max_rounds: int = 15, say=print) -> bool:
    """反复「启动 -> 看缺什么 -> 补回 -> 再启动」，直到跑通或用完次数。

    逐个人肉猜哪个包能用太慢——最初把 librosa / torchaudio 删了，
    结果起不来；这个循环就是那次教训固化的产物。
    """
    logf = ROOT / "data" / "verify_runtime.log"
    for rnd in range(1, max_rounds + 1):
        logf.unlink(missing_ok=True)
        if verify(dest, voice, port, say):
            return True
        text = logf.read_text(encoding="utf-8", errors="replace") if logf.exists() else ""
        mod = _missing_module(text)
        if not mod:
            say("\n  !! 不是缺模块导致的失败，停止自动修复。")
            return False
        say(f"\n  [自愈 {rnd}/{max_rounds}] 补齐缺失模块：{mod}")
        if not _restore_module(src_root, dest, mod, say):
            say(f"  !! 原包里也找不到 {mod}，无法修复。")
            return False
    say("\n  !! 超过自愈轮数上限")
    return False


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="裁剪 GPT-SoVITS 推理最小集")
    ap.add_argument("--gsv-root", default=None, help="整合包目录（默认取声库档案里的）")
    ap.add_argument("--voice", default=None, help="声库名（默认取 config.json 里的）")
    ap.add_argument("--dest", required=True, help="输出目录")
    ap.add_argument("--verify", action="store_true", help="裁剪后实际启动服务验证")
    ap.add_argument("--port", type=int, default=9881)
    args = ap.parse_args()

    from screenreader.config import AppConfig

    cfg = AppConfig.load()
    voice = args.voice or cfg.tts.voice
    if not voice:
        print("没有指定声库：用 --voice，或先 voice use <名字>")
        return 2
    prof = VoiceStore(ROOT / "voices").load(voice)
    src_root = Path(args.gsv_root or prof.train.gsv_root)
    if not src_root.exists():
        print(f"整合包目录不存在：{src_root}")
        return 2

    res = build(src_root, Path(args.dest), voice)
    if args.verify:
        ok = verify_with_repair(src_root, Path(args.dest), voice, args.port)
        total = dir_size(Path(args.dest))
        print(f"\n最终大小：{human(total)}（原包 {human(dir_size(src_root))}）")
        return 0 if ok else 1
    print("\n（加 --verify 会实际启一次服务，缺什么自动补回来）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
