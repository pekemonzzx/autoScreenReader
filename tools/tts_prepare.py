"""从直播录像一键跑到 GPT-SoVITS 训练用的 .list 标注文件。

流程（对应 WebUI 的 0a -> 0b -> 0c）：
    1. ffmpeg 抽音轨            录像/录音 -> work/00_raw/*.wav
    2. UVR5 人声分离            去掉背景音乐/游戏声 -> work/01_vocal/
    3. 音频切片（slicer2）      按静音切成 3-10 秒片段 -> work/02_sliced/
    4. ASR 自动标注（FunASR）   中文识别 + 标点 -> work/03_list/*.list
    5.（人工）校对 .list        这一步必须做，见文末说明

必须用 GPT-SoVITS 集成包自带的内嵌 Python 运行（它有 torch/funasr/librosa）：

    D:\\GPT-SoVITS\\GPT-SoVITS-v2pro-20250604\\runtime\\python.exe ^
        tools\\tts_prepare.py --input <录像路径> --work D:\\GPT-SoVITS\\work

整合包根目录可以用 --gsv-root 指定，或设环境变量 GPT_SOVITS_ROOT。
默认必须是**训练用的那个包**（v2pro）—— 早先这里写死过 v3lora，
和训练用的不是同一个包，权重目录也对不上（v3lora 里没有 v2ProPlus 权重目录）。

踩过的坑（已在本脚本里处理，勿删）：
  * tools/uvr5/webui.py 内部用 os.system("ffmpeg ...") 调**裸命令名**，
    所以必须把集成包根目录（内含 ffmpeg.exe/ffprobe.exe）前置到 PATH，并 chdir 到根目录。
  * tools/uvr5/webui.py 在**模块级**读 sys.argv[1..4]（device/is_half/port/is_share），
    import 前必须临时替换 sys.argv，否则会被本脚本自己的命令行参数污染而 SyntaxError。
  * tools/uvr5/webui.py 与 tools/slice_audio.py 都用**裸模块名** import
    （from mdxnet import ... / from slicer2 import Slicer），
    因此 GSV_ROOT/tools 与 GSV_ROOT/tools/uvr5 也必须进 sys.path。
  * tools/uvr5/webui.py 里 weight_uvr5_root 是相对路径，所以必须 chdir 到 GSV_ROOT。
  * uvr() 是生成器，必须迭代消费，否则整个分离过程不会执行。
  * uvr() 只接受「目录」或「目录 + 文件名列表」，不接受单个文件路径。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_GSV_ROOT = Path(
    os.environ.get("GPT_SOVITS_ROOT", r"D:\GPT-SoVITS\GPT-SoVITS-v2pro-20250604"))

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".wma"}
VIDEO_EXT = {".mp4", ".flv", ".mkv", ".ts", ".mov", ".avi", ".webm", ".wmv"}


# --------------------------------------------------------------------------- #
def setup_env(gsv_root: Path) -> None:
    """让 os.system("ffmpeg ...") 能找到集成包自带的 ffmpeg，并满足各工具的 import 约定。"""
    if not (gsv_root / "runtime" / "python.exe").exists():
        sys.exit(f"找不到集成包内嵌 Python：{gsv_root}\\runtime\\python.exe")
    if not (gsv_root / "ffmpeg.exe").exists():
        sys.exit(f"集成包根目录没有 ffmpeg.exe：{gsv_root}")
    os.environ["PATH"] = (
        f"{gsv_root}{os.pathsep}{gsv_root / 'runtime'}{os.pathsep}" + os.environ.get("PATH", "")
    )
    # uvr5/webui.py 里的 weight_uvr5_root 是相对路径，必须 chdir 到根目录
    os.chdir(gsv_root)
    # tools/uvr5/webui.py 用 `from mdxnet import ...`、
    # tools/slice_audio.py 用 `from slicer2 import Slicer` —— 都是裸模块名
    for p in (gsv_root, gsv_root / "tools", gsv_root / "tools" / "uvr5"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def collect_inputs(src: Path) -> list:
    if src.is_file():
        return [src]
    files = [
        p
        for p in sorted(src.rglob("*"))
        if p.is_file() and p.suffix.lower() in (AUDIO_EXT | VIDEO_EXT)
    ]
    if not files:
        sys.exit(f"目录里没有找到音视频文件：{src}")
    return files


def human(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def parse_time(text: str) -> float:
    """'5:07' / '1:02:03' / '307' -> 秒"""
    parts = [float(x) for x in text.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def free_ram_gb() -> float:
    """当前可用物理内存（GB）。Windows 用 GlobalMemoryStatusEx，其余平台退回 -1。"""
    if os.name != "nt":
        return -1.0
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return -1.0
        return st.ullAvailPhys / 1024 ** 3
    except Exception:
        return -1.0


def uv5_ram_estimate_gb(seconds: float) -> float:
    """粗估 UVR5 处理这么长的音频要多少内存。

    UVR5 会把**整段音频**重采样成 44.1kHz 立体声 float64 再进模型：
        1480s × 44100 × 2ch × 8B ≈ 1.04 GB —— 只是一份。
    实际会同时存在好几份（原始/重采样/中间张量/结果），按 4 份估，
    再加 torch+CUDA 上下文约 1 GB。

    实测教训：24 分钟整段跑，166 MB 的 numpy 分配失败，
    报 `_ArrayMemoryError`，而且是在流程中间炸掉，前面的活白干。
    """
    per_copy = seconds * 44100 * 2 * 8 / 1024 ** 3
    return per_copy * 4 + 1.0


def check_memory(seconds: float, say=print) -> None:
    """跑 UVR5 之前先看看内存够不够，不够就明确报错而不是中途崩。"""
    need = uv5_ram_estimate_gb(seconds)
    free = free_ram_gb()
    if free < 0:
        return
    say(f"  内存预检：可用 {free:.1f} GB，UVR5 预计需要约 {need:.1f} GB")
    if free < need:
        say("")
        say("  !! 内存不足，UVR5 会中途崩掉（numpy._ArrayMemoryError）。")
        say(f"     现在可用 {free:.1f} GB，预计需要 {need:.1f} GB。")
        say("     解决办法（任选）：")
        say("       1. 关掉占内存的程序（浏览器、游戏、聊天软件）后重跑")
        say("       2. 用 --segments 把素材截短一点")
        say("       3. 把素材切成几段分别跑（会牺牲一点断句质量）")
        say("")
        sys.exit(2)


def parse_segments(text: str) -> list:
    """'0:00-5:07,18:08-25:47' -> [(0.0, 307.0), (1088.0, 1547.0)]"""
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            sys.exit(f"时间段格式错误（应为 起始-结束）：{chunk}")
        a, b = chunk.split("-", 1)
        start, end = parse_time(a), parse_time(b)
        if end <= start:
            sys.exit(f"时间段结束必须大于起始：{chunk}")
        out.append((start, end))
    return out


def fmt_ts(sec: float) -> str:
    return f"{int(sec) // 60:02d}{int(sec) % 60:02d}"


def probe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def wav_duration(path: Path) -> float:
    """直接读 WAV 头，避免每条切片都起一个 ffprobe 进程。"""
    import wave

    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return probe_duration(path)


# --------------------------------------------------------------------------- #
def step1_extract(files: list, raw_dir: Path, segments: list = None,
                  sample_rate: int = 44100) -> list:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = []

    # 展开成 (源文件, 起始秒, 结束秒或 None) 的任务列表
    jobs = []
    for f in files:
        if segments:
            for start, end in segments:
                jobs.append((f, start, end))
        else:
            jobs.append((f, None, None))

    for i, (f, start, end) in enumerate(jobs, 1):
        if start is None:
            dst = raw_dir / f"{f.stem}.wav"
            label = f.name
        else:
            dst = raw_dir / f"{f.stem}_{fmt_ts(start)}-{fmt_ts(end)}.wav"
            label = f"{f.name}  [{human(start)} - {human(end)}]  共 {human(end - start)}"
        print(f"  [{i}/{len(jobs)}] {label}")

        if dst.exists() and dst.stat().st_size > 1000:
            print("         已存在，跳过")
            out.append(dst)
            continue

        cmd = ["ffmpeg", "-v", "error", "-y"]
        if start is not None:
            # -ss 放在 -i 之前做快速定位，配合重编码仍然是精确的
            cmd += ["-ss", f"{start:.3f}", "-i", str(f), "-t", f"{end - start:.3f}"]
        else:
            cmd += ["-i", str(f)]
        # **不要在这里降采样！**
        #
        # 这里以前写死 `-ar 16000`，把源音频砍到 16kHz（奈奎斯特 8kHz）。
        # 后果很具体：擦音/塞擦音（ch zh sh c z s）的区分特征主要在 4~10kHz，
        # 砍到 8kHz 之后就只剩个"闷"的轮廓 —— 训练出来的语音这些音会不清晰。
        #
        # 实测（480p 源视频，音轨本身是 48kHz AAC）：
        #     频段        源视频      砍成 16kHz 后
        #     8-11kHz     -36.1dB     -59.0dB     丢掉约 23dB
        #     11-16kHz    -40.0dB     -85.0dB     丢掉约 45dB
        # 信息本来就在，是管线自己扔的。
        #
        # 后面 UVR5 反正会把音频重采样到 44.1kHz，
        # 所以这里保持 44.1kHz 不增加任何负担，纯粹是别再丢东西。
        cmd += ["-vn", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1", str(dst)]

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"         !! ffmpeg 失败：{r.stderr.strip()[:300]}")
            continue
        print(f"         -> {dst.name}  ({dst.stat().st_size / 1e6:.1f} MB)")
        out.append(dst)
    return out


def step2_separate(raw_dir: Path, vocal_dir: Path, model: str, agg: int) -> None:
    # 逐行计时：这一步曾经"卡在内存预检之后、模型打印之前"，没有任何输出，
    # 只能靠打点定位到具体是哪一句。加了 flush=True 保证立刻可见。
    import time as _t

    _t0 = _t.time()

    def _mark(msg: str) -> None:
        print(f"      [sep {_t.time() - _t0:6.1f}s] {msg}", flush=True)

    _mark("进入 step2_separate")

    # uvr5/webui.py 在模块级执行 device=sys.argv[1] / is_half=eval(sys.argv[2]) / ...
    # 必须先把 argv 换成它期望的形状，import 完再还原。
    saved_argv = sys.argv[:]
    sys.argv = ["webui.py", "cuda", "True", "9873", "False"]
    try:
        # 更关键：webui.py 末尾直接 app.queue().launch(...)，且**没有 __main__ 保护**。
        # 直接 import 会拉起一个 Gradio 服务并阻塞（实测卡在 "Running on local URL"）。
        # 把 launch 换成空操作，只为拿到 uvr() 函数。
        _mark(f"import gradio 开始（TEMP={os.environ.get('TEMP')}）")
        import gradio as gr

        _mark("import gradio 完成")
        gr.Blocks.launch = lambda self, *a, **k: self
        _mark("开始 import tools.uvr5.webui")
        from tools.uvr5.webui import uvr

        _mark("import tools.uvr5.webui 完成")
    finally:
        sys.argv = saved_argv

    vocal_dir.mkdir(parents=True, exist_ok=True)
    ins_dir = vocal_dir.parent / "01_instrumental"
    ins_dir.mkdir(parents=True, exist_ok=True)
    _mark(f"输出目录就绪 vocal={vocal_dir}")

    print(f"  模型={model}  激进度={agg}  （GPU 推理，按音频时长可能需要数分钟）", flush=True)
    # 注意：uvr() 是生成器，必须迭代消费
    for msg in uvr(str(model), str(raw_dir), str(vocal_dir), [], str(ins_dir), str(agg), "wav"):
        if msg:
            tail = msg.strip().splitlines()[-1]
            print(f"         {tail[:200]}", flush=True)
    _mark("uvr() 迭代结束")


def step3_slice(vocal_dir: Path, slice_dir: Path, **kw) -> None:
    # tools/slice_audio.py 末行是裸的 `print(slice(*sys.argv[1:]))`（同样没有 __main__ 保护）。
    # 把 argv 换成「不存在的输入路径 + 合法参数」，让这次副作用调用立刻返回、什么都不做。
    saved_argv = sys.argv[:]
    sys.argv = [
        "slice_audio.py",
        str(vocal_dir / "__nonexistent__.wav"),
        str(slice_dir),
        str(kw["threshold"]), str(kw["min_length"]), str(kw["min_interval"]),
        str(kw["hop_size"]), str(kw["max_sil_kept"]), str(kw["max"]), str(kw["alpha"]),
        "0", "1",
    ]
    try:
        from tools.slice_audio import slice as slice_one
    finally:
        sys.argv = saved_argv

    slice_dir.mkdir(parents=True, exist_ok=True)
    vocs = sorted(p for p in vocal_dir.glob("*.wav") if p.stat().st_size > 1000)
    for i, v in enumerate(vocs, 1):
        print(f"  [{i}/{len(vocs)}] {v.name}")
        slice_one(
            str(v), str(slice_dir),
            kw["threshold"], kw["min_length"], kw["min_interval"],
            kw["hop_size"], kw["max_sil_kept"], kw["max"], kw["alpha"],
            0, 1,
        )


def step4_asr(slice_dir: Path, list_dir: Path, language: str, model_size: str) -> list:
    # **必须屏蔽 ModelScope 的联网检查，否则这一步会永久 hang。**
    #
    # 整合包的 tools/asr/funasr_asr.py 里 create_model() **无条件**调用
    #     snapshot_download("iic/speech_paraformer-large...", local_dir=...)
    # 它是联网接口，会去服务器查更新 —— 即使模型本来就在本地。
    # 网络不通时它不报错、也不超时，就是**一直挂着**：
    #     0 CPU + 0 I/O + 线程全在 Wait
    # 实测卡了 16 分钟没动静。
    #
    # 模型已经在 tools/asr/models/ 里了，所以把 snapshot_download 换成空操作。
    # 同时设一下几个离线环境变量兜底。
    os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import modelscope

        def _no_download(*a, **k):
            print("      (跳过 modelscope 联网检查，直接用本地模型)", flush=True)
            return None

        modelscope.snapshot_download = _no_download
        # 子模块里可能还有一份名字，一并换掉
        try:
            import modelscope.hub.snapshot_download as _sdmod
            if hasattr(_sdmod, "snapshot_download"):
                _sdmod.snapshot_download = _no_download
        except Exception:
            pass
    except Exception as e:
        print(f"      !! 没法屏蔽 modelscope（{e}），可能会卡在联网检查上", flush=True)

    from tools.asr.funasr_asr import execute_asr

    list_dir.mkdir(parents=True, exist_ok=True)
    print(f"  FunASR  Paraformer-{model_size}  语言={language}", flush=True)
    execute_asr(str(slice_dir), str(list_dir), model_size, language)
    return sorted(list_dir.glob("*.list"))


# --------------------------------------------------------------------------- #
def report(list_files: list, slice_dir: Path) -> dict:
    total, rows = 0.0, []
    for lf in list_files:
        for line in lf.read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) < 4:
                continue
            p = Path(parts[0])
            if not p.is_absolute():
                p = slice_dir / p.name
            d = wav_duration(p) if p.exists() else 0.0
            if d <= 0 or not parts[3].strip():
                continue
            total += d
            rows.append({"wav": str(p), "text": parts[3], "dur": round(d, 2)})

    n = len(rows)
    durs = sorted(r["dur"] for r in rows)
    stats = {
        "utterances": n,
        "total_seconds": round(total, 1),
        "avg_seconds": round(total / n, 2) if n else 0,
        "median_seconds": round(durs[n // 2], 2) if n else 0,
        "min_seconds": round(durs[0], 2) if n else 0,
        "max_seconds": round(durs[-1], 2) if n else 0,
        "chars": sum(len(r["text"]) for r in rows),
    }
    return {"stats": stats, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="直播录像 -> GPT-SoVITS 训练标注")
    ap.add_argument("--input", required=True, help="录像/录音文件或目录")
    ap.add_argument("--work", required=True, help="工作目录")
    ap.add_argument(
        "--segments",
        default=None,
        help="只取指定时间段，逗号分隔，如 '0:00-5:07,18:08-25:47'（不给则取全部）",
    )
    ap.add_argument("--force", action="store_true", help="忽略已有的中间结果，强制重跑")
    ap.add_argument("--gsv-root", default=str(DEFAULT_GSV_ROOT))
    ap.add_argument("--model", default="HP5_only_main_vocal", help="UVR5 模型名（不含扩展名）")
    ap.add_argument("--agg", type=int, default=10, help="UVR5 激进度 0-20")
    ap.add_argument("--language", default="zh", choices=["zh", "yue", "auto"])
    ap.add_argument("--asr-size", default="large", choices=["large", "medium", "small"])
    ap.add_argument("--threshold", type=float, default=-34)
    ap.add_argument("--min-length", type=int, default=4000)
    ap.add_argument("--min-interval", type=int, default=300)
    ap.add_argument("--hop-size", type=int, default=10)
    ap.add_argument("--max-sil-kept", type=int, default=500)
    ap.add_argument("--max", type=float, default=0.9)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-separate", action="store_true")
    ap.add_argument("--skip-slice", action="store_true")
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument(
        "--sample-rate", type=int, default=44100,
        help="抽音轨时的采样率。**别调低**：擦音 ch/zh/sh/c/z/s 的特征在 4~10kHz，"
             "降到 16kHz 会让这些音发闷（默认 44100）",
    )
    args = ap.parse_args()

    t_start = time.time()

    # **先把路径解析成绝对路径，再 setup_env()**。
    #
    # setup_env() 里会 `os.chdir(gsv_root)`（UVR5 的 weight_uvr5_root 是相对路径），
    # 之后再 resolve 相对路径就会按 GSV_ROOT 去算 —— 必然找不到文件。
    # 这个顺序错过一次，表现是"输入不存在：data\xxx.m4a"，很费解。
    src = Path(args.input).resolve()
    if not src.exists():
        sys.exit(f"输入不存在：{src}")
    work = Path(args.work).resolve()

    setup_env(Path(args.gsv_root))

    # **显式指定一个可写的临时目录**。
    #
    # 踩过的坑：setup_env() 会 os.chdir(gsv_root)，而 tempfile 在系统 TEMP
    # 不可写时会**退回到当前目录**。如果整合包目录也不能写，就会：
    #     FileNotFoundError: No usable temporary directory found in [...]
    # 表现是 import gradio 直接崩（gradio_client 在模块级就调 gettempdir）。
    #
    # 放在 work 下最稳：work 是我们本来就要写的目录，一定可写，
    # 而且和这次的语料在一起，出问题好找。
    tmp_dir = work / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)
    print(f"  临时目录  -> {tmp_dir}")

    # **numba 编译缓存也必须指到可写目录，否则 `import librosa` 会永久卡死。**
    #
    # numba 默认把 JIT 编译结果写进**源码旁边的 __pycache__**。整合包目录
    # 不一定可写 —— 那样它会无限重试：CPU 满载、零输出、永不返回。
    # 实测症状：`from bsroformer import Roformer_Loader` 卡死 20+ 分钟，
    # 二分下来真正的元凶是它依赖的 `import librosa`；
    # 而设了 NUMBA_CACHE_DIR 之后 1.4 秒就过了。
    #
    # 这个坑是我自己踩出来的：早先清理项目时递归删掉了所有 __pycache__，
    # 把 numba 的缓存一并清了，于是每次 import 都要重建 -> 卡死。
    cache_dir = work / "_cache"
    (cache_dir / "numba").mkdir(parents=True, exist_ok=True)
    (cache_dir / "mpl").mkdir(parents=True, exist_ok=True)
    (cache_dir / "librosa").mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir / "numba")
    os.environ["MPLCONFIGDIR"] = str(cache_dir / "mpl")
    os.environ["LIBROSA_CACHE_DIR"] = str(cache_dir / "librosa")
    print(f"  缓存目录  -> {cache_dir}")

    raw_dir, vocal_dir = work / "00_raw", work / "01_vocal"
    slice_dir, list_dir = work / "02_sliced", work / "03_list"
    for d in (raw_dir, vocal_dir, slice_dir, list_dir):
        d.mkdir(parents=True, exist_ok=True)

    files = collect_inputs(src)
    segments = parse_segments(args.segments) if args.segments else None
    print("=" * 78)
    print(f"输入 {len(files)} 个文件  ->  工作目录 {work}")
    if segments:
        total = sum(b - a for a, b in segments)
        for a, b in segments:
            print(f"  取段：{human(a)} - {human(b)}   共 {human(b - a)}")
        print(f"  取段合计：{human(total)}")
    print("=" * 78)

    if not args.skip_extract:
        print("\n[1/4] 抽音轨")
        step1_extract(files, raw_dir, segments, args.sample_rate)
    else:
        print("\n[1/4] 抽音轨  跳过")

    if not args.skip_separate:
        existing = [p for p in vocal_dir.glob("*.wav") if p.stat().st_size > 1000]
        if existing and not args.force:
            print(f"\n[2/4] UVR5 人声分离  已存在 {len(existing)} 个人声文件，跳过（--force 可强制重跑）")
        else:
            print("\n[2/4] UVR5 人声分离")
            # 先看内存够不够：UVR5 会把整段音频读进内存，
            # 不够的话会在流程中间抛 numpy._ArrayMemoryError，前面的活全白干。
            raws = [p for p in raw_dir.glob("*.wav") if p.stat().st_size > 1000]
            secs = max((wav_duration(p) for p in raws), default=0.0)
            if secs > 0:
                check_memory(secs)
            step2_separate(raw_dir, vocal_dir, args.model, args.agg)
    else:
        print("\n[2/4] 人声分离  跳过")

    if not args.skip_slice:
        print("\n[3/4] 音频切片")
        step3_slice(
            vocal_dir, slice_dir,
            threshold=args.threshold, min_length=args.min_length,
            min_interval=args.min_interval, hop_size=args.hop_size,
            max_sil_kept=args.max_sil_kept, max=args.max, alpha=args.alpha,
        )
    else:
        print("\n[3/4] 切片  跳过")

    if not args.skip_asr:
        print("\n[4/4] ASR 标注")
        list_files = step4_asr(slice_dir, list_dir, args.language, args.asr_size)
    else:
        list_files = sorted(list_dir.glob("*.list"))

    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    rep = report(list_files, slice_dir)
    s = rep["stats"]
    print(f"  标注文件     : {[str(p) for p in list_files]}")
    print(f"  有效片段数   : {s['utterances']}")
    print(f"  音频总时长   : {human(s['total_seconds'])}")
    print(
        f"  单条 平均/中位/最短/最长 : "
        f"{s['avg_seconds']}s / {s['median_seconds']}s / {s['min_seconds']}s / {s['max_seconds']}s"
    )
    print(f"  总字数       : {s['chars']}")

    if s["total_seconds"] < 60:
        print("\n  !! 有效音频不足 1 分钟，训练效果会很差。建议再补素材。")
    elif s["total_seconds"] < 300:
        print("\n  ~  1-5 分钟：能训出效果，但音色稳定性一般。")
    else:
        print(f"\n  OK  {human(s['total_seconds'])}：足够微调出稳定的音色。")

    rep["list_files"] = [str(p) for p in list_files]
    rep["elapsed_seconds"] = round(time.time() - t_start, 1)
    out = work / "prepare_report.json"
    out.write_text(
        json.dumps({k: v for k, v in rep.items() if k != "rows"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (work / "train.list").write_text(
        "\n".join(f"{r['wav']}|speaker|zh|{r['text']}" for r in rep["rows"]), encoding="utf-8"
    )

    print("\n" + "-" * 78)
    print("下一步（必须人工做，不能跳）：")
    print(f"  1. 打开 {list_dir} 里的 .list，逐条校对识别文本。")
    print("     ASR 会听错同音字、漏掉气口，校对的干净程度直接决定最终音色质量。")
    print("  2. 删掉听不清、有杂音、有背景音乐残留、或超过 10 秒的条目。")
    print(f"  3. 校对好的列表在 {work / 'train.list'}（已合并成单文件）。")
    print(f"  4. 打开 {DEFAULT_GSV_ROOT / 'go-webui.bat'}，在「1-GPT-SoVITS-TTS」页签")
    print("     填实验名 -> 填标注文件路径 -> 选择版本 v2ProPlus -> 一键三连。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
