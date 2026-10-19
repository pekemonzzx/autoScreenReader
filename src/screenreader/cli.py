"""命令行入口。

    python -m screenreader devices            列出音频输出设备
    python -m screenreader gpu                查看显存与自适应档位
    python -m screenreader capture            截屏自检
    python -m screenreader read               截屏 + OCR 一次，打印识别结果
    python -m screenreader say "要读的话"      合成并播放一句
    python -m screenreader run                循环读屏（正式使用）
    python -m screenreader selftest           端到端自检（不发声）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .audio import Player, describe_default_device, find_output_device, list_output_devices
from .capture import enable_dpi_awareness, list_monitors, selfcheck
from .config import AppConfig, default_config_path
from .segment import split_text
from .watchdog import DEFAULT_RESERVE_MB, decide_tier, query_gpus
from . import __version__

log = logging.getLogger("screenreader")


def _load_cfg(args) -> AppConfig:
    path = Path(args.config) if getattr(args, "config", None) else default_config_path()
    cfg = AppConfig.load(path)
    # 配置里指定了声库就用档案里的推理参数（参考音频/prompt_text/权重）；
    # 没指定就自动选第一个可用的，避免"显示了音色但 ref_audio 是空的"。
    before = (cfg.tts.voice or "").strip()
    got = cfg.apply_voice()
    if got and got != before:
        try:
            cfg.save(path)
        except Exception as e:
            log.debug("写回声库选择失败：%s", e)
    cfg.apply_log_level()
    if getattr(args, "region", None):
        cfg.capture.region = [int(x) for x in args.region.split(",")]
    if getattr(args, "audio_device", None) is not None:
        cfg.audio.device = args.audio_device
    return cfg


def _require(cond, ok_msg="", fail_msg="条件不满足"):
    if not cond:
        raise RuntimeError(fail_msg)
    return ok_msg


# --------------------------------------------------------------------------- #
def cmd_devices(args) -> int:
    print(f"系统默认输出设备：{describe_default_device()}\n")
    print("全部输出设备：")
    for d in list_output_devices():
        print(f"  {d}")
    hit = find_output_device("CABLE Input")
    print()
    if hit:
        print(f"VB-CABLE：已找到 {hit}")
    else:
        print("VB-CABLE：未找到（直播场景需要装 VB-Audio Virtual Cable）")
    return 0


def cmd_gpu(args) -> int:
    gpus = query_gpus()
    if not gpus:
        print("未检测到 NVIDIA 显卡（或 nvidia-smi 不可用）")
    for g in gpus:
        print(f"  {g}")
    tier = decide_tier(gpus, reserve_mb=args.reserve)
    print(f"\n自适应档位：{tier.key} — {tier.label}")
    print(f"  streaming={tier.streaming}  sample_steps={tier.sample_steps}  "
          f"synth_workers={tier.synth_workers}  max_lines={tier.max_lines}")
    print(f"  说明：{tier.note}")
    return 0


def cmd_capture(args) -> int:
    enable_dpi_awareness()
    print("显示器：")
    for m in list_monitors():
        print(f"  {m}")
    region = tuple(int(x) for x in args.region.split(",")) if args.region else None
    print("\n截屏自检：")
    info = selfcheck(region=region, backend=args.backend)
    print("  " + json.dumps(info, ensure_ascii=False))
    return 0 if info.get("ok") else 1


def cmd_read(args) -> int:
    from .capture import ScreenCapturer
    from .ocr import OcrEngine, tune_intra_threads

    cfg = _load_cfg(args)
    enable_dpi_awareness()
    cap = ScreenCapturer(
        monitor_index=cfg.capture.monitor_index,
        backend=cfg.capture.backend,
        region=tuple(cfg.capture.region) if cfg.capture.region else None,
    )
    frame = cap.grab()
    if frame is None:
        print("抓帧失败")
        return 1
    n = tune_intra_threads(frame, cfg.ocr.det_limit_side_len)
    eng = OcrEngine(det_limit_side_len=cfg.ocr.det_limit_side_len, intra_threads=n)
    eng.warmup(frame, 2)

    t0 = time.perf_counter()
    res = eng.read(frame)
    dt = (time.perf_counter() - t0) * 1000
    print(f"识别 {len(res.lines)} 行，用时 {dt:.0f}ms（intra_threads={n}）\n")
    for ln in res.lines:
        print(f"  {ln.score:.2f}  {ln.text}")

    text = "\n".join(ln.text for ln in res.lines)
    segs = split_text(text)
    print(f"\n分句（{len(segs)} 句）：")
    for i, s in enumerate(segs, 1):
        print(f"  {i:>2}. {s}")
    cap.close()
    return 0


def cmd_say(args) -> int:
    from .tts import create_backend

    cfg = _load_cfg(args)
    cfg.apply_log_level()
    backend = create_backend(cfg.tts)
    player = Player(device=cfg.audio.device, latency=cfg.audio.latency)

    print(f"TTS 后端 {backend.name}  输出设备 "
          f"{player.resolve_device() or describe_default_device()}")
    print(f"文本：{args.text}")
    if not backend.health():
        print("!! TTS 服务不可用，请先启动 api_v2.py")
        return 1

    t0 = time.perf_counter()
    if args.stream:
        chunks = backend.synthesize_stream(args.text)
        first = None
        buf = bytearray()

        def spy():
            nonlocal first
            for c in chunks:
                if first is None:
                    first = (time.perf_counter() - t0) * 1000
                buf.extend(c)
                yield c

        fmt = None
        player.play_pcm_stream(
            spy(),
            samplerate=32000,  # 首块到达后 backend.stream_format 才有效；v2ProPlus 固定 32k
            channels=1,
            sampwidth=2,
        )
        fmt = backend.stream_format
        print(f"  流式首包 {first:.0f} ms   总 {(time.perf_counter() - t0) * 1000:.0f} ms   "
              f"格式 {(fmt.samplerate, fmt.channels, fmt.sampwidth) if fmt else '?'}")
    else:
        data = backend.synthesize(args.text)
        print(f"  非流式 {(time.perf_counter() - t0) * 1000:.0f} ms   {len(data)} bytes")
        player.play_wav(data)
    return 0


def cmd_run(args) -> int:
    from .pipeline import ScreenReaderPipeline

    cfg = _load_cfg(args)
    cfg.apply_log_level()
    if getattr(args, "no_skip", False):
        cfg.pipeline.skip_unchanged = False
    if getattr(args, "no_dedupe", False):
        cfg.pipeline.dedupe_lines = False
    if getattr(args, "no_cache", False):
        cfg.cache.enabled = False

    def on_event(e):
        t = e.get("type")
        if t == "ocr":
            log.info("OCR %d 行 / %d 字  (截屏 %.0fms + 识别 %.0fms)",
                     e["lines"], e["chars"], e["capture_ms"], e["ocr_ms"])
        elif t == "segment":
            log.info("分句 %d 句：%s", e["count"], " / ".join(e["segments"])[:120])
        elif t == "cache_hit":
            log.info("★ 缓存命中「%s」（0 ms）", e["text"])
        elif t == "tts":
            if e.get("streamed"):
                log.info("合成「%s」首包 %s ms / 总 %.0f ms",
                         e["text"], e.get("first_packet_ms"), e.get("total_ms", 0))
            else:
                log.info("合成「%s」%.0f ms", e["text"], e.get("ms", 0))
        elif t == "backlog_drop":
            log.info("积压过多，丢弃 %d 句已滚出屏幕的内容", e["dropped"])
        elif t == "tier":
            log.warning("显存降档 -> %s（%s）", e["tier"], e["label"])
        elif t == "error":
            log.warning("出错：%s", e)

    pipe = ScreenReaderPipeline(cfg, on_event=on_event)
    pipe.start()
    print(f"region = {pipe.capturer.region}  monitor = {pipe.capturer.monitor}")
    print(f"输出设备 = {pipe.player.resolve_device() or describe_default_device()}")
    print(f"显存档位 = {pipe.tier.key} — {pipe.tier.label}")
    print(f"缓存 = {'开' if cfg.cache.enabled else '关'}   流式 = {cfg.tts.streaming}   "
          f"行去重 = {'开' if cfg.pipeline.dedupe_lines else '关'}")
    print("按 Ctrl+C 退出\n" if not args.duration else f"限时运行 {args.duration} 秒\n")

    t0 = time.time()
    try:
        while not (args.duration and time.time() - t0 >= args.duration):
            rr = pipe.read_once()
            if rr.new_text and not rr.skipped:
                pipe.speak(rr.new_text)
            elif rr.reason:
                log.debug("跳过：%s", rr.reason)
            time.sleep(max(0.05, args.interval if args.interval is not None
                           else cfg.pipeline.min_interval_s))
    except KeyboardInterrupt:
        print("\n收到中断…")
    finally:
        pipe.stop()
    print("\n统计：" + json.dumps(pipe.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_gui(args) -> int:
    """图形界面：框选区域 -> 持续识别 -> 按句列出 -> 手动播放。"""
    from .ui.app import main as ui_main

    argv = [sys.argv[0]]
    if getattr(args, "config", None):
        argv += ["-c", args.config]
    return ui_main(argv)


def cmd_selftest(args) -> int:
    """不发声的端到端自检：截屏 → OCR → 分句 → TTS 探活 → 缓存读写。"""
    from .cache import AudioCache, make_key, wav_from_pcm, wav_info
    from .capture import ScreenCapturer
    from .ocr import OcrEngine, tune_intra_threads
    from .tts import create_backend

    cfg = _load_cfg(args)
    cfg.apply_log_level()
    ok = True

    def step(name, fn):
        nonlocal ok
        t0 = time.perf_counter()
        try:
            detail = fn()
            ms = (time.perf_counter() - t0) * 1000
            print(f"  [OK]   {name:<26} {ms:>7.0f} ms   {detail or ''}")
            return True
        except Exception as e:
            print(f"  [FAIL] {name:<26} {type(e).__name__}: {e}")
            ok = False
            return False

    print("=" * 78)
    print(f"端到端自检  screenreader {__version__}")
    print("=" * 78)

    enable_dpi_awareness()
    state = {}

    print("\n[环境]")
    step("显示器枚举", lambda: f"{len(list_monitors())} 个")
    step("音频输出设备", lambda: f"{len(list_output_devices())} 个，默认 {describe_default_device()}")
    step("显存探测", lambda: (lambda g: f"{len(g)} 张卡，档位 {decide_tier(g).key}")(query_gpus()))
    step(
        "TTS 服务探活",
        lambda: _require(
            create_backend(cfg.tts).health(),
            ok_msg=f"可用 {cfg.tts.api_url}",
            fail_msg=f"不可用，请先启动 api_v2.py（{cfg.tts.api_url}）",
        ),
    )

    print("\n[截屏 + OCR]")

    def _cap():
        cap = ScreenCapturer(
            monitor_index=cfg.capture.monitor_index,
            backend=cfg.capture.backend,
            region=tuple(cfg.capture.region) if cfg.capture.region else None,
        )
        f = cap.grab()
        if f is None:
            raise RuntimeError("抓帧失败")
        state["cap"] = cap
        state["frame"] = f
        return str(f.shape)

    if step("抓帧", _cap):
        def _ocr():
            n = tune_intra_threads(state["frame"], cfg.ocr.det_limit_side_len)
            eng = OcrEngine(det_limit_side_len=cfg.ocr.det_limit_side_len, intra_threads=n)
            eng.warmup(state["frame"], 1)
            t0 = time.perf_counter()
            r = eng.read(state["frame"])
            state["ocr_ms"] = (time.perf_counter() - t0) * 1000
            state["text"] = "\n".join(x.text for x in r.lines)
            return f"{len(r.lines)} 行，{state['ocr_ms']:.0f} ms (intra={n})"

        if step("OCR", _ocr):
            step("分句", lambda: f"{len(split_text(state['text']))} 句")

    print("\n[缓存]")

    def _cache():
        c = AudioCache(cfg.cache_dir(), max_entries=10, max_mb=8)
        data = wav_from_pcm(b"\x00\x00" * 1600, 32000)
        k = make_key("自检样本")
        c.put(k, data)
        got = c.get(k)
        sr, ch, dur = wav_info(got or b"")
        if not got:
            raise RuntimeError("回读失败")
        c.clear()
        return f"{len(data)} bytes 回读成功，{sr}Hz {dur:.2f}s"

    step("写入/回读/清空", _cache)

    print("\n[合成（不播放）]")

    def _synth():
        b = create_backend(cfg.tts)
        t0 = time.perf_counter()
        chunks = b.synthesize_stream("自检，这是一句话。")
        total = 0
        first = None
        for c in chunks:
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            total += len(c)
        return f"流式首包 {first:.0f} ms，共 {total} bytes"

    step("流式合成", _synth)

    if "cap" in state:
        state["cap"].close()

    print("\n" + "=" * 78)
    print("全部通过 ✅" if ok else "存在失败项 ❌")
    print("=" * 78)
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    # 公共参数做成 parent，这样 `screenreader read --region ...` 和
    # `screenreader --region ... read` 两种写法都能用。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", default=None, help="配置文件路径（默认 data/config.json）")
    common.add_argument("--region", default=None, help="截屏区域 left,top,width,height")
    common.add_argument("--audio-device", default=None, help="输出设备名子串，如 'CABLE Input'")

    ap = argparse.ArgumentParser(prog="screenreader", description="自动读屏工具")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("-c", "--config", default=None, help="配置文件路径（默认 data/config.json）")
    ap.add_argument("--region", default=None, help="截屏区域 left,top,width,height")
    ap.add_argument("--audio-device", default=None, help="输出设备名子串，如 'CABLE Input'")

    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("devices", help="列出音频输出设备", parents=[common])
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser("gpu", help="查看显存与自适应档位", parents=[common])
    p.add_argument("--reserve", type=int, default=DEFAULT_RESERVE_MB,
                   help=f"给游戏/OBS 留的安全余量 MB（默认 {DEFAULT_RESERVE_MB}）")
    p.set_defaults(func=cmd_gpu)

    p = sub.add_parser("capture", help="截屏自检", parents=[common])
    p.add_argument("--backend", default="auto", choices=["auto", "dxgi", "mss"])
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("read", help="截屏 + OCR + 分句一次", parents=[common])
    p.set_defaults(func=cmd_read)

    p = sub.add_parser("say", help="合成并播放一句", parents=[common])
    p.add_argument("text")
    p.add_argument("--stream", action="store_true", default=True)
    p.add_argument("--no-stream", dest="stream", action="store_false")
    p.set_defaults(func=cmd_say)

    p = sub.add_parser("run", help="循环读屏", parents=[common])
    p.add_argument("--interval", type=float, default=None, help="两轮之间的间隔秒数")
    p.add_argument("--duration", type=float, default=None, help="限时运行秒数（测试用，不给则一直跑）")
    p.add_argument("--no-skip", action="store_true",
                   help="内容没变也重读（用来验证缓存命中，正式使用别开）")
    p.add_argument("--no-dedupe", action="store_true", help="关闭行级去重（每轮整块重读）")
    p.add_argument("--no-cache", action="store_true", help="关闭音频缓存")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("gui", help="图形界面（框选区域 + 手动播放读屏台）", parents=[common])
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("selftest", help="端到端自检", parents=[common])
    p.set_defaults(func=cmd_selftest)

    # 声库管理（列表/新建/训练/打包/安装/切换）
    from .voice_cli import add_parser as add_voice_parser

    add_voice_parser(sub)

    # TTS 推理运行时（打包/安装/配置/自检）
    from .runtime_cli import add_parser as add_runtime_parser

    add_runtime_parser(sub)

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        build_parser().print_help()
        return 2
    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
