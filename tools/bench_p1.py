"""P1 基准测试：截屏与 OCR 的延迟。

用途：校验方案 v2.1 §5 的延迟预估，并为目标机选型（≥12GB / 8GB 降级档）提供依据。

用法：
    .\\.venv\\Scripts\\python tools\\bench_p1.py                 # 合成图 + 自动标定线程数
    .\\.venv\\Scripts\\python tools\\bench_p1.py --runs 30
    .\\.venv\\Scripts\\python tools\\bench_p1.py --real          # 用真实屏幕内容
    .\\.venv\\Scripts\\python tools\\bench_p1.py --region 520,140,520,140
    .\\.venv\\Scripts\\python tools\\bench_p1.py --retune        # 强制重新标定线程数
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from screenreader.capture import (  # noqa: E402
    ScreenCapturer,
    enable_dpi_awareness,
    list_monitors,
    selfcheck,
)
from screenreader.ocr import OcrEngine, default_params, tune_intra_threads  # noqa: E402

SAMPLE_LINES = [
    "欢迎来到直播间，今天我们来玩一个全新的游戏",
    "弹幕：主播这个操作也太秀了吧 666",
    "当前任务：击败boss并收集三件稀有装备",
    "物品栏 金币 12345 生命值 87/100 魔法值 42/60",
]

# 方案 §5 的延迟预算（目标机空闲态）
BUDGET = {"ocr": "80-200ms", "capture": "5-20ms", "e2e": "100-250ms"}


# --------------------------------------------------------------------------- #
def make_test_image(width: int = 900, height: int = 420) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    font_path = None
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc"):
        p = Path(r"C:\Windows\Fonts") / name
        if p.exists():
            font_path = p
            break
    if font_path is None:
        raise RuntimeError("找不到中文字体（msyh.ttc / simhei.ttf / simsun.ttc）")

    img = Image.new("RGB", (width, height), (248, 248, 250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(font_path), 26)
    y = 28
    for line in SAMPLE_LINES:
        draw.text((32, y), line, font=font, fill=(28, 28, 32))
        y += 62
    return np.asarray(img)[:, :, ::-1].copy()  # RGB -> BGR


def stats(samples_ms):
    arr = sorted(samples_ms)
    n = len(arr)

    def pct(p):
        return arr[min(n - 1, int(round(p / 100 * (n - 1))))]

    return {
        "n": n,
        "mean": round(statistics.fmean(arr), 1),
        "p50": round(pct(50), 1),
        "p95": round(pct(95), 1),
        "min": round(arr[0], 1),
        "max": round(arr[-1], 1),
    }


def fmt(name: str, s: dict, budget: str = "") -> str:
    tail = f"   方案目标 {budget}" if budget else ""
    return (
        f"  {name:<20} n={s['n']:<3} p50={s['p50']:>7.1f}ms  p95={s['p95']:>7.1f}ms  "
        f"min={s['min']:>7.1f}  max={s['max']:>7.1f}{tail}"
    )


def verdict(actual: float, budget_text: str) -> str:
    """把 '80-200ms' 这样的区间解析成上下限做判定。"""
    try:
        lo, hi = (float(x.replace("ms", "")) for x in budget_text.split("-"))
    except Exception:
        return ""
    if actual < lo:
        return "达标（优于目标）"
    if actual <= hi:
        return "达标"
    return f"超标 {actual / hi:.1f}x"


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--monitor", type=int, default=0)
    ap.add_argument("--backend", default="auto", choices=["auto", "dxgi", "mss"])
    ap.add_argument("--region", default=None, help="left,top,width,height（默认整屏）")
    ap.add_argument("--image", default=None, help="用一张图片替代合成图")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="用合成中文图做 OCR 输入（默认用真实抓到的屏幕帧，即真实管线）",
    )
    ap.add_argument("--retune", action="store_true", help="强制重新标定 OCR 线程数")
    ap.add_argument("--det-limit", type=int, default=960)
    ap.add_argument("--out", default=str(ROOT / "bench_out" / "p1_bench.json"))
    args = ap.parse_args()

    enable_dpi_awareness()

    region = None
    if args.region:
        region = tuple(int(x) for x in args.region.split(","))
        if len(region) != 4:
            print("--region 需要 4 个整数：left,top,width,height")
            return 2

    print("=" * 84)
    print("P1 基准：截屏 + OCR")
    print("=" * 84)
    print(f"  CPU {platform.processor()} / {__import__('os').cpu_count()} 逻辑核")

    print("\n[显示器]")
    for m in list_monitors():
        print(f"  {m}")

    print("\n[截屏自检]")
    chk = selfcheck(region=region, backend=args.backend)
    print(f"  {json.dumps(chk, ensure_ascii=False)}")
    if not chk.get("ok"):
        print("\n  !! 截屏自检未通过：")
        print(f"     {chk.get('error') or chk.get('warning')}")

    report: dict = {
        "monitors": [str(m) for m in list_monitors()],
        "selfcheck": chk,
        "region": region,
        "cpu_count": __import__("os").cpu_count(),
        "cpu": platform.processor(),
    }

    # -- 准备截屏器 -------------------------------------------------------- #
    cap = ScreenCapturer(monitor_index=args.monitor, backend=args.backend, region=region)
    report.update(backend=cap.backend, monitor=str(cap.monitor), capture_region=cap.region)

    # -- 准备 OCR 输入 ----------------------------------------------------- #
    # 默认用真实抓到的帧：只有纯 OCR 与端到端用**同一个输入**，两个数字才可比。
    # （曾经踩过：纯 OCR 用合成图 4 行、端到端用真实屏幕 33 行，看起来凭空多出 280ms。）
    cap_frame = cap.grab()
    if cap_frame is None:
        print("\n!! 抓帧失败，无法继续")
        return 3
    if args.synthetic:
        probe = make_test_image()
        src_desc = f"合成中文图 {probe.shape}（仅测 OCR 吞吐，端到端仍用真实帧）"
    elif args.image:
        from PIL import Image

        probe = np.asarray(Image.open(args.image).convert("RGB"))[:, :, ::-1].copy()
        src_desc = f"图片 {args.image} {probe.shape}"
    else:
        probe = cap_frame
        src_desc = f"真实屏幕帧 {probe.shape}"

    print(f"\n[OCR 输入] {src_desc}")

    # -- 线程数标定 -------------------------------------------------------- #
    print("\n[线程数标定]（Intel 大小核上默认 auto 会慢 3-4 倍，必须标定）")
    t0 = time.perf_counter()
    n_threads = tune_intra_threads(
        probe, det_limit_side_len=args.det_limit, use_cache=not args.retune
    )
    print(f"  intra_op_num_threads = {n_threads}   （耗时 {time.perf_counter()-t0:.1f}s）")
    report["intra_threads"] = n_threads

    engine = OcrEngine(det_limit_side_len=args.det_limit, intra_threads=n_threads)

    print("\n[预热] OCR 首次调用含 ONNX 会话初始化，不计入基准 ...")
    t0 = time.perf_counter()
    engine.read(probe)
    print(f"  冷启动首帧 {(time.perf_counter() - t0) * 1000:.0f}ms")

    # -- 纯 OCR ------------------------------------------------------------ #
    print(f"\n[纯 OCR 基准] {args.runs} 次")
    ocr_ms = []
    last = None
    for _ in range(args.runs):
        last = engine.read(probe)
        ocr_ms.append(last.elapsed_ms)
    s_ocr = stats(ocr_ms)
    print(fmt("OCR (CPU)", s_ocr, BUDGET["ocr"]) + "  -> " + verdict(s_ocr["p95"], BUDGET["ocr"]))
    print(f"\n  识别结果（{len(last.lines)} 行）：")
    for ln in last.lines[:12]:
        print(f"    {ln.score:.2f}  {ln.text}")
    if len(last.lines) > 12:
        print(f"    ... 另有 {len(last.lines) - 12} 行")
    report["sample_lines"] = last.texts[:30]

    # -- 纯截屏 ------------------------------------------------------------ #
    print(f"\n[纯截屏基准] {args.runs} 次")
    cap_ms = []
    frame = None
    for _ in range(args.runs):
        t0 = time.perf_counter()
        frame = cap.grab()
        cap_ms.append((time.perf_counter() - t0) * 1000)
    s_cap = stats(cap_ms)
    print(fmt(f"截屏 ({cap.backend})", s_cap, BUDGET["capture"]) + "  -> " + verdict(s_cap["p95"], BUDGET["capture"]))

    # -- 端到端（截屏→OCR） ------------------------------------------------ #
    print(f"\n[端到端 截屏→OCR 基准] {args.runs} 次")
    e2e_ms = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        f = cap.grab()
        if f is not None:
            engine.read(f)
        e2e_ms.append((time.perf_counter() - t0) * 1000)
    s_e2e = stats(e2e_ms)
    print(fmt("截屏 + OCR", s_e2e, BUDGET["e2e"]) + "  -> " + verdict(s_e2e["p95"], BUDGET["e2e"]))
    cap.close()

    # -- 汇总 -------------------------------------------------------------- #
    tts_budget = 600 - s_e2e["p95"]
    n_lines = len(last.lines)
    per_line = s_ocr["p50"] / max(n_lines, 1)
    print("\n" + "=" * 84)
    print("汇总")
    print("=" * 84)
    print(f"  识别行数       : {n_lines} 行（OCR 耗时与行数近似线性，换区域要重新看这个数）")
    print(f"  OCR p95        : {s_ocr['p95']:>7.1f} ms   {verdict(s_ocr['p95'], BUDGET['ocr'])}"
          f"   （{per_line:.1f} ms/行）")
    print(f"  截屏 p95       : {s_cap['p95']:>7.1f} ms   {verdict(s_cap['p95'], BUDGET['capture'])}")
    print(f"  截屏+OCR p95   : {s_e2e['p95']:>7.1f} ms   {verdict(s_e2e['p95'], BUDGET['e2e'])}")
    print(f"  -> 留给 TTS 的首包预算（按端到端 600ms 倒推）：{tts_budget:.0f} ms")
    print(f"  帧尺寸 / 区域  : {None if frame is None else frame.shape} / {cap.region}")

    report.update(
        ocr_ms=s_ocr,
        capture_ms=s_cap,
        e2e_ms=s_e2e,
        line_count=n_lines,
        ms_per_line=round(per_line, 1),
        tts_budget_p95_ms=round(tts_budget, 1),
        frame_shape=list(frame.shape) if frame is not None else None,
        ocr_params=default_params(args.det_limit),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告已写入 {out}")
    return 0


if __name__ == "__main__":
    code = main()
    # dxcam/comtypes 在解释器退出时会触发一次访问违例告警（不影响结果），
    # 这里直接退出以拿到干净的退出码，便于 CI / 脚本判断。
    sys.stdout.flush()
    os._exit(code)
