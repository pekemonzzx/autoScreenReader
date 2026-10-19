"""OCR 线程数标定工具。

用途：为当前机器（以及每一台目标机）找出 ONNX Runtime 的最优 intra_op 线程数。

为什么需要：intra_op_num_threads 默认 auto 会在 Intel 大小核
（如 i7-14700KF：8 P-core + 12 E-core）上把线程池铺满全部逻辑核，
栅栏同步需要等最慢的 E-core 线程。本机实测慢 3-4 倍。

用法：
    .\\.venv\\Scripts\\python tools\\tune_ocr.py                 # 用合成图，扫描候选值
    .\\.venv\\Scripts\\python tools\\tune_ocr.py --real          # 用真实屏幕内容
    .\\.venv\\Scripts\\python tools\\tune_ocr.py --region 520,140,520,140
    .\\.venv\\Scripts\\python tools\\tune_ocr.py --clear-cache   # 清掉已缓存的标定结果
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.disable(logging.INFO)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from screenreader.capture import ScreenCapturer, enable_dpi_awareness  # noqa: E402
from screenreader.ocr import (  # noqa: E402
    THREAD_CANDIDATES,
    OcrEngine,
    _cache_path,
    _heuristic_threads,
    _machine_key,
    default_params,
)

TEXTS = [
    "欢迎来到直播间，今天我们来玩一个全新的游戏",
    "弹幕：主播这个操作也太秀了吧 666",
    "当前任务：击败boss并收集三件稀有装备",
    "物品栏 金币 12345 生命值 87/100 魔法值 42/60",
]


def make_image(w=900, h=300, size=26) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", size)
    img = Image.new("RGB", (w, h), (248, 248, 250))
    d = ImageDraw.Draw(img)
    y, i = int(size * 0.8), 0
    while y + size < h:
        d.text((int(size * 1.2), y), TEXTS[i % len(TEXTS)], font=font, fill=(28, 28, 32))
        y += int(size * 2.2)
        i += 1
    return np.asarray(img)[:, :, ::-1].copy()


def sweep(image: np.ndarray, det_limit: int, repeat: int, use_cls: bool) -> tuple[int, list]:
    rows = []
    best_n, best_ms = None, float("inf")
    for n in THREAD_CANDIDATES:
        e = OcrEngine(det_limit_side_len=det_limit, use_cls=use_cls, intra_threads=n)
        e.warmup(image, 2)
        ts = []
        for _ in range(repeat):
            t = time.perf_counter()
            r = e.read(image)
            ts.append((time.perf_counter() - t) * 1000)
        p50 = sorted(ts)[len(ts) // 2]
        rows.append((n, p50, len(r.lines)))
        if p50 < best_ms:
            best_n, best_ms = n, p50
    return best_n, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="用真实屏幕内容做标定图")
    ap.add_argument("--region", default=None, help="left,top,width,height")
    ap.add_argument("--det-limit", type=int, default=960)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--no-cls", action="store_true")
    ap.add_argument("--clear-cache", action="store_true")
    args = ap.parse_args()

    enable_dpi_awareness()

    if args.clear_cache:
        p = _cache_path()
        if p and p.exists():
            p.unlink()
            print(f"已清除缓存：{p}")
        else:
            print("没有可清除的缓存")
        raise SystemExit(0)

    region = tuple(int(x) for x in args.region.split(",")) if args.region else None
    if region and len(region) != 4:
        print("--region 需要 4 个整数")
        raise SystemExit(2)

    if args.real:
        cap = ScreenCapturer(backend="dxgi", region=region)
        image = cap.grab()
        cap.close()
        desc = f"真实屏幕 {image.shape}"
    else:
        image = make_image()
        desc = f"合成图 {image.shape}"

    print(f"标定图：{desc}")
    print(f"候选线程数：{THREAD_CANDIDATES}   每档重复 {args.repeat} 次\n")

    best, rows = sweep(image, args.det_limit, args.repeat, not args.no_cls)
    base = max(r[1] for r in rows)
    for n, p50, lines in rows:
        flag = "  <== 最优" if n == best else ""
        print(f"  intra={n:<3} p50={p50:>7.1f}ms  行数={lines:<3} 相对最差提速 {base/p50:>4.1f}x{flag}")

    print(f"\n最优 intra_op_num_threads = {best}")

    # 写入缓存，供 OcrEngine.auto() / tune_intra_threads() 下次直接命中
    cp = _cache_path()
    if cp is not None:
        key = _machine_key(args.det_limit, image.shape)
        try:
            data = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
            data[key] = best
            cp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"已写入缓存：{cp}")
        except Exception as e:
            print(f"缓存写入失败：{e}")
    else:
        print("没有可写的缓存目录，标定结果不会被记住")

    print(f"启发式兜底值（标定失败时用）= {_heuristic_threads()}")
    print(f"默认参数 = {json.dumps(default_params(args.det_limit), ensure_ascii=False)}")
    print("\n提示：标定结果按 (机器, CPU 核数, 图片尺寸) 缓存，换机器/换区域需重跑。")
