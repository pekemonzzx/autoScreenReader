"""读屏台引擎的无界面联调 / 排障工具。

把 BoardEngine 跑起来，实时打印卡片事件和当前卡片表。
用途：验证识别→合成→播放链路，以及在目标机上排查"为什么某个区域读不出来"。

用法（需要 TTS 服务在跑）：
    $env:PYTHONPATH='src'
    .\\.venv\\Scripts\\python tools\\board_demo.py --region 0,0,760,300 --duration 30
    .\\.venv\\Scripts\\python tools\\board_demo.py --region "0,0,760,300" --region "0,700,760,200"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screenreader.board import BoardConfig  # noqa: E402
from screenreader.config import AppConfig  # noqa: E402
from screenreader.worker import BoardEngine, EngineConfig  # noqa: E402

ICON = {
    "card_added": "＋ 新句",
    "card_ready": "✓ 合成好",
    "card_failed": "✗ 合成失败",
    "card_rearmed": "↻ 需重合成",
    "card_died": "－ 消失",
    "play_started": "▶ 播放",
    "play_stopped": "⏹ 停止",
    "play_finished": "■ 播完",
    "region_added": "▣ 新区域",
    "error": "! 出错",
    "started": "▶ 引擎启动",
    "stopped": "■ 引擎停止",
    "cleared": "⌫ 清空",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="读屏台引擎联调")
    ap.add_argument("--region", action="append", required=True,
                    help="区域 left,top,width,height；可重复给多个")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--interval", type=float, default=0.5, help="轮询间隔秒")
    ap.add_argument("--granularity", default="line", choices=["line", "sentence"])
    ap.add_argument("--misses", type=int, default=3, help="连续几轮看不到算消失")
    ap.add_argument("--config", default=None)
    ap.add_argument("--auto-play", action="store_true", help="新句合成就自动播（测试用）")
    ap.add_argument("--json", action="store_true", help="输出机器可读的结论")
    args = ap.parse_args()

    cfg = AppConfig.load(Path(args.config) if args.config else None)
    cfg.apply_log_level()

    engine = BoardEngine(
        cfg,
        EngineConfig(poll_interval_s=args.interval, segment_enabled=(args.granularity == "sentence")),
        board_cfg=BoardConfig(granularity=args.granularity, destroy_after_misses=args.misses),
    )
    for spec in args.region:
        parts = [int(x) for x in spec.split(",")]
        if len(parts) != 4:
            print(f"区域格式错误：{spec}")
            return 2
        engine.add_region(f"区域{len(engine.store.regions) + 1}", tuple(parts))

    print("=" * 84)
    print(f"读屏台引擎联调  区域 {len(engine.store.regions)} 个  轮询 {args.interval}s  "
          f"粒度 {args.granularity}  消失阈值 {args.misses} 轮")
    for r in engine.store.regions:
        print(f"  {r.label()}  {r.color}")
    print("=" * 84)

    seen = {"added": 0, "ready": 0, "failed": 0, "died": 0}

    def on_event(e: dict) -> None:
        t = e.get("type", "")
        if t == "card_added":
            seen["added"] += 1
        elif t == "card_ready":
            seen["ready"] += 1
        elif t == "card_failed":
            seen["failed"] += 1
        elif t == "card_died":
            seen["died"] += 1
        if args.json:
            return
        tag = ICON.get(t)
        if not tag:
            return
        detail = ""
        if t in ("card_added", "card_died", "card_ready", "card_failed"):
            detail = e.get("text", "")[:46]
        elif t == "card_ready":
            detail = f"{'缓存' if e.get('from_cache') else str(e.get('ms', 0)) + 'ms'}"
        elif t == "play_started":
            detail = e.get("text", "")[:46]
        elif t == "error":
            detail = f"{e.get('where')} {str(e.get('error'))[:60]}"
        print(f"  [{time.strftime('%H:%M:%S')}] {tag:<10} {detail}")
        if args.auto_play and t == "card_ready":
            engine.play_card(e["card_id"])

    engine.on_event = on_event
    engine.start()

    t0 = time.time()
    try:
        while time.time() - t0 < args.duration:
            time.sleep(2)
            if args.json:
                continue
            cards = engine.store.sorted_cards()
            if cards:
                print(f"  --- {time.strftime('%H:%M:%S')} 当前 {len(cards)} 张 ---")
                for c in cards[:8]:
                    mark = {"ready": "●", "pending": "○", "failed": "✗"}.get(c.audio_state, "?")
                    if c.play_state == "playing":
                        mark = "▶"
                    print(f"      {mark} {c.display(56)}")
    except KeyboardInterrupt:
        print("\n中断…")
    finally:
        engine.stop()

    st = engine.stats_snapshot()
    print("\n" + "=" * 84)
    print("结果")
    print("=" * 84)
    print(f"  轮询次数        {st['polls']}   单轮耗时 {st['ocr_ms']}ms")
    print(f"  新识别句子      {seen['added']}")
    print(f"  合成成功/失败    {st['synth_ok']} / {st['synth_fail']}")
    print(f"  缓存命中        {st['cache_hits']}")
    print(f"  自动消亡句子    {seen['died']}")
    print(f"  退出时残留卡片  {st['cards']}")
    if args.json:
        print(json.dumps({"stats": st, "seen": seen}, ensure_ascii=False))

    ok = seen["added"] > 0 and st["synth_ok"] + st["cache_hits"] > 0
    print("\n  " + ("链路正常 ✅" if ok else "!! 没有识别到内容或合成失败，检查区域坐标与 TTS 服务"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
