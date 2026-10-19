"""清理项目目录里的临时/测试残留。

**默认只干跑（列出要删的东西），加 --apply 才真删。**

分类：
  A 测试产生的临时目录   sr_testcfg_* / sr_uismoke_* / pip-* / tmp*
  B 运行日志与中间产物   *.log / _r2.json / tts_req.json / config.json.bak*
  C 我的验证音频与分析    AB* / cmp2_* / compare_* / newref_* / trunc_* / _src48k*

**不动的东西**（都是要留的）：
  · 源语料        data/*.m4a（48kHz）、..._16k.wav
  · 人工标注明细  data/voice*_train_clean.list(.raw)
  · 配置          data/config.json
  · 模型与缓存    tools/asr（1.1GB FunASR）、data/probe、data/tts_cache
  · 构建产物      build/ dist/ release_*/
  · 源码与文档    src/ tests/ tools/*.py packaging/ vendor/
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A 类：这些前缀的**目录**是测试/工具跑出来的临时目录
TEMP_DIR_PREFIXES = ("sr_testcfg_", "sr_uismoke_", "pip-build-tracker", "pip-ephem",
                     "pip-install", "pip-metadata", "pip-unpack", "tmp")
# 额外的空/无用目录
TEMP_DIRS = ("bench_out",)

# B 类：日志与中间产物（项目根 + data/）
LOG_PATTERNS = ("*.log",)
JUNK_FILES = (
    "debug.log",
    "data/_r2.json",
    "data/tts_req.json",
    "data/config.json.bak2",
)

# C 类：我为了验证/分析生成的音频
VERIFY_WAV_PATTERNS = (
    "data/AB*.wav", "data/cmp2_*.wav", "data/compare_*.wav", "data/newref_*.wav",
    "data/_src48k*.wav", "data/voice_test.wav",
)
VERIFY_DIRS = ("data/trunc_test", "data/trunc_asr")

# 白名单：绝不删。支持通配符（源语料的具体文件名不在代码里写死）
KEEP = (
    "data/*.m4a",                            # 源语料（48kHz）
    "data/sample_source.txt",                 # 指向源语料的一行配置
    "data/config.json",
    "data/config.example.json",
    "data/voice*_train_clean.list",
    "data/voice*_train_clean.list.raw",
)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def dir_size(p: Path) -> int:
    try:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except Exception:
        return 0


def collect() -> list:
    """返回 [(路径, 类型, 大小)]。**只产出确实存在的。**"""
    items = []

    # A：临时目录
    for p in sorted(ROOT.iterdir()):
        if not p.is_dir():
            continue
        if p.name in TEMP_DIRS or any(p.name.startswith(x) for x in TEMP_DIR_PREFIXES):
            items.append((p, "临时目录", dir_size(p)))

    # B：日志与中间产物
    for pat in LOG_PATTERNS:
        for p in sorted(ROOT.glob(pat)):
            if p.is_file():
                items.append((p, "日志", p.stat().st_size))
        for p in sorted((ROOT / "data").glob(pat)):
            if p.is_file():
                items.append((p, "日志", p.stat().st_size))
    for rel in JUNK_FILES:
        p = ROOT / rel
        if p.exists():
            items.append((p, "中间产物", p.stat().st_size))

    # C：验证音频
    for pat in VERIFY_WAV_PATTERNS:
        for p in sorted(ROOT.glob(pat)):
            if p.is_file():
                items.append((p, "验证音频", p.stat().st_size))
    for rel in VERIFY_DIRS:
        p = ROOT / rel
        if p.is_dir():
            items.append((p, "验证目录", dir_size(p)))

    # 安全检查 + 去重：白名单绝不能出现，同一个路径只留一次
    # 白名单支持通配符（`data/*.m4a` 这种），所以拆成"精确集合"+"通配模式"两套判断。
    keeps = {(ROOT / k).resolve() for k in KEEP if "*" not in k}
    keep_globs = tuple(k for k in KEEP if "*" in k)
    safe = []
    seen = set()
    for p, kind, sz in items:
        rp = p.resolve()
        rel = None
        try:
            rel = p.relative_to(ROOT).as_posix()
        except Exception:
            pass
        if rp in keeps or (rel and any(fnmatch.fnmatch(rel, g) for g in keep_globs)):
            print(f"  !! 跳过白名单文件：{p.relative_to(ROOT)}")
            continue
        # 再确认一次：必须在项目目录内，且不是项目根自己
        if rp == ROOT or ROOT.resolve() not in rp.parents:
            print(f"  !! 跳过越界路径：{rp}")
            continue
        if rp in seen:          # debug.log 之类的会被两个类别同时命中
            continue
        seen.add(rp)
        safe.append((p, kind, sz))
    return safe


def main() -> int:
    ap = argparse.ArgumentParser(description="清理临时/测试残留")
    ap.add_argument("--apply", action="store_true", help="真的删（不加只干跑）")
    args = ap.parse_args()

    items = collect()
    total = sum(sz for _p, _k, sz in items)

    print("=" * 76)
    print(f"项目根目录：{ROOT}")
    print(f"{'将删除' if args.apply else '将要删除（干跑）'} {len(items)} 项，共 {human(total)}")
    print("=" * 76)

    by_kind: dict = {}
    for p, k, sz in items:
        by_kind.setdefault(k, []).append((p, sz))

    for kind, lst in by_kind.items():
        sz = sum(s for _p, s in lst)
        print(f"\n  [{kind}] {len(lst)} 项，{human(sz)}")
        show = lst[:8]
        for p, s in show:
            print(f"    {p.relative_to(ROOT)}")
        if len(lst) > len(show):
            print(f"    …… 还有 {len(lst) - len(show)} 项")

    print("\n" + "=" * 76)
    print("保留（不动）：")
    for k in KEEP:
        if "*" in k:
            hit = list(ROOT.glob(k))
            if hit:
                print(f"  {k}  （匹配 {len(hit)} 个）")
        elif (ROOT / k).exists():
            print(f"  {k}")
    print("  tools/asr、data/probe、data/tts_cache（模型与缓存）")
    print("  build/、dist/、release_*/（构建产物）")
    print("  src/、tests/、tools/~.py、packaging/、vendor/（源码与依赖）")
    print("=" * 76)

    if not args.apply:
        print("\n这是干跑。确认无误后加 --apply 真删。")
        return 0

    print("\n开始删除…")
    ok = 0
    for p, kind, _sz in items:
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=False)
            else:
                p.unlink()
            ok += 1
        except Exception as e:
            print(f"  !! 删不掉 {p.relative_to(ROOT)}：{e}")
    print(f"\n完成：删除 {ok}/{len(items)} 项，释放 {human(total)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
