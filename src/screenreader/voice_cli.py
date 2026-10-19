"""`screenreader voice` 子命令：声库的增删查改、训练、自检、切换。

设计目标（用户需求）：**训练、训练配置、推理配置都独立出来，后续可配置替换**。

    voices/
      voice01/
        voice.json      元数据 + 推理配置
        train.yaml      训练配置（换语料只改这里）
        ref.wav         参考音频
        train.list      训练标注
        tts_infer.yaml  自动生成给 api_v2.py

一次完整流程：

    python -m screenreader voice new  myvoice --list <train.list> --wav-dir <切片目录>
    python -m screenreader voice pick-ref myvoice      # 自动挑一条当参考音频
    python -m screenreader voice verify   myvoice      # 自检（含 prompt 是否对得上）
    python -m screenreader voice train    myvoice      # 训练
    python -m screenreader voice refresh  myvoice      # 训练完刷新权重指向
    python -m screenreader voice use      myvoice      # 切到它
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import wave
from pathlib import Path
from typing import List, Optional

from .config import AppConfig, default_config_path
from .voice import DEFAULT_GSV_ROOT, SUPPORTED_VERSIONS, VoiceProfile, VoiceStore

log = logging.getLogger(__name__)


def _store(args) -> VoiceStore:
    cfg = AppConfig.load(Path(args.config) if getattr(args, "config", None) else None)
    return VoiceStore(cfg.voices_root())


def _cfg(args) -> AppConfig:
    return AppConfig.load(Path(args.config) if getattr(args, "config", None) else None)


def _dur(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        return -1.0


def _cjk_ratio(s: str) -> float:
    if not s:
        return 0.0
    n = len(re.findall(r"[\u4e00-\u9fff]", s))
    return n / len(s)


# --------------------------------------------------------------------------- #
def cmd_prefix() -> str:
    """打包后要显示 `读屏台.exe --voice ...`，目标机没有 Python。"""
    import sys

    if getattr(sys, "frozen", False):
        return f'"{Path(sys.executable).name}" --voice'
    return "python -m screenreader voice"


def cmd_list(args) -> int:
    store = _store(args)
    cfg = _cfg(args)
    names = store.list_names()
    active = cfg.tts.voice
    c = cmd_prefix()
    if not names:
        print(f"还没有声库。目录：{store.root}")
        print(f"安装声库包：{c} install <xxx.voicepack.zip>")
        print("或者把 .voicepack.zip 直接解压到这个目录下。")
        return 0
    print(f"声库目录：{store.root}")
    print(f"{'':2} {'名字':<14} {'版本':<12} {'参考音频':<8} {'可分发':<8} 显示名")
    for n in names:
        try:
            p = store.load(n)
            d = store.dir_of(n)
            ref = p.resolve(p.ref_audio, d) if p.ref_audio else ""
            okref = "有" if ref and Path(ref).exists() else "缺"
            sc = "是" if p.is_self_contained(d) else "否"
            print(f"{'*' if n == active else ' '}  {n:<14} {p.version:<12} {okref:<8} {sc:<8} {p.display}")
        except Exception as e:
            print(f"   {n:<14} !! 档案损坏：{e}")
    print("\n（* = 当前使用；可分发 = 权重与参考音频都在档案里，能打成声库包）")
    print(f"  切换：{c} use <名字>")
    return 0


def cmd_pack(args) -> int:
    """把声库打包成可独立分发的压缩包。"""
    store = _store(args)
    out = Path(args.out) if args.out else (store.root / "_packs")
    try:
        p = store.pack(args.name, out, include_train=args.include_train)
    except Exception as e:
        print(f"打包失败：{e}")
        return 1
    size = p.stat().st_size
    print(f"\n声库包已生成：{p}")
    print(f"  大小 {size / 1024 / 1024:.1f} MB")
    print("\n  分发给别人后，对方只要：")
    print(f"    python -m screenreader voice install \"{p.name}\"")
    print("  或者直接把这个 zip 解压到程序的 voices\\ 目录下（会自动识别）。")
    return 0


def cmd_install(args) -> int:
    """从声库包安装。"""
    store = _store(args)
    try:
        store.install(Path(args.pack), overwrite=args.overwrite)
    except Exception as e:
        print(f"安装失败：{e}")
        return 1
    print("\n  自检一下：")
    print(f"    python -m screenreader voice verify {_peek_name(Path(args.pack))}")
    return 0


def _peek_name(pack: Path) -> str:
    try:
        import zipfile

        with zipfile.ZipFile(pack) as z:
            import json as _j

            return _j.loads(z.read("manifest.json").decode("utf-8"))["name"]
    except Exception:
        return "<名字>"


def cmd_scan(args) -> int:
    """扫描 voices/ 目录，把手工解压进来的声库档案登记并校验。"""
    store = _store(args)
    names = store.list_names()
    print(f"在 {store.root} 下发现 {len(names)} 个声库：")
    for n in names:
        try:
            p = store.load(n)
            d = store.dir_of(n)
            g, s = p.weight_paths(d)
            bad = [str(x) for x in (g, s) if x is None or not x.exists()]
            ref = Path(p.resolve(p.ref_audio, d)) if p.ref_audio else None
            if ref is None or not ref.exists():
                bad.append("参考音频")
            print(f"  [{'OK' if not bad else '!!'}] {n:<16} {p.display}"
                  + (f"   缺：{bad}" if bad else ""))
        except Exception as e:
            print(f"  [!!] {n:<16} 档案损坏：{e}")
    return 0


def cmd_show(args) -> int:
    store = _store(args)
    p = store.load(args.name)
    d = store.dir_of(args.name)
    print(f"声库「{p.display}」  名字={p.name}")
    print(f"  版本      : {p.version}")
    print(f"  创建      : {p.created or '—'}")
    print(f"  语料来源  : {p.source or '—'}")
    print(f"  参考音频  : {p.resolve(p.ref_audio, d) or '（未设置）'}")
    print(f"  参考文本  : {p.prompt_text or '（未设置）'}")
    print(f"  GPT 权重  : {p.gpt_weights or '（未设置）'}")
    print(f"  SoVITS权重: {p.sovits_weights or '（未设置）'}")
    print(f"  API       : {p.api_url}")
    t = p.train
    print("  训练配置  :")
    print(f"      gsv_root = {t.gsv_root}")
    print(f"      list     = {t.list_path}")
    print(f"      wav_dir  = {t.wav_dir}")
    print(f"      s2 batch/epoch = {t.s2_batch}/{t.s2_epoch}    s1 batch/epoch = {t.s1_batch}/{t.s1_epoch}")
    print(f"  API 启动命令：")
    print(f"      {p.api_launch_command(d)}")
    return 0


def cmd_new(args) -> int:
    store = _store(args)
    # 标注文件复制进档案（很小，且这是你以后要编辑的东西）；
    # 切片音频目录通常几十 MB，保持外部路径引用即可。
    list_path = ""
    if args.list:
        src = Path(args.list).resolve()
        if not src.exists():
            print(f"标注文件不存在：{src}")
            return 1
        dst_dir = store.dir_of(args.name)
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst_dir / "train.list")
        list_path = "train.list"

    p = store.create(
        args.name,
        display_name=args.display or "",
        version=args.version,
        list_path=list_path,
        wav_dir=str(Path(args.wav_dir).resolve()) if args.wav_dir else "",
        gsv_root=args.gsv_root,
        source=args.source or "",
    )
    store.write_train_yaml(p)
    d = store.dir_of(args.name)
    print(f"已创建声库「{p.display}」 -> {d}")
    if list_path:
        print(f"  标注已复制进档案：{d / 'train.list'}"
              f"（{sum(1 for _ in (d / 'train.list').read_text(encoding='utf-8').splitlines())} 行）")
    print("  下一步：")
    print(f"    1) 参考音频： python -m screenreader voice pick-ref {args.name}")
    print(f"       或手动：   python -m screenreader voice set-ref {args.name} --wav a.wav --text \"音频里说的话\"")
    print(f"    2) 训练：     python -m screenreader voice train {args.name}")
    print(f"    3) 刷新权重： python -m screenreader voice refresh {args.name}")
    print(f"    4) 自检：     python -m screenreader voice verify {args.name}")
    print(f"    5) 启用：     python -m screenreader voice use {args.name}")
    return 0


def cmd_pick_ref(args) -> int:
    """从训练标注里自动挑一条干净片段当参考音频。

    注意用 `train_paths()` 解析路径：档案里的 `train.list` 是**相对档案目录**的，
    直接 `Path(train.list_path)` 会在当前工作目录下找，永远找不到
    （实测报"训练标注不存在：train.list"）。
    """
    store = _store(args)
    p = store.load(args.name)
    d = store.dir_of(args.name)
    lst, wav_dir = p.train_paths(d)
    if not lst or not lst.exists():
        print(f"训练标注不存在：{lst}")
        return 1

    cands = []
    for line in lst.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        wav = Path(parts[0])
        if not wav.is_absolute():
            base = wav_dir or d
            wav = base / wav.name
        if not wav.exists():
            continue
        text = parts[3].strip()
        dur = _dur(wav)
        if not (args.min_sec <= dur <= args.max_sec):
            continue
        if _cjk_ratio(text) < 0.85 or len(text) < 8:
            continue
        cands.append((abs(dur - 4.5), dur, wav, text))

    # --calm：按「情绪平和度」重排。
    #
    # 为什么需要：参考音频的情绪会被一起克隆。挑到"太太好玩了！"这种，
    # 之后念任何正常文本都带那股劲儿。叙述型参考要的是音高稳、能量稳、
    # 没有感叹号。评分实现在 screenreader.calm。
    if getattr(args, "calm", False) and cands:
        from .calm import calm_score

        scored = []
        for rank_key, dur, wav, text in cands:
            try:
                sc, detail = calm_score(wav, text)
            except Exception:
                continue
            if sc <= -1e8:
                continue
            scored.append((-sc, dur, wav, text, detail))
        if scored:
            scored.sort(key=lambda x: x[0])
            best = scored[0]
            print(f"已按「叙述型/平和」评分挑选（共 {len(scored)} 条候选）")
            for s, dur, wav, text, dt in scored[:5]:
                print(f"    {-s:6.1f}  {dur:.1f}s  f0波动={dt.get('f0_cv')} "
                      f"能量波动={dt.get('env_cv')} 语速={dt.get('rate')}")
                print(f"           {text[:50]}")
            cands = [(s, dur, wav, text) for s, dur, wav, text, _ in scored]

    if not cands:
        print(f"在 {args.min_sec}~{args.max_sec} 秒范围内没找到合适的干净片段")
        return 1
    cands.sort(key=lambda x: x[0])
    _, dur, wav, text = cands[0]
    dst = d / "ref.wav"
    shutil.copyfile(wav, dst)
    p.ref_audio = "ref.wav"
    p.prompt_text = text
    store.save(p)
    print(f"已选取参考音频（{dur:.1f}s）：{wav.name}")
    print(f"  -> {dst}")
    print(f"  参考文本：{text}")
    print("  !! 这段文本必须与音频内容一致，否则每句开头会多出听不懂的音。")
    print(f"     请跑 python -m screenreader voice verify {args.name} 确认。")
    return 0


def cmd_set_ref(args) -> int:
    store = _store(args)
    p = store.load(args.name)
    d = store.dir_of(args.name)
    src = Path(args.wav)
    if not src.exists():
        print(f"音频不存在：{src}")
        return 1
    dst = d / "ref.wav"
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    p.ref_audio = "ref.wav"
    p.prompt_text = args.text.strip()
    store.save(p)
    print(f"参考音频 -> {dst}   （{_dur(dst):.1f}s）")
    print(f"参考文本 -> {p.prompt_text}")
    print(f"请跑 python -m screenreader voice verify {args.name} 确认两者对得上。")
    return 0


def cmd_refresh(args) -> int:
    """训练完之后刷新权重指向：自动找出最新的 SoVITS/GPT 权重并写进档案。"""
    store = _store(args)
    p = store.load(args.name)
    root = Path(p.train.gsv_root)
    from .trainer import GPT_OUT, SOVITS_OUT

    if p.version not in SOVITS_OUT:
        print(f"版本 {p.version} 没有对应的权重目录映射")
        return 1
    # 权重可能在两个地方：
    #   1. <GSV_ROOT>/<dir>              —— 和 WebUI 一致（可写时权重就存这儿）
    #   2. <GSV_ROOT>/../_sr_train/work/weights/<dir>
    #      或 <项目>/data/_tmp/screenreader_train/work/weights/<dir>
    #      —— 整合包目录**不可写**时，trainer 会把权重落到可写目录里
    # 以前只找第 1 处，于是"训练明明成功了，refresh 却说没找到权重"。
    def _cands(sub: str):
        out = [root / sub]
        for base in (root.parent / "_sr_train",
                     root.parent,
                     Path(__file__).resolve().parents[2] / "data" / "_tmp" / "screenreader_train"):
            out.append(base / "work" / "weights" / sub)
        return out

    sovits, gpts = [], []
    for d in _cands(SOVITS_OUT[p.version]):
        if d.is_dir():
            sovits += sorted(d.glob(f"{p.name}*.pth"))
    for d in _cands(GPT_OUT[p.version]):
        if d.is_dir():
            gpts += sorted(d.glob(f"{p.name}*.ckpt"))
    sovits = sorted(set(sovits))
    gpts = sorted(set(gpts))
    if not sovits or not gpts:
        print("没找到权重。检查下列位置：")
        for d in _cands(SOVITS_OUT[p.version]) + _cands(GPT_OUT[p.version]):
            print(f"  {d}")
        print("（名字前缀必须与声库名一致）")
        return 1

    def _last_epoch(files):
        import re as _re

        # 注意：不能用 [eE](\d+)——"voice01_e4_s124" 里会把 "voic**e01**" 匹配成 1，
        # 结果 e4 和 e8 算出来一样大，取到的是列表里第一个。
        # 必须要求 e 前面有分隔符（_ 或 -）。
        def key(f):
            m = _re.search(r"[_\-]e(\d+)", f.stem, _re.IGNORECASE)
            return int(m.group(1)) if m else -1

        return max(files, key=key)

    s, g = _last_epoch(sovits), _last_epoch(gpts)
    p.sovits_weights = f"{SOVITS_OUT[p.version]}/{s.name}"
    p.gpt_weights = f"{GPT_OUT[p.version]}/{g.name}"
    store.save(p)
    # 已经装了独立运行时的话，推理配置要指向**运行时**，不是训练用的整合包目录
    runtime_root = None
    try:
        rt = (_cfg(args).tts.runtime_root or "").strip()
        if rt and Path(rt).exists():
            runtime_root = Path(rt)
    except Exception:
        pass
    cfg_path = p.write_api_config(store.dir_of(p.name), runtime_root=runtime_root)
    print("已刷新权重指向：")
    print(f"  SoVITS : {p.sovits_weights}")
    print(f"  GPT    : {p.gpt_weights}")
    print(f"  推理运行时 : {runtime_root or p.train.gsv_root}")
    print(f"  推理配置 -> {cfg_path}")
    print(f"  启动命令：{p.api_launch_command(store.dir_of(p.name))}")
    print()
    print("提示：想让这个声库能独立分发（自带权重），跑：")
    print(f"  python -m screenreader voice pack {p.name}")
    return 0


def cmd_verify(args) -> int:
    """自检：文件是否齐、权重是否存在、参考音频与文本是否可能配串。"""
    from .config import AppConfig
    from .cache import wav_info

    store = _store(args)
    p = store.load(args.name)
    d = store.dir_of(args.name)
    root = Path(p.train.gsv_root)
    ok = True

    def chk(label, cond, detail=""):
        nonlocal ok
        print(f"  [{'OK' if cond else 'FAIL'}] {label:<26} {detail}")
        ok = ok and bool(cond)

    print("=" * 78)
    print(f"声库自检：{p.display}（{p.name}）  版本 {p.version}")
    print("=" * 78)

    print("\n[档案]")
    chk("voice.json", (d / "voice.json").exists(), str(d))
    chk("train.yaml", (d / "train.yaml").exists())
    chk("版本受支持", p.version in SUPPORTED_VERSIONS, p.version)

    print("\n[参考音频 / 参考文本]")
    ref = Path(p.resolve(p.ref_audio, d)) if p.ref_audio else None
    chk("ref_audio 已设置", bool(p.ref_audio))
    chk("ref_audio 存在", bool(ref and ref.exists()), str(ref))
    if ref and ref.exists():
        sr, ch_, dur = wav_info(ref.read_bytes())
        chk("参考音频时长 2~12 秒", 2.0 <= dur <= 12.0, f"{dur:.1f}s @ {sr}Hz")
        print("      提示：太短克隆不稳，太长会拖慢推理；3~6 秒最合适")
    chk("prompt_text 已设置", bool(p.prompt_text), (p.prompt_text or "")[:34])
    chk("prompt_text 含足够汉字", _cjk_ratio(p.prompt_text) >= 0.6,
        f"汉字占比 {_cjk_ratio(p.prompt_text):.2f}")

    print("\n[权重]")
    # 用 weight_paths() 解析，而不是简单 root / rel ——
    # 整合包目录不可写时权重会落在可写目录的 work/weights/ 下，
    # 只看 root 会误报"权重不存在"，让人以为训练白跑了。
    gp, sp = p.weight_paths(store.dir_of(p.name))
    chk("GPT 权重已设置", bool(p.gpt_weights))
    chk("GPT 权重存在", bool(gp and gp.exists()), str(gp or ""))
    chk("SoVITS 权重已设置", bool(p.sovits_weights))
    chk("SoVITS 权重存在", bool(sp and sp.exists()), str(sp or ""))

    print("\n[推理配置]")
    y = d / "tts_infer.yaml"
    chk("tts_infer.yaml 存在", y.exists(), "（没有就 voice refresh）")
    if y.exists():
        txt = y.read_text(encoding="utf-8", errors="replace")
        chk("yaml 里 version 正确", f"version: {p.version}" in txt)
        if p.gpt_weights:
            chk("yaml 指向当前 GPT 权重", Path(p.gpt_weights).name in txt)
        if p.sovits_weights:
            chk("yaml 指向当前 SoVITS 权重", Path(p.sovits_weights).name in txt)

    print("\n[训练]")
    lst, wav = p.train_paths(d)
    chk("train.list 已设置", bool(p.train.list_path), p.train.list_path)
    chk("train.list 存在", bool(lst and lst.exists()), str(lst or ""))
    chk("wav_dir 存在", bool(wav and wav.exists()), str(wav or ""))
    chk("gsv_root 存在", root.exists(), str(root))
    if lst and lst.exists():
        n = len([x for x in lst.read_text(encoding="utf-8").splitlines() if x.strip()])
        print(f"      标注 {n} 条")

    # prompt_text 与 ref_audio 是否一致，只能靠 ASR 判断——这里给出命令
    print("\n[★ 最容易被忽略的一项：prompt_text 是否与 ref_audio 内容一致]")
    print("  两者不一致时，模型会把错位内容续写到**每句话开头**（听起来像听不懂的怪音）。")
    print("  光看波形看不出来，必须把参考音频转成文字比对。请把下面这条贴到 PowerShell 执行：")
    print()
    if ref and ref.exists():
        from .voice import VoiceProfile as _VP

        print("  " + _asr_cmd(ref))
    else:
        print("  （先设置参考音频）")
    print()
    print("  如果转写结果和下面这句对不上，就用 voice set-ref 改成转写出来的文本：")
    print(f"      {p.prompt_text!r}")

    print("\n" + "=" * 78)
    print("检查通过 ✅" if ok else "存在问题 ❌（按上面 FAIL 项处理）")
    print("=" * 78)
    if ok and ref and ref.exists():
        print("\n启动推理服务：")
        print("  " + p.api_launch_command(d))
    return 0 if ok else 1


def _asr_cmd(ref: Path) -> str:
    from .voice import DEFAULT_GSV_ROOT as _R

    stage = Path(_R).parent.parent  # 只是占位，实际用仓库 data 目录
    from .config import _app_data_base

    stage = _app_data_base() / "data" / "_asr_stage"
    R = DEFAULT_GSV_ROOT
    return (
        f'$R="{R}"; $env:PATH="$R;$R\\runtime;$env:PATH"; '
        f'New-Item -ItemType Directory -Force -Path "{stage}\\in","{stage}\\out" | Out-Null; '
        f'Copy-Item "{ref}" "{stage}\\in\\" -Force; '
        f'& "$R\\runtime\\python.exe" -s "$R\\tools\\asr\\funasr_asr.py" '
        f'-i "{stage}\\in" -o "{stage}\\out" -s large -l zh; '
        f'Get-Content (Get-ChildItem "{stage}\\out\\*.list").FullName -Encoding UTF8'
    )


def cmd_use(args) -> int:
    store = _store(args)
    if not store.exists(args.name):
        print(f"没有这个声库：{args.name}")
        return 1
    cfg = _cfg(args)
    cfg.tts.voice = args.name
    path = Path(args.config) if getattr(args, "config", None) else default_config_path()
    cfg.save(path)
    print(f"已把当前声库切到「{args.name}」（写入 {path}）")
    print("重启界面 / 重新启动引擎后生效。")
    return 0


def cmd_train(args) -> int:
    store = _store(args)
    p = store.load(args.name)
    # 训练配置以 train.yaml 为准（可以在文件里改）
    y = store.read_train_yaml(args.name)
    if y:
        from .voice import TrainConfig

        p.train = TrainConfig.from_dict({**p.train.__dict__, **y})
    for k in ("s2_batch", "s2_epoch", "s1_batch", "s1_epoch"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(p.train, k, v)

    from .trainer import train_profile

    res = train_profile(
        p,
        exp=args.exp,
        profile_dir=store.dir_of(args.name),
        skip_prepare=args.skip_1abc,
        skip_sovits=args.skip_s2,
        skip_gpt=args.skip_s1,
    )
    if not res.ok:
        print("\n训练失败，看上面的日志末尾。")
        return 1
    print(f"\n训练完成，用时 {res.seconds:.0f} 秒 -> {res.sovits + res.gpt}")
    print(f"下一步：python -m screenreader voice refresh {args.name}")
    return 0


def cmd_delete(args) -> int:
    store = _store(args)
    if not args.yes:
        print(f"要删除声库「{args.name}」？确认请加 --yes")
        return 1
    return 0 if store.delete(args.name) else 1


# --------------------------------------------------------------------------- #
def add_parser(sub) -> None:
    v = sub.add_parser("voice", help="声库管理（列表/新建/训练/自检/切换）")
    vs = v.add_subparsers(dest="voice_cmd", required=True)

    def common(p):
        p.add_argument("-c", "--config", default=None, help="配置文件路径")

    p = vs.add_parser("list", help="列出声库")
    common(p)
    p.set_defaults(func=cmd_list)

    p = vs.add_parser("show", help="显示声库详情")
    common(p)
    p.add_argument("name")
    p.set_defaults(func=cmd_show)

    p = vs.add_parser("new", help="新建声库档案")
    common(p)
    p.add_argument("name")
    p.add_argument("--display", default="", help="显示名")
    p.add_argument("--version", default="v2ProPlus", choices=list(SUPPORTED_VERSIONS))
    p.add_argument("--list", default="", help="训练标注 .list")
    p.add_argument("--wav-dir", default="", help="切片音频目录")
    p.add_argument("--gsv-root", default=DEFAULT_GSV_ROOT, help="GPT-SoVITS 整合包目录")
    p.add_argument("--source", default="", help="语料来源说明")
    p.set_defaults(func=cmd_new)

    p = vs.add_parser("pick-ref", help="从训练集自动挑一条当参考音频")
    common(p)
    p.add_argument("name")
    p.add_argument("--min-sec", type=float, default=3.0)
    p.add_argument("--max-sec", type=float, default=6.0)
    p.add_argument("--calm", action="store_true",
                   help="按「叙述型/情绪平和」评分挑选（音高稳、能量稳、无感叹号）。"
                        "参考音频的情绪会被一起克隆，念正常文本建议开这个")
    p.set_defaults(func=cmd_pick_ref)

    p = vs.add_parser("set-ref", help="手动指定参考音频与文本")
    common(p)
    p.add_argument("name")
    p.add_argument("--wav", required=True)
    p.add_argument("--text", required=True, help="音频里实际说的话")
    p.set_defaults(func=cmd_set_ref)

    p = vs.add_parser("refresh", help="训练后刷新权重指向并重建推理配置")
    common(p)
    p.add_argument("name")
    p.set_defaults(func=cmd_refresh)

    p = vs.add_parser("verify", help="自检声库（含 prompt_text 一致性提示）")
    common(p)
    p.add_argument("name")
    p.set_defaults(func=cmd_verify)

    p = vs.add_parser("use", help="切换当前使用的声库")
    common(p)
    p.add_argument("name")
    p.set_defaults(func=cmd_use)

    p = vs.add_parser("train", help="按档案训练（用整合包内嵌 Python 运行）")
    common(p)
    p.add_argument("name")
    p.add_argument("--exp", default=None, help="实验名（默认与声库同名）")
    p.add_argument("--s2-batch", type=int, default=None)
    p.add_argument("--s2-epoch", type=int, default=None)
    p.add_argument("--s1-batch", type=int, default=None)
    p.add_argument("--s1-epoch", type=int, default=None)
    p.add_argument("--skip-1abc", action="store_true")
    p.add_argument("--skip-s2", action="store_true")
    p.add_argument("--skip-s1", action="store_true")
    p.set_defaults(func=cmd_train)

    p = vs.add_parser("delete", help="删除声库档案")
    common(p)
    p.add_argument("name")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_delete)

    p = vs.add_parser("pack", help="打包成可独立分发的声库包（自带权重和参考音频）")
    common(p)
    p.add_argument("name")
    p.add_argument("--out", default=None, help="输出目录（默认 voices\\_packs）")
    p.add_argument("--include-train", action="store_true",
                   help="把训练标注也打进去（默认不打，训练用的东西又大又没用）")
    p.set_defaults(func=cmd_pack)

    p = vs.add_parser("install", help="安装声库包")
    common(p)
    p.add_argument("pack", help="xxx.voicepack.zip")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_install)

    p = vs.add_parser("scan", help="扫描 voices\\ 目录，校验手工放入的声库")
    common(p)
    p.set_defaults(func=cmd_scan)
