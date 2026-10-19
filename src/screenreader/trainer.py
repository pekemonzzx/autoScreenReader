"""GPT-SoVITS 微调训练（可被 CLI / 界面 / 脚本共用）。

从最早的 tools/tts_train.py 搬进来并参数化——训练参数现在来自**声库档案**
（`voices/<名字>/train.yaml`），不再写死在代码里。

必须用 **v2pro 整合包的内嵌 Python** 运行（v3lora 那个包不支持 v2ProPlus）：

    D:\\GPT-SoVITS\\GPT-SoVITS-v2pro-20250604\\runtime\\python.exe -s -m ... train ...

六个踩过的坑（都已在下面处理，勿删）：
  1. 集成包里的 `runtime/Lib/site-packages/users.pth` 硬编码了打包者的绝对路径，
     解压到别处就 `ModuleNotFoundError: No module named 'text'`，
     而且 PYTHONPATH 救不了（该运行时带 `python39._pth`）。→ repair_users_pth
  2. prepare_datasets/*.py 既有 `from text.cleaner`（在 GPT_SoVITS/ 下）
     又有 `from tools.my_utils`（在根目录下），两个路径都要进 PYTHONPATH。
  3. v2Pro/v2ProPlus 多一步 `2-get-sv.py`，缺了会 `assert os.path.exists(path7)` 失败。
  4. 1a/1c 的产物是分片的（`2-name2text-0.txt`），要按 WebUI 的逻辑合并。
  5. 实验目录必须是 `<GSV_ROOT>/logs/<exp>`，否则存 checkpoint 时目录不存在，
     **要跑完一轮才失败**。
  6. `-c "<多行代码>"` 在 Windows 上会被参数解析搞坏，一律写脚本文件。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)

LogFn = Callable[[str], None]

# 各版本对应的预训练权重（跟整合包 config.py 保持一致）
PRETRAINED = {
    "v2ProPlus": {
        "s2G": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
        "s2D": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth",
        "s1": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        "s2config": "GPT_SoVITS/configs/s2v2ProPlus.json",
    },
    "v2Pro": {
        "s2G": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
        "s2D": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth",
        "s1": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        "s2config": "GPT_SoVITS/configs/s2v2Pro.json",
    },
}
COMMON = {
    "bert": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    "hubert": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    "sv": "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
    "s1config": "GPT_SoVITS/configs/s1longer-v2.yaml",
}
SOVITS_OUT = {
    "v2ProPlus": "SoVITS_weights_v2ProPlus",
    "v2Pro": "SoVITS_weights_v2Pro",
}
GPT_OUT = {
    "v2ProPlus": "GPT_weights_v2ProPlus",
    "v2Pro": "GPT_weights_v2Pro",
}


# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    ok: bool
    exp: str
    sovits: List[Path]
    gpt: List[Path]
    seconds: float
    steps: List[str]


def log_default(msg: str) -> None:
    log.info(msg)
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
def repair_users_pth(root: Path, say: LogFn = log_default) -> None:
    """修正整合包里 users.pth 的硬编码路径（见模块顶部坑 1）。"""
    pth = root / "runtime" / "Lib" / "site-packages" / "users.pth"
    if not pth.exists():
        return
    # **内嵌 Python 是隔离模式（带 python39._pth），PYTHONPATH 会被忽略** ——
    # 唯一能给它加搜索路径的办法就是这里的 users.pth。
    #
    # 为什么必须加 eres2net：`2-get-sv.py`（v2Pro/v2ProPlus 的 1B-SV）里写死了
    #     now_dir = os.getcwd()
    #     sys.path.append(f"{now_dir}/GPT_SoVITS/eres2net")
    #     from ERes2NetV2 import ERes2NetV2
    # 我们为了绕开"整合包目录不可写"把 cwd 换成了 work，
    # 那个相对路径就落空了 -> ModuleNotFoundError: No module named 'ERes2NetV2'。
    # 走 users.pth 给绝对路径，cwd 在哪都成立。
    subs = ("GPT_SoVITS/BigVGAN", "tools", "tools/asr", "GPT_SoVITS",
            "GPT_SoVITS/eres2net", "tools/uvr5")
    want = "\n".join([str(root)] + [f"{root}/{s}" for s in subs])
    try:
        cur = pth.read_text(encoding="utf-8", errors="replace")
    except Exception:
        cur = ""
    if cur.strip() != want.strip():
        try:
            pth.write_text(want + "\n", encoding="utf-8")
            say(f"  已修正 users.pth 的硬编码路径 -> {root}")
        except Exception as e:
            say(f"  !! users.pth 修正失败：{e}")


def setup_env(root: Path) -> None:
    global _GSV_ROOT
    _GSV_ROOT = root
    if not (root / "runtime" / "python.exe").exists():
        raise SystemExit(f"找不到整合包内嵌 Python：{root}\\runtime\\python.exe")
    repair_users_pth(root)
    os.chdir(root)
    os.environ["PATH"] = (
        f"{root}{os.pathsep}{root / 'runtime'}{os.pathsep}" + os.environ.get("PATH", "")
    )
    # prepare_datasets 的脚本靠 cwd 找同级模块（`sys.path.append(now_dir)` 之类），
    # 我们把 cwd 换成可写目录后那些就失效了，所以这里用 PYTHONPATH 补齐：
    #   root                    -> tools.my_utils
    #   root/GPT_SoVITS         -> text.cleaner
    #   root/GPT_SoVITS/eres2net-> ERes2NetV2（2-get-sv.py 要）
    pp = [str(root), str(root / "GPT_SoVITS"), str(root / "GPT_SoVITS" / "eres2net")]
    if os.environ.get("PYTHONPATH"):
        pp.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pp)
    os.environ.setdefault("hz", "25hz")
    os.environ["_CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for p in (root, root / "tools", root / "tools" / "uvr5"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def run(cmd: str, label: str, logfile: Path, say: LogFn = log_default,
        cwd: Optional[Path] = None) -> int:
    say(f"  $ {cmd}")
    t0 = time.time()
    logfile.parent.mkdir(parents=True, exist_ok=True)

    # **cwd 必须显式给一个可写目录**，不能让它继承。
    #
    # setup_env() 里有 `os.chdir(root)`，root 是整合包根目录 —— 而整合包目录
    # 在有些环境下**不可写**。整合包里的 GPT_SoVITS/prepare_datasets/1-get-text.py
    # 的 my_save() 会 `torch.save(..., "<时间戳>.pth")` 写**相对路径**，
    # 于是落到 cwd（整合包根目录）上，直接：
    #     RuntimeError: File 1792150703.6197020.pth cannot be opened.
    # 而且它被 try/except 吞掉，只表现为"1A 退出码=0 但什么都没生成"。
    #
    # logfile.parent 就是 work 目录（调用方都传 work / "log_*.txt"），
    # 它本来就是我们挑出来的可写目录。
    work_dir = cwd or logfile.parent

    # 命令里的脚本路径是**相对**的（`GPT_SoVITS/prepare_datasets/1-get-text.py`），
    # 原来靠 cwd=整合包根目录 才能找到。既然我们现在把 cwd 换成可写目录，
    # 就得把这些相对路径补成绝对路径，否则就是：
    #     can't open file '...\work\train_xxx\GPT_SoVITS\prepare_datasets\1-get-text.py'
    if _GSV_ROOT is not None:
        cmd = cmd.replace("GPT_SoVITS/", f"{_GSV_ROOT}/GPT_SoVITS/") \
                 .replace("GPT_SoVITS\\", f"{_GSV_ROOT}/GPT_SoVITS/")

    with logfile.open("w", encoding="utf-8", errors="replace") as fh:
        from .proc import popen_hidden

        # 走 popen_hidden：训练虽然主要在命令行跑，但界面上也会触发，
        # 不隐藏窗口的话每跑一步都闪一下黑窗。
        p = popen_hidden(cmd, shell=True, stdout=fh, stderr=subprocess.STDOUT,
                         cwd=str(work_dir))
        rc = p.wait()
    say(f"    {label} 退出码={rc}  用时 {time.time() - t0:.0f}s  日志={logfile.name}")
    if rc != 0:
        say("    !! 失败，日志末尾：")
        try:
            for line in logfile.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]:
                say(f"       {line}")
        except Exception:
            pass
    return rc


# exp_dir_of 定义在文件后半部分（带可写目录回退），这里不再重复定义。


# --------------------------------------------------------------------------- #
def merge_parts(opt_dir: Path, all_parts: int = 1, say: LogFn = log_default) -> None:
    """合并分片产物（见模块顶部坑 4）。"""
    txt_path = opt_dir / "2-name2text.txt"
    if not txt_path.exists():
        opt = []
        for i in range(all_parts):
            p = opt_dir / f"2-name2text-{i}.txt"
            if p.exists():
                opt += p.read_text(encoding="utf-8").strip("\n").split("\n")
                p.unlink()
        if opt:
            txt_path.write_text("\n".join(opt) + "\n", encoding="utf-8")

    sem_path = opt_dir / "6-name2semantic.tsv"
    if not sem_path.exists():
        opt = ["item_name\tsemantic_audio"]
        for i in range(all_parts):
            p = opt_dir / f"6-name2semantic-{i}.tsv"
            if p.exists():
                opt += p.read_text(encoding="utf-8").strip("\n").split("\n")
                p.unlink()
        if len(opt) > 1:
            sem_path.write_text("\n".join(opt) + "\n", encoding="utf-8")


def run_prepare(root: Path, version: str, exp: str, list_path: Path, wav_dir: Path,
                work: Path, say: LogFn = log_default) -> int:
    """1A/1B/1B-SV/1C：文本→音素、音频→HuBERT、说话人向量、语义 token。"""
    py = sys.executable
    pre = dict(PRETRAINED[version], **COMMON)

    # **全部换成绝对路径**。
    #
    # 整合包这些脚本是从环境变量/配置里读预训练模型路径的
    # （`bert_pretrained_dir` / `cnhubert_base_dir` / `sv_path` /
    #   `pretrained_s2G` / `s2config_path` / `pretrained_s1`），
    # 而 PRETRAINED/COMMON 里写的是**相对路径**，靠 cwd=整合包根目录才成立。
    #
    # 但那些脚本同时会往 cwd 写相对路径的临时文件（`my_save` 的
    # `torch.save(fea, "<时间戳>.pth")`），所以 cwd 又必须是可写目录 ——
    # 而整合包目录不一定可写。两个要求冲突，以前只能靠建 junction 绕。
    #
    # 既然路径本来就是从环境变量读的，直接给绝对路径就同时满足两边：
    # 模型按绝对路径找到，临时文件写进可写的 cwd。**不再需要 junction。**
    pre = {k: str((root / v).resolve()) for k, v in pre.items()}
    opt_dir = exp_dir_of(root, exp)
    base = {
        "inp_text": str(list_path),
        "inp_wav_dir": str(wav_dir),
        "exp_name": exp,
        "opt_dir": str(opt_dir),
        "i_part": "0",
        "all_parts": "1",
        "is_half": "True",
        "version": version,
    }

    say("\n[1A] 文本 -> 音素")
    # `bert_path` 和 `bert_pretrained_dir` 都要给，而且都要**绝对路径**：
    #   - `1-get-text.py` 读 bert_pretrained_dir
    #   - `text/chinese2.py` 里 g2pw 的 model_source 读的是 `bert_path`
    #     （os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/...")）
    # 默认值都是相对路径、靠 cwd 成立；cwd 换到 work 后就会报
    #   OSError: Incorrect path_or_model_id: 'GPT_SoVITS/pretrained_models/...'
    os.environ.update(dict(base, bert_pretrained_dir=pre["bert"],
                           bert_path=pre["bert"]))
    if run(f'"{py}" -s GPT_SoVITS/prepare_datasets/1-get-text.py', "1A", work / "log_1a.txt", say) != 0:
        return 1

    say("\n[1B] 音频 -> HuBERT 特征")
    os.environ.update(dict(base, cnhubert_base_dir=pre["hubert"]))
    if run(f'"{py}" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py', "1B", work / "log_1b.txt", say) != 0:
        return 1

    # v2Pro / v2ProPlus 独有的一步（见模块顶部坑 3）
    say("\n[1B-SV] 说话人向量 (7-sv_cn)")
    os.environ.update(dict(base, cnhubert_base_dir=pre["hubert"], sv_path=pre["sv"]))
    if run(f'"{py}" -s GPT_SoVITS/prepare_datasets/2-get-sv.py', "1B-SV", work / "log_1b_sv.txt", say) != 0:
        return 1

    say("\n[1C] 语义 token")
    os.environ.update({
        "inp_text": str(list_path), "exp_name": exp, "opt_dir": str(opt_dir),
        "pretrained_s2G": pre["s2G"], "s2config_path": pre["s2config"],
        "i_part": "0", "all_parts": "1", "is_half": "True", "version": version,
    })
    if run(f'"{py}" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py', "1C", work / "log_1c.txt", say) != 0:
        return 1

    say("\n[1ABC] 合并分片")
    merge_parts(opt_dir, 1, say)

    say("\n[1ABC] 产物校验")
    ok = True
    for name in ("2-name2text.txt", "3-bert", "4-cnhubert", "5-wav32k", "7-sv_cn", "6-name2semantic.tsv"):
        p = opt_dir / name
        exists = p.exists()
        size = ""
        if p.is_file():
            size = f"{p.stat().st_size / 1e6:.1f} MB"
        elif p.is_dir():
            size = f"{len(list(p.glob('*')))} 个文件"
        say(f"    {'OK ' if exists else '!! '}{name:<24} {size}")
        ok = ok and exists
    if (opt_dir / "2-name2text.txt").exists():
        say(f"    音素条数 {len((opt_dir / '2-name2text.txt').read_text(encoding='utf-8').strip().splitlines())}")
    if (opt_dir / "6-name2semantic.tsv").exists():
        n = len((opt_dir / "6-name2semantic.tsv").read_text(encoding="utf-8").strip().splitlines()) - 1
        say(f"    语义条数 {n}")
    return 0 if ok else 1


def run_sovits(root: Path, version: str, exp: str, work: Path, train, say: LogFn = log_default) -> int:
    """2A：SoVITS 训练。"""
    say("\n[2A] SoVITS 训练")
    import json as _json

    pre = {k: str((root / v).resolve()) for k, v in PRETRAINED[version].items()}
    s2_dir = exp_dir_of(root, exp)
    (s2_dir / f"logs_s2_{version}").mkdir(parents=True, exist_ok=True)
    data = _json.loads((root / pre["s2config"]).read_text(encoding="utf-8"))
    if not train.fp16:
        data["train"]["fp16_run"] = False
    data["train"].update({
        "batch_size": train.s2_batch, "epochs": train.s2_epoch, "text_low_lr_rate": 0.4,
        "pretrained_s2G": pre["s2G"], "pretrained_s2D": pre["s2D"],
        "if_save_latest": True, "if_save_every_weights": True,
        "save_every_epoch": train.save_every, "gpu_numbers": "0",
        "grad_ckpt": False, "lora_rank": 32,
    })
    data["model"]["version"] = version
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = str(s2_dir)

    # **save_weight_dir 必须是绝对路径**。
    # 它是相对的（`SoVITS_weights_v2ProPlus`）时会跟着**子进程的 cwd** 走 ——
    # 而我们为了绕开"整合包目录不可写"把 cwd 换成了 work，
    # 于是权重落进 work 里，后面 refresh 去整合包目录找不到。
    # 同时 **save_every 不能大于 epochs**，否则 `epoch % save_every == 0`
    # 永远不成立，一个权重都不存（实测 epochs=2 / save_every=4 就是这个结果）。
    every = max(1, min(int(train.save_every), int(train.s2_epoch)))
    data["train"]["save_every_epoch"] = every
    wdir = _weights_dir(work, SOVITS_OUT[version], say)
    data["save_weight_dir"] = str(wdir)
    data["name"] = exp
    data["version"] = version
    tmp = work / "tmp_s2.json"
    tmp.write_text(_json.dumps(data), encoding="utf-8")
    say(f"  batch={train.s2_batch}  epochs={train.s2_epoch}  fp16={train.fp16}  "
        f"每 {every} epoch 存权重  -> {wdir}")
    return run(f'"{sys.executable}" -s GPT_SoVITS/s2_train.py --config "{tmp}"',
               "2A", work / "log_s2.txt", say)


def run_gpt(root: Path, version: str, exp: str, work: Path, train, say: LogFn = log_default) -> int:
    """2B：GPT 训练。"""
    say("\n[2B] GPT 训练")
    import yaml

    s1_dir = exp_dir_of(root, exp)
    (s1_dir / "logs_s1").mkdir(parents=True, exist_ok=True)
    (s1_dir / f"logs_s1_{version}").mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load((root / COMMON["s1config"]).read_text(encoding="utf-8"))
    if not train.fp16:
        data["train"]["precision"] = "32"
    every1 = max(1, min(int(train.save_every), int(train.s1_epoch)))
    data["train"].update({
        "batch_size": train.s1_batch, "epochs": train.s1_epoch,
        "save_every_n_epoch": every1,
        "if_save_every_weights": True, "if_save_latest": True, "if_dpo": False,
        "half_weights_save_dir": str(_weights_dir(work, GPT_OUT[version], say)),
        "exp_name": exp,
    })
    data["pretrained_s1"] = str((root / PRETRAINED[version]["s1"]).resolve())
    data["train_semantic_path"] = str(s1_dir / "6-name2semantic.tsv")
    data["train_phoneme_path"] = str(s1_dir / "2-name2text.txt")
    data["output_dir"] = str(s1_dir / f"logs_s1_{version}")
    tmp = work / "tmp_s1.yaml"
    tmp.write_text(yaml.safe_dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")
    say(f"  batch={train.s1_batch}  epochs={train.s1_epoch}  权重输出={GPT_OUT[version]}/")
    return run(f'"{sys.executable}" -s GPT_SoVITS/s1_train.py --config_file "{tmp}"',
               "2B", work / "log_s1.txt", say)


# --------------------------------------------------------------------------- #
# 当前整合包根目录 —— setup_env() 会设置，run() 用它把命令里的相对脚本路径补成绝对路径
_GSV_ROOT: Optional[Path] = None

_TRAIN_BASE_CACHE: Dict[str, Path] = {}


def _writable_dir(cands, label: str, say=print) -> Path:
    """在候选里挑第一个**真的能写**的目录（写探针文件验证）。"""
    last = None
    for d in cands:
        try:
            d.mkdir(parents=True, exist_ok=True)
            # exist_ok 对"已存在但不可写"不报错，所以要真写一下才算数
            probe = d / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return d
        except Exception as e:
            last = (d, e)
            say(f"  （{d} 不可写：{type(e).__name__}，试下一个）")
    raise RuntimeError(
        f"找不到可写的{label}目录。最后尝试 {last[0] if last else '?'}："
        f"{last[1] if last else '?'}")


def _train_base(root: Path, say=print) -> Path:
    """训练用的**可写基目录** —— `work/` 和 `logs/` 都建在它下面。

    为什么需要：整合包目录在有些环境里**不可写** —— 实测本机 Python 进程
    往里建目录会 `PermissionError: [WinError 5] 拒绝访问`，而 PowerShell 却可以。
    训练要建 `<root>/work/train_<exp>` 和 `<root>/logs/<exp>` 两个目录，
    默认位置都在整合包内，于是双双失败。

    候选顺序：
      1. 环境变量 SCREENREADER_TRAIN_WORK
      2. <GSV_ROOT>                  —— 和 WebUI 一致（可写时行为完全不变）
      3. <GSV_ROOT>/../_sr_train     —— 整合包旁边的兄弟目录，通常可写
      4. <系统临时目录>/screenreader_train
    """
    key = str(root)
    if key in _TRAIN_BASE_CACHE:
        return _TRAIN_BASE_CACHE[key]

    env = os.environ.get("SCREENREADER_TRAIN_WORK")
    cands = []
    if env:
        cands.append(Path(env))
    cands.append(root)
    cands.append(root.parent / "_sr_train")
    cands.append(Path(tempfile.gettempdir()) / "screenreader_train")

    base = _writable_dir(cands, "训练基", say)
    if base != root:
        say(f"  （整合包目录不可写，训练输出统一改到 {base}）")
    _TRAIN_BASE_CACHE[key] = base
    return base


def _pick_work_dir(root: Path, exp: str, say=print) -> Path:
    """训练工作目录（放可写基目录下）。"""
    return _train_base(root, say) / "work" / f"train_{exp}"


def exp_dir_of(root: Path, exp: str, say=print) -> Path:
    """实验输出目录（放可写基目录下）。

    训练脚本要往里存 checkpoint，所以目录必须先存在；
    默认位置 `<GSV_ROOT>/logs/<exp>` 不可写时自动换到别处。
    """
    d = _train_base(root, say) / "logs" / exp
    d.mkdir(parents=True, exist_ok=True)
    return d


def _link_gsv_into(work: Path, root: Path, say=print) -> None:
    """在 work 里给整合包的 GPT_SoVITS 建一个 junction。

    **为什么必须这么做**：整合包的 prepare_datasets 脚本里写死了**相对路径**，
    例如 `1-get-text.py` 要找 `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`，
    所以子进程的 cwd 得是整合包根目录；可同一个脚本还会往 cwd 写相对路径的
    `.pth`（`my_save`），而整合包目录**不可写** —— 两个要求互相打架：

        cwd = 整合包根目录  ->  找得到模型，但写不进去 (RuntimeError: .pth cannot be opened)
        cwd = 可写的 work    ->  写得进去，但找不到模型 (FileNotFoundError: pretrained_models/...)

    解法：cwd 用 work，并在 work 里建一个 `GPT_SoVITS` 的 junction 指向整合包。
    读走 junction（找得到模型），写落在 work（可写）。

    junction 用 `mklink /J`，**不需要管理员权限**。
    """
    link = work / "GPT_SoVITS"
    target = root / "GPT_SoVITS"
    if link.exists() or not target.exists():
        return
    # 必须先建出 work 本身，否则 mklink 会报 "系统找不到指定的路径"
    try:
        work.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        say(f"  !! 建 work 目录失败：{e}")
        return

    ok = False
    try:
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                           capture_output=True, text=True, errors="replace")
        ok = r.returncode == 0
        if not ok:
            say(f"  （mklink 不可用：{(r.stdout or '').strip()[:60]}，改用复制）")
    except Exception as e:
        say(f"  （mklink 异常：{e}，改用复制）")

    if ok:
        say("  （已在 work 里建立 GPT_SoVITS 链接：模型读整合包，输出写 work）")
        return

    # **兜底：把脚本真正需要的那几样镜像过来（优先硬链接）。**
    #
    # 为什么不能建 junction 就得兜底：脚本里写死了 cwd 相对路径，例如
    #   `1-get-text.py` -> g2pw 的 `model_dir="G2PWModel/"`（在 GPT_SoVITS/text 下）
    #   `2-get-sv.py`   -> `sys.path.append(f"{os.getcwd()}/GPT_SoVITS/eres2net")`
    # 也就是说 **cwd 下必须真的存在那些文件**。
    #
    # 用硬链接（os.link）：同一个磁盘卷上不需要管理员权限，**也不额外占空间** ——
    # 这点很关键，因为 `text/G2PWModel/g2pW.onnx` 就有 607 MB。
    # 跨卷或硬链接失败时退回真复制。
    import os as _os

    want = [
        ("GPT_SoVITS/eres2net", ("*.py",)),
        ("GPT_SoVITS/text/G2PWModel", None),
    ]
    total_links = total_copies = 0
    for rel, patterns in want:
        src = root / rel
        if not src.is_dir():
            say(f"  !! 找不到 {src}，相关步骤会失败")
            continue
        dst = work / rel
        try:
            dst.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            say(f"  !! 建 {dst} 失败：{e}")
            continue
        files = []
        if patterns:
            for pat in patterns:
                files += list(src.glob(pat))
        else:
            files = [p for p in src.rglob("*") if p.is_file()]
        for f in files:
            relp = f.relative_to(src)
            if "__pycache__" in relp.parts:
                continue
            target = dst / relp
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                continue
            try:
                _os.link(f, target)
                total_links += 1
            except Exception:
                try:
                    import shutil as _sh

                    _sh.copy2(f, target)
                    total_copies += 1
                except Exception:
                    pass
    say(f"  （已在 work 里镜像 GPT_SoVITS 的必要部分："
        f"硬链接 {total_links} 个、复制 {total_copies} 个；"
        f"脚本按 cwd 相对路径就能找到）")


def _weights_dir(work: Path, name: str, say=print) -> Path:
    """权重输出目录（**绝对路径**）。

    训练脚本里 `save_weight_dir` / `half_weights_save_dir` 是相对路径时，
    会跟着子进程的 cwd 走。我们为了绕开"整合包目录不可写"把 cwd 换成了 work，
    所以这里统一给绝对路径，放在 work 的兄弟目录 `weights/` 下。

    理想位置是 `<GSV_ROOT>/<name>`（refresh 默认去那儿找），但整合包目录
    不一定可写；能写就用它，不能写就退回 work 旁边，稍后由调用方复制过去。
    """
    d = work.parent / "weights" / name
    try:
        d.mkdir(parents=True, exist_ok=True)
        return d
    except Exception as e:
        say(f"  !! 建权重目录失败 {d}：{e}")
        return work / name


def train_profile(
    profile,
    exp: Optional[str] = None,
    profile_dir: Optional[Path] = None,
    *,
    skip_prepare: bool = False,
    skip_sovits: bool = False,
    skip_gpt: bool = False,
    say: LogFn = log_default,
) -> TrainResult:
    """按声库档案训练。profile 是 screenreader.voice.VoiceProfile。

    `profile_dir` 用于解析档案内的相对路径（如 `train.list`）。
    """
    t_all = time.time()
    train = profile.train
    root = Path(train.gsv_root)
    exp = exp or profile.name
    steps: List[str] = []

    if train.version not in PRETRAINED:
        raise SystemExit(
            f"训练暂不支持版本 {train.version!r}（只做了 {sorted(PRETRAINED)}）。\n"
            "注意：只有 v2 / v2ProPlus 有 ONNX 推理路径，打包要用。"
        )

    if profile_dir is not None:
        lst, wav_dir = profile.train_paths(Path(profile_dir))
    else:
        lst = Path(train.list_path) if train.list_path else None
        wav_dir = Path(train.wav_dir) if train.wav_dir else None

    if not lst or not lst.exists():
        raise SystemExit(f"训练标注不存在：{lst}")
    if not wav_dir or not wav_dir.exists():
        raise SystemExit(f"切片音频目录不存在：{wav_dir}")

    setup_env(root)

    lst = lst.resolve()
    wav_dir = wav_dir.resolve()
    lines = [l for l in lst.read_text(encoding="utf-8").splitlines() if l.strip()]
    say("=" * 78)
    say(f"训练声库「{profile.display}」  version={train.version}")
    say(f"  标注 {len(lines)} 条   {lst}")
    say(f"  音频目录           {wav_dir}")
    say(f"  实验名             {exp}")
    say("=" * 78)

    work = _pick_work_dir(root, exp, say)
    _link_gsv_into(work, root, say)

    if not skip_prepare:
        if run_prepare(root, train.version, exp, lst, wav_dir, work, say) != 0:
            return TrainResult(False, exp, [], [], time.time() - t_all, steps + ["1ABC"])
        steps.append("1ABC")
    else:
        say("\n[1ABC] 跳过")

    if not skip_sovits:
        if run_sovits(root, train.version, exp, work, train, say) != 0:
            return TrainResult(False, exp, [], [], time.time() - t_all, steps + ["2A"])
        steps.append("2A")
    else:
        say("\n[2A] SoVITS 训练  跳过")

    if not skip_gpt:
        if run_gpt(root, train.version, exp, work, train, say) != 0:
            return TrainResult(False, exp, [], [], time.time() - t_all, steps + ["2B"])
        steps.append("2B")
    else:
        say("\n[2B] GPT 训练  跳过")

    # 权重可能在两个地方：整合包目录（能写的话）或 work 旁边的 weights/
    sovits, gpt = [], []
    for d in (root / SOVITS_OUT[train.version],
              work.parent / "weights" / SOVITS_OUT[train.version]):
        if d.is_dir():
            sovits += sorted(d.glob(f"{exp}*.pth"))
    for d in (root / GPT_OUT[train.version],
              work.parent / "weights" / GPT_OUT[train.version]):
        if d.is_dir():
            gpt += sorted(d.glob(f"{exp}*.ckpt"))

    # 能写整合包目录的话，顺手把权重放回去 —— refresh / 推理默认去那儿找
    for f in sovits + gpt:
        want = root / f.parent.name / f.name
        if want.parent.is_dir() and not want.exists():
            try:
                import shutil as _sh
                _sh.copy2(f, want)
                say(f"  （已复制权重到 {want.parent.name}\\）")
            except Exception:
                pass

    say("\n" + "=" * 78)
    say("产物")
    say("=" * 78)
    for f in sovits + gpt:
        say(f"  {f.parent.name}/{f.name}  ({f.stat().st_size / 1e6:.0f} MB)")
    if not sovits and not gpt:
        say("  !! 没有产出权重 —— 检查 epochs 和 save_every 的关系")
    say(f"\n训练日志在 {work}")
    return TrainResult(True, exp, sovits, gpt, time.time() - t_all, steps)
