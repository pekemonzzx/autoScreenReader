"""GPT-SoVITS v2ProPlus 微调一键训练（无界面）。

对应 WebUI「1-GPT-SoVITS-TTS」页签的「一键三连」，把 5 个步骤串起来跑：

    1A  1-get-text.py            文本 -> 音素
    1B  2-get-hubert-wav32k.py   音频 -> HuBERT 特征
    1C  3-get-semantic.py        语义 token
    2A  s2_train.py              SoVITS 训练
    2B  s1_train.py              GPT 训练

必须用 **v2pro 整合包**的内嵌 Python 运行（v3lora 那个包不支持 v2ProPlus）：

    D:\\GPT-SoVITS\\GPT-SoVITS-v2pro-20250604\\runtime\\python.exe -s ^
        tools\\tts_train.py --list <train.list> --wav-dir <切片目录> --exp voice01

踩过的坑（勿删）：
  * WebUI 是通过**环境变量**给 prepare_datasets 传参的，而且 1-get-text.py 还读
    `version`（决定音素前端），必须一并设置，否则 v2ProPlus 会走错前端。
  * 1a/1b/1c 都按 i_part/all_parts 分片输出，WebUI 之后会把分片合并成一个文件；
    这里 all_parts=1，仍然照它的命名与合并逻辑走，避免以后开多分片时行为不一致。
  * 这些子脚本要用 `-s` 启动，否则可能吃到用户 site-packages 里的同名包。
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

DEFAULT_GSV_ROOT = Path(r"D:\GPT-SoVITS\GPT-SoVITS-v2pro-20250604")
VERSION = "v2ProPlus"

# 各版本对应的预训练权重与输出目录（与整合包 config.py 保持一致）
PRETRAINED = {
    "s2G": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
    "s2D": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth",
    "s1": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
    "bert": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    "hubert": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    # v2Pro/v2ProPlus 独有：说话人向量模型，产出 7-sv_cn/
    "sv": "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
}
S2_CONFIG = "GPT_SoVITS/configs/s2v2ProPlus.json"
S1_CONFIG = "GPT_SoVITS/configs/s1longer-v2.yaml"
SOVITS_OUT = "SoVITS_weights_v2ProPlus"
GPT_OUT = "GPT_weights_v2ProPlus"


# --------------------------------------------------------------------------- #
def log(msg: str = "") -> None:
    print(msg, flush=True)


def repair_users_pth(root: Path) -> None:
    """修正整合包里 users.pth 的硬编码路径。

    整合包在打包时会把**打包者的绝对路径**写进
    `runtime/Lib/site-packages/users.pth`（v2pro 包的原始值是
    `D:\\下载\\GPT-SoVITS-v2pro-20250604`）。
    解压到别的目录后，这个 .pth 指向的目录不存在，于是
    `from text.cleaner import ...` / `from tools.my_utils import ...` 全部导入失败。
    PYTHONPATH 救不了：这个运行时带 `python39._pth`，会忽略 PYTHONPATH。

    所以必须按实际解压位置重写这个文件。
    """
    pth = root / "runtime" / "Lib" / "site-packages" / "users.pth"
    if not pth.exists():
        return
    subs = ("GPT_SoVITS/BigVGAN", "tools", "tools/asr", "GPT_SoVITS", "tools/uvr5")
    want = "\n".join([str(root)] + [f"{root}/{s}" for s in subs])
    try:
        cur = pth.read_text(encoding="utf-8", errors="replace")
    except Exception:
        cur = ""
    if cur.strip() != want.strip():
        try:
            pth.write_text(want + "\n", encoding="utf-8")
            log(f"  已修正 users.pth 的硬编码路径 -> {root}")
        except Exception as e:
            log(f"  !! users.pth 修正失败：{e}")


def setup_env(root: Path) -> None:
    if not (root / "runtime" / "python.exe").exists():
        sys.exit(f"找不到 v2pro 整合包的内嵌 Python：{root}\\runtime\\python.exe")
    if not (root / "GPT_SoVITS" / "configs" / "s2v2ProPlus.json").exists():
        sys.exit(
            f"{root} 不支持 v2ProPlus（缺 GPT_SoVITS/configs/s2v2ProPlus.json）。\n"
            "请改用 GPT-SoVITS-v2pro-* 整合包。"
        )
    repair_users_pth(root)
    os.chdir(root)
    os.environ["PATH"] = f"{root}{os.pathsep}{root / 'runtime'}{os.pathsep}" + os.environ.get("PATH", "")
    # prepare_datasets/*.py 里既有 `from text.cleaner import ...`（在 GPT_SoVITS/ 下）
    # 又有 `from tools.my_utils import ...`（在根目录下），
    # 而子进程的 sys.path[0] 只有脚本自己所在目录，所以两个路径都必须给到 PYTHONPATH。
    pp = [str(root), str(root / "GPT_SoVITS")]
    if os.environ.get("PYTHONPATH"):
        pp.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pp)
    os.environ["version"] = VERSION
    os.environ["hz"] = "25hz"
    os.environ["_CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def run(cmd: str, label: str, logfile: Path) -> int:
    log(f"  $ {cmd}")
    t0 = time.time()
    with logfile.open("w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.Popen(cmd, shell=True, stdout=fh, stderr=subprocess.STDOUT)
        rc = p.wait()
    log(f"    {label} 退出码={rc}  用时 {time.time() - t0:.0f}s  日志={logfile}")
    if rc != 0:
        log(f"    !! 失败，日志末尾：")
        try:
            for line in logfile.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]:
                log(f"       {line}")
        except Exception:
            pass
    return rc


# --------------------------------------------------------------------------- #
def merge_parts(opt_dir: Path, all_parts: int = 1) -> None:
    """合并分片产物。

    1a/1c 都是按 i_part 分片输出的（2-name2text-0.txt / 6-name2semantic-0.tsv），
    WebUI 之后会把分片拼成正式文件名再删掉分片。这里 all_parts=1 也同样照做，
    保持和 WebUI 完全一致的行为（以后要开多进程分片时不用改逻辑）。
    """
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


def exp_dir_of(cfg: dict) -> Path:
    """实验输出目录。

    必须和 WebUI 的 exp_root 一致：`<GSV_ROOT>/logs/<exp>`。
    （曾经写成 `work/<exp>/logs/<exp>`，结果 s2_train 存 checkpoint 时
     目录不存在，训练跑完一轮才失败。）
    """
    d = Path(cfg["root"]) / "logs" / cfg["exp"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_1abc(cfg: dict, work: Path) -> int:
    py = sys.executable
    opt_dir = exp_dir_of(cfg)
    opt_dir.mkdir(parents=True, exist_ok=True)
    env_common = {
        "inp_text": str(cfg["list"]),
        "inp_wav_dir": str(cfg["wav_dir"]),
        "exp_name": cfg["exp"],
        "opt_dir": str(opt_dir),
        "i_part": "0",
        "all_parts": "1",
        "is_half": "True",
        "version": VERSION,
    }

    # ---- 1A 文本 -> 音素 ----
    log("\n[1A] 文本 -> 音素")
    env = dict(env_common, bert_pretrained_dir=PRETRAINED["bert"])
    os.environ.update(env)
    if run(f'"{py}" -s GPT_SoVITS/prepare_datasets/1-get-text.py',
           "1A", work / "log_1a.txt") != 0:
        return 1

    # ---- 1B 音频 -> HuBERT ----
    log("\n[1B] 音频 -> HuBERT 特征")
    os.environ.update(dict(env_common, cnhubert_base_dir=PRETRAINED["hubert"]))
    if run(f'"{py}" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py',
           "1B", work / "log_1b.txt") != 0:
        return 1

    # ---- 1B-SV 说话人向量（v2Pro/v2ProPlus 特有）----
    log("\n[1B-SV] 说话人向量 (7-sv_cn)")
    os.environ.update(dict(
        env_common, cnhubert_base_dir=PRETRAINED["hubert"], sv_path=PRETRAINED["sv"]
    ))
    if run(f'"{py}" -s GPT_SoVITS/prepare_datasets/2-get-sv.py',
           "1B-SV", work / "log_1b_sv.txt") != 0:
        return 1

    # ---- 1C 语义 token ----
    log("\n[1C] 语义 token")
    os.environ.update(dict(
        inp_text=str(cfg["list"]),
        exp_name=cfg["exp"],
        opt_dir=str(opt_dir),
        pretrained_s2G=PRETRAINED["s2G"],
        s2config_path=S2_CONFIG,
        i_part="0",
        all_parts="1",
        is_half="True",
        version=VERSION,
    ))
    if run(f'"{py}" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py',
           "1C", work / "log_1c.txt") != 0:
        return 1

    # ---- 合并分片（WebUI 同款步骤）----
    log("\n[1ABC] 合并分片")
    merge_parts(opt_dir, all_parts=1)

    # ---- 校验产物 ----
    log("\n[1ABC] 产物校验")
    ok = True
    for name in ("2-name2text.txt", "3-bert", "4-cnhubert", "5-wav32k", "7-sv_cn", "6-name2semantic.tsv"):
        p = opt_dir / name
        exists = p.exists()
        size = ""
        if p.is_file():
            size = f"{p.stat().st_size / 1e6:.1f} MB"
        elif p.is_dir():
            n = len(list(p.glob("*")))
            size = f"{n} 个文件"
        log(f"    {'OK ' if exists else '!! '}{name:<24} {size}")
        ok = ok and exists
    if (opt_dir / "2-name2text.txt").exists():
        n = len((opt_dir / "2-name2text.txt").read_text(encoding="utf-8").strip().split("\n"))
        log(f"    音素条数 {n}")
    if (opt_dir / "6-name2semantic.tsv").exists():
        n = len((opt_dir / "6-name2semantic.tsv").read_text(encoding="utf-8").strip().split("\n")) - 1
        log(f"    语义条数 {n}")
    return 0 if ok else 1


def run_s2(cfg: dict, work: Path) -> int:
    log("\n[2A] SoVITS 训练")
    py = sys.executable
    s2_dir = exp_dir_of(cfg)
    (s2_dir / f"logs_s2_{VERSION}").mkdir(parents=True, exist_ok=True)
    data = json.loads((Path(S2_CONFIG)).read_text(encoding="utf-8"))
    if not cfg["half"]:
        data["train"]["fp16_run"] = False
    data["train"]["batch_size"] = cfg["s2_batch"]
    data["train"]["epochs"] = cfg["s2_epoch"]
    data["train"]["text_low_lr_rate"] = 0.4
    data["train"]["pretrained_s2G"] = PRETRAINED["s2G"]
    data["train"]["pretrained_s2D"] = PRETRAINED["s2D"]
    data["train"]["if_save_latest"] = True
    data["train"]["if_save_every_weights"] = True
    data["train"]["save_every_epoch"] = cfg["save_every"]
    data["train"]["gpu_numbers"] = "0"
    data["train"]["grad_ckpt"] = False
    data["train"]["lora_rank"] = 32
    data["model"]["version"] = VERSION
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = str(s2_dir)
    data["save_weight_dir"] = SOVITS_OUT
    data["name"] = cfg["exp"]
    data["version"] = VERSION
    tmp = work / "tmp_s2.json"
    tmp.write_text(json.dumps(data), encoding="utf-8")
    log(f"  batch={cfg['s2_batch']}  epochs={cfg['s2_epoch']}  fp16={cfg['half']}  权重输出={SOVITS_OUT}/")
    return run(f'"{py}" -s GPT_SoVITS/s2_train.py --config "{tmp}"',
               "2A", work / "log_s2.txt")


def run_s1(cfg: dict, work: Path) -> int:
    log("\n[2B] GPT 训练")
    import yaml

    py = sys.executable
    s1_dir = exp_dir_of(cfg)
    (s1_dir / "logs_s1").mkdir(parents=True, exist_ok=True)
    (s1_dir / f"logs_s1_{VERSION}").mkdir(parents=True, exist_ok=True)
    data = yaml.load(Path(S1_CONFIG).read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if not cfg["half"]:
        data["train"]["precision"] = "32"
    data["train"]["batch_size"] = cfg["s1_batch"]
    data["train"]["epochs"] = cfg["s1_epoch"]
    data["pretrained_s1"] = PRETRAINED["s1"]
    data["train"]["save_every_n_epoch"] = cfg["save_every"]
    data["train"]["if_save_every_weights"] = True
    data["train"]["if_save_latest"] = True
    data["train"]["if_dpo"] = False
    data["train"]["half_weights_save_dir"] = GPT_OUT
    data["train"]["exp_name"] = cfg["exp"]
    data["train_semantic_path"] = str(s1_dir / "6-name2semantic.tsv")
    data["train_phoneme_path"] = str(s1_dir / "2-name2text.txt")
    data["output_dir"] = str(s1_dir / f"logs_s1_{VERSION}")
    tmp = work / "tmp_s1.yaml"
    tmp.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")
    log(f"  batch={cfg['s1_batch']}  epochs={cfg['s1_epoch']}  权重输出={GPT_OUT}/")
    return run(f'"{py}" -s GPT_SoVITS/s1_train.py --config_file "{tmp}"',
               "2B", work / "log_s1.txt")


# --------------------------------------------------------------------------- #
def pick_work_dir(root: Path, exp: str) -> Path:
    """挑一个**确定能写**的训练工作目录。

    首选 `<GSV_ROOT>/work/train_<exp>`（和 WebUI 一致），但整合包目录
    在有些环境下是**不可写**的 —— 实测本机 Python 进程往里建目录会
    `PermissionError: [WinError 5] 拒绝访问`，而 PowerShell 却可以。
    所以这里逐个候选试建，第一个成功的就用。

    优先级：
      1. 环境变量 SCREENREADER_TRAIN_WORK（显式指定）
      2. <GSV_ROOT>/work/train_<exp>          —— 和 WebUI 一致
      3. <GSV_ROOT>/../work/train_<exp>       —— 整合包的兄弟目录，通常可写
      4. <系统临时目录>/screenreader_train/train_<exp>
    """
    name = f"train_{exp}"
    cands = []
    env = os.environ.get("SCREENREADER_TRAIN_WORK")
    if env:
        cands.append(Path(env) / name)
    cands.append(root / "work" / name)
    cands.append(root.parent / "work" / name)
    cands.append(Path(tempfile.gettempdir()) / "screenreader_train" / name)

    last = None
    for d in cands:
        try:
            d.mkdir(parents=True, exist_ok=True)
            # exist_ok 对"已存在但不可写"不报错，所以要真写一下验证
            probe = d / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            if d != cands[0]:
                print(f"  （整合包目录不可写，工作目录改用 {d}）", flush=True)
            return d
        except Exception as e:
            last = (d, e)
            print(f"  （{d} 不可写：{type(e).__name__}，试下一个）", flush=True)
    raise RuntimeError(
        f"找不到可写的训练工作目录。最后尝试 {last[0] if last else '?'}："
        f"{last[1] if last else '?'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="GPT-SoVITS v2ProPlus 微调一键训练")
    ap.add_argument("--list", required=True, help="训练标注 .list（vocal_path|speaker|lang|text）")
    ap.add_argument("--wav-dir", required=True, help="切片音频目录")
    ap.add_argument("--exp", required=True, help="实验名（英文/数字）")
    ap.add_argument("--gsv-root", default=str(DEFAULT_GSV_ROOT))
    ap.add_argument("--s2-batch", type=int, default=6)
    ap.add_argument("--s2-epoch", type=int, default=8)
    ap.add_argument("--s1-batch", type=int, default=8)
    ap.add_argument("--s1-epoch", type=int, default=15)
    ap.add_argument("--save-every", type=int, default=4)
    ap.add_argument("--fp32", action="store_true", help="关掉半精度（显存/精度异常时用）")
    ap.add_argument("--skip-1abc", action="store_true")
    ap.add_argument("--skip-s2", action="store_true")
    ap.add_argument("--skip-s1", action="store_true")
    args = ap.parse_args()

    root = Path(args.gsv_root)
    setup_env(root)

    lst = Path(args.list).resolve()
    wav_dir = Path(args.wav_dir).resolve()
    if not lst.exists():
        sys.exit(f"标注文件不存在：{lst}")
    if not wav_dir.exists():
        sys.exit(f"音频目录不存在：{wav_dir}")

    lines = [l for l in lst.read_text(encoding="utf-8").splitlines() if l.strip()]
    log("=" * 78)
    log(f"GPT-SoVITS 微调  version={VERSION}")
    log(f"  标注 {len(lines)} 条   {lst}")
    log(f"  音频目录           {wav_dir}")
    log(f"  实验名             {args.exp}")
    log("=" * 78)

    cfg = dict(
        list=lst, wav_dir=wav_dir, exp=args.exp, half=not args.fp32, root=root,
        s2_batch=args.s2_batch, s2_epoch=args.s2_epoch,
        s1_batch=args.s1_batch, s1_epoch=args.s1_epoch,
        save_every=args.save_every,
    )
    work = pick_work_dir(root, args.exp)
    log(f"工作目录 {work}")

    if not args.skip_1abc:
        if run_1abc(cfg, work) != 0:
            return 1
    else:
        log("\n[1ABC] 跳过")

    if not args.skip_s2:
        if run_s2(cfg, work) != 0:
            return 1
    else:
        log("\n[2A] SoVITS 训练  跳过")

    if not args.skip_s1:
        if run_s1(cfg, work) != 0:
            return 1
    else:
        log("\n[2B] GPT 训练  跳过")

    log("\n" + "=" * 78)
    log("产物")
    log("=" * 78)
    for d in (root / SOVITS_OUT, root / GPT_OUT):
        log(f"  {d.name}/")
        for f in sorted(d.glob("*")):
            log(f"    {f.name}  ({f.stat().st_size / 1e6:.0f} MB)")
    log(f"\n  SoVITS 训练日志 {work / 'log_s2.txt'}")
    log(f"  GPT    训练日志 {work / 'log_s1.txt'}")
    log("\n下一步：用 SoVITS_weights_v2ProPlus 里的 .pth 与 GPT_weights_v2ProPlus 里的 .ckpt 做推理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
