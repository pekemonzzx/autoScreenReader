"""声库档案：把"一个音色"所需的一切打包成一个可插拔的目录。

动机（用户需求）：训练语料会换、音色会换（自己重训 / 用别人的模型），
所以**训练、训练配置、推理配置都要能独立替换**，不能写死在代码或全局 config 里。

目录结构：

    voices/
      voice01/
        voice.json      元数据 + 推理配置（权重路径 / 参考音频 / prompt_text）
        train.yaml      训练配置（版本、预训练模型、batch/epoch…）
        ref.wav         参考音频（推理时用来克隆音色）
        train.list      训练标注（wav路径|说话人|语言|文本）
        README.md       可选：这个声库的来源/注意事项

切换声库 = 改 `data/config.json` 里的 `tts.voice`，或者在界面上选。
训练另一个声库 = `python -m screenreader voice new <名字>` 生成骨架，填好语料后 `voice train`。

为什么 `prompt_text` 要跟 `ref.wav` 放在一起：这两者**必须对得上**，
否则每句话开头会多出听不懂的音（实测踩过，见《读屏台使用说明》教训 5）。
把它们放在同一个档案里、并提供 `voice verify` 自检，就能避免配串。
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_GSV_ROOT = r"D:\GPT-SoVITS\GPT-SoVITS-v2pro-20250604"
# 只有 v2 / v2ProPlus 有 ONNX 推理路径（P3 要用），所以默认锁 v2ProPlus
SUPPORTED_VERSIONS = ("v1", "v2", "v2Pro", "v2ProPlus")

# 声库包的格式版本。以后改结构时递增，安装时据此判断兼容性。
PACK_VERSION = 1
PACK_SUFFIX = ".voicepack.zip"
# 权重在包内的固定位置（相对档案目录）
WEIGHTS_SUBDIR = "weights"


# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """训练参数。放在档案里，换声库时可以各调各的。"""

    version: str = "v2ProPlus"
    gsv_root: str = DEFAULT_GSV_ROOT
    # 语料
    list_path: str = ""          # 训练标注 .list
    wav_dir: str = ""            # 切片音频目录
    # 数据处理
    clean_text: bool = True
    # SoVITS
    s2_batch: int = 6
    s2_epoch: int = 8
    # GPT
    s1_batch: int = 8
    s1_epoch: int = 15
    save_every: int = 4
    fp16: bool = True

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TrainConfig":
        cfg = cls()
        valid = {f.name for f in fields(cls)}
        for k, v in (raw or {}).items():
            if k in valid:
                setattr(cfg, k, v)
            else:
                log.warning("训练配置项 %s 不存在，已忽略", k)
        return cfg


@dataclass
class VoiceProfile:
    """一个音色的全部信息。"""

    name: str
    display_name: str = ""
    version: str = "v2ProPlus"
    # --- 推理 ---
    ref_audio: str = ""          # 参考音频绝对路径（或相对档案目录）
    prompt_text: str = ""        # 必须与 ref_audio 内容一致，否则输出开头会有杂音
    gpt_weights: str = ""        # 如 GPT_weights_v2ProPlus/voice01-e12.ckpt
    sovits_weights: str = ""     # 如 SoVITS_weights_v2ProPlus/voice01_e8_s248.pth
    api_url: str = "http://127.0.0.1:9880"
    speed_factor: float = 1.0
    # --- 训练 ---
    train: TrainConfig = field(default_factory=TrainConfig)
    # --- 元数据 ---
    created: str = ""
    source: str = ""             # 语料来源说明
    notes: str = ""

    # ---------------------------------------------------------------- #
    @property
    def display(self) -> str:
        return self.display_name or self.name

    def resolve(self, path: str, base: Path) -> str:
        """相对路径按档案目录解析；绝对路径原样返回。"""
        if not path:
            return ""
        p = Path(path)
        return str(p if p.is_absolute() else (base / p).resolve())

    # ---------------------------------------------------------------- #
    @classmethod
    def load(cls, directory: Path) -> "VoiceProfile":
        d = Path(directory)
        meta = d / "voice.json"
        if not meta.exists():
            raise FileNotFoundError(f"声库档案缺少 voice.json：{d}")
        raw = json.loads(meta.read_text(encoding="utf-8"))
        name = raw.pop("name", d.name)
        train_raw = raw.pop("train", {})
        prof = cls(name=name, **{k: v for k, v in raw.items() if k in {f.name for f in fields(cls)}})
        prof.train = TrainConfig.from_dict(train_raw)
        return prof

    def save(self, directory: Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["train"] = asdict(self.train)
        (d / "voice.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---------------------------------------------------------------- #
    def to_tts_kwargs(self, directory: Path) -> Dict[str, Any]:
        """转成 TtsConfig 需要的字段。"""
        d = Path(directory)
        return {
            "ref_audio": self.resolve(self.ref_audio, d),
            "prompt_text": self.prompt_text,
            "speed_factor": self.speed_factor,
        }

    # ---------------------------------------------------------------- #
    def weight_paths(self, directory: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """解析两个权重文件的绝对路径。

        查找顺序（让声库包能**自带权重**，不依赖整合包目录布局）：
          1. 绝对路径 -> 直接用
          2. 档案目录内存在该相对路径 -> 用档案内的（声库包的情形）
          3. 否则当作相对整合包根目录（早期手填路径的情形）
        """
        d = Path(directory)
        out: List[Optional[Path]] = []
        for rel in (self.gpt_weights, self.sovits_weights):
            if not rel:
                out.append(None)
                continue
            p = Path(rel)
            if p.is_absolute():
                out.append(p)
                continue
            inside = d / rel
            if inside.exists():
                out.append(inside)
                continue

            # 4. **训练输出目录的 weights/ 下**。
            #    整合包目录不可写时，trainer 会把权重落到可写目录里
            #    （<GSV 同级>/_sr_train/work/weights 或项目内的 _tmp 下），
            #    档案里记的仍是相对路径 —— 不在这里找一遍就会误报
            #    "权重不存在"，让人以为训练白跑了。
            found = None
            # 注意 parents 的下标：档案目录是 <项目>/voices/<名字>，
            # 所以 parents[1] 才是项目根（parents[2] 会跑到上一层去）。
            bases = [
                Path(self.train.gsv_root).parent / "_sr_train",
                Path(self.train.gsv_root).parent,
            ]
            dd = Path(directory).resolve()
            for up in (0, 1, 2, 3):
                try:
                    bases.append(dd.parents[up] / "data" / "_tmp" / "screenreader_train")
                except IndexError:
                    pass
            for base in bases:
                cand = base / "work" / "weights" / rel
                if cand.exists():
                    found = cand
                    break
            out.append(found if found else Path(self.train.gsv_root) / rel)
        return out[0], out[1]

    def is_self_contained(self, directory: Path) -> bool:
        """权重和参考音频是否都在档案目录里（能独立打包分发）。"""
        d = Path(directory)
        g, s = self.weight_paths(d)
        ref = Path(self.resolve(self.ref_audio, d)) if self.ref_audio else None
        ok = bool(g and s and ref)
        for p in (g, s, ref):
            if p is None or not p.exists():
                ok = False
            elif d not in p.resolve().parents and p.resolve() != d.resolve():
                ok = False   # 指向档案目录之外 -> 不是自包含的
        return ok

    def train_paths(self, directory: Path) -> tuple:
        """把训练用的两个路径解析成绝对路径。

        标注文件通常是档案内的相对路径（`train.list`），
        切片目录通常是外部绝对路径——两种都要能处理。
        """
        d = Path(directory)
        lst = Path(self.train.list_path) if self.train.list_path else None
        if lst is not None and not lst.is_absolute():
            lst = (d / lst).resolve()
        wav = Path(self.train.wav_dir) if self.train.wav_dir else None
        if wav is not None and not wav.is_absolute():
            wav = (d / wav).resolve()
        return lst, wav

    def api_launch_command(self, directory: Path, runtime_root: Optional[Path] = None) -> str:
        """生成启动推理 API 的命令行（可直接粘贴到 PowerShell）。

        `runtime_root` 是推理运行时目录（按需下载来的）；不传就用档案里的
        `train.gsv_root`（开发机训练与推理共用整合包的情形）。
        """
        d = Path(directory)
        root = Path(runtime_root) if runtime_root else Path(self.train.gsv_root)
        cfg = d / "tts_infer.yaml"
        return (
            f'& "{root / "runtime" / "python.exe"}" -s api_v2.py '
            f'-a 127.0.0.1 -p {self.api_url.rsplit(":", 1)[-1]} -c "{cfg}"'
        )

    def write_api_config(self, directory: Path, runtime_root: Optional[Path] = None) -> Path:
        """生成 api_v2.py 用的 tts_infer.yaml（指向这个声库的权重）。

        `bert_base_path` / `cnhuhbert_base_path` 指向**推理运行时**里的预训练模型；
        两个权重走 `weight_paths()`——优先用档案自带的，退回整合包目录。
        """
        d = Path(directory)
        root = Path(runtime_root) if runtime_root else Path(self.train.gsv_root)
        out = d / "tts_infer.yaml"
        gpt, sovits = self.weight_paths(d)
        body = (
            f"# 由 screenreader voice 生成，指向声库「{self.display}」的 {self.version} 权重\n"
            f"# 重新生成：python -m screenreader voice refresh {self.name}\n"
            f"# 推理运行时：{root}\n"
            "custom:\n"
            f"  bert_base_path: {root / 'GPT_SoVITS' / 'pretrained_models' / 'chinese-roberta-wwm-ext-large'}\n"
            f"  cnhuhbert_base_path: {root / 'GPT_SoVITS' / 'pretrained_models' / 'chinese-hubert-base'}\n"
            "  device: cuda\n"
            "  is_half: true\n"
            f"  t2s_weights_path: {gpt or ''}\n"
            f"  version: {self.version}\n"
            f"  vits_weights_path: {sovits or ''}\n"
        )
        out.write_text(body, encoding="utf-8")
        return out


# --------------------------------------------------------------------------- #
class VoiceStore:
    """管理 voices/ 目录下的一堆声库，以及"当前用哪个"。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def dir_of(self, name: str) -> Path:
        return self.root / name

    def exists(self, name: str) -> bool:
        return (self.dir_of(name) / "voice.json").exists()

    def list_names(self) -> List[str]:
        return sorted(
            p.parent.name for p in self.root.glob("*/voice.json")
        )

    def load(self, name: str) -> VoiceProfile:
        if not self.exists(name):
            raise FileNotFoundError(f"没有这个声库：{name}（在 {self.root} 下）")
        return VoiceProfile.load(self.dir_of(name))

    def save(self, prof: VoiceProfile) -> Path:
        d = self.dir_of(prof.name)
        prof.save(d)
        return d

    # ---------------------------------------------------------------- #
    def create(
        self,
        name: str,
        *,
        display_name: str = "",
        version: str = "v2ProPlus",
        list_path: str = "",
        wav_dir: str = "",
        gsv_root: str = DEFAULT_GSV_ROOT,
        source: str = "",
    ) -> VoiceProfile:
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"版本 {version!r} 不受支持。可用：{SUPPORTED_VERSIONS}。\n"
                "注意：只有 v2 / v2ProPlus 有 ONNX 推理路径（P3 打包要用）。"
            )
        if self.exists(name):
            raise FileExistsError(f"声库已存在：{name}")
        prof = VoiceProfile(
            name=name,
            display_name=display_name or name,
            version=version,
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
            source=source,
            train=TrainConfig(version=version, gsv_root=gsv_root, list_path=list_path, wav_dir=wav_dir),
        )
        self.save(prof)
        (self.dir_of(name) / "README.md").write_text(
            f"# 声库 {prof.display}\n\n"
            f"- 版本：{version}\n"
            f"- 创建：{prof.created}\n"
            f"- 语料来源：{source or '（待填）'}\n\n"
            "## 目录里的东西\n\n"
            "- `voice.json` 元数据 + 推理配置\n"
            "- `train.yaml` 训练配置（换语料时改这里）\n"
            "- `ref.wav` 参考音频（推理时克隆音色用）\n"
            "- `train.list` 训练标注\n"
            "- `tts_infer.yaml` 自动生成，给 api_v2.py 用\n\n"
            "## 注意\n\n"
            "`prompt_text` 必须与 `ref.wav` 的实际内容一致，否则每句开头会多出听不懂的音。\n"
            "生成/修改后跑 `python -m screenreader voice verify " + name + "` 自检。\n",
            encoding="utf-8",
        )
        return prof

    def delete(self, name: str) -> bool:
        d = self.dir_of(name)
        if not d.exists():
            return False
        shutil.rmtree(d)
        return True

    def install_any(self, src, overwrite: bool = False, say=print) -> List[str]:
        """什么形状的声库来源都能装：zip、单个声库文件夹、装着多个声库的文件夹。

        用户拿到手的可能是一堆不同形态的东西：
          * `voice01.voicepack.zip`
          * 已经解压好的 `voice01/` 文件夹（里面是 voice.json + weights/）
          * 一个文件夹里塞了好几个声库（或好几个 zip）
        这三种都要能直接吃下去，不然又得教用户"先解压/先找对目录"。
        """
        src = Path(src)
        if not src.exists():
            raise FileNotFoundError(f"路径不存在：{src}")

        # 1. 单个 zip
        if src.is_file():
            if src.suffix.lower() != ".zip":
                raise ValueError(f"不认识的声库文件类型：{src.name}（需要 .zip）")
            return [self.install(src, overwrite=overwrite, say=say).name]

        # 2. 目录本身就是个声库
        if (src / "voice.json").exists():
            name = self._name_of_dir(src)
            dest = self.dir_of(name)
            if dest.exists():
                if not overwrite:
                    raise FileExistsError(f"声库「{name}」已存在（加 --overwrite 覆盖）")
                shutil.rmtree(dest)
            shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            prof = self.load(name)
            try:
                prof.write_api_config(dest)
            except Exception as e:
                say(f"  !! 生成 tts_infer.yaml 失败：{e}")
            say(f"  已安装声库「{prof.display}」-> {dest}")
            return [name]

        # 3. 一个装了很多东西的目录：逐个处理里面的 zip 和声库文件夹
        found: List[str] = []
        for item in sorted(src.iterdir()):
            try:
                if item.is_file() and item.suffix.lower() == ".zip":
                    found += self.install_any(item, overwrite, say)
                elif item.is_dir() and (item / "voice.json").exists():
                    found += self.install_any(item, overwrite, say)
            except Exception as e:
                say(f"  !! {item.name} 安装失败：{e}")
        if not found:
            raise ValueError(
                f"{src} 里没找到声库：\n"
                "  · 要么选 .voicepack.zip\n"
                "  · 要么选里面带 voice.json 的声库文件夹（或其上层目录）"
            )
        return found

    @staticmethod
    def _name_of_dir(d: Path) -> str:
        try:
            raw = json.loads((d / "voice.json").read_text(encoding="utf-8"))
            return raw.get("name") or d.name
        except Exception:
            return d.name

    # ---------------------------------------------------------------- #
    # 声库包：把"模型 + 声库名 + 参考音频"打成一个可独立分发的压缩包
    # ---------------------------------------------------------------- #
    def pack(self, name: str, out_dir: Path, include_train: bool = False,
             say=print) -> Path:
        """打包成 `<名字>.voicepack.zip`。

        包里自带权重和参考音频，所以拿到包的人**不需要**有整合包里的那些目录。
        训练相关的东西（train.list / 切片目录）默认不打进去——那是训练用的，
        分发时又大又没用。
        """
        prof = self.load(name)
        d = self.dir_of(name)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        gpt, sovits = prof.weight_paths(d)
        ref = Path(prof.resolve(prof.ref_audio, d)) if prof.ref_audio else None
        for label, p in (("GPT 权重", gpt), ("SoVITS 权重", sovits), ("参考音频", ref)):
            if p is None or not p.exists():
                raise FileNotFoundError(f"{label} 不存在，无法打包：{p}")

        # 把权重收进档案内的 weights/，让包自包含
        wdir = d / WEIGHTS_SUBDIR
        wdir.mkdir(parents=True, exist_ok=True)
        new_gpt = new_sovits = ""
        for label, src, attr in (("gpt", gpt, "gpt_weights"), ("sovits", sovits, "sovits_weights")):
            dst = wdir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
                say(f"  收入权重 {src.name}  ({dst.stat().st_size / 1e6:.0f} MB)")
            setattr(prof, attr, f"{WEIGHTS_SUBDIR}/{src.name}")
            if attr == "gpt_weights":
                new_gpt = dst.name
            else:
                new_sovits = dst.name

        # 参考音频统一叫 ref.wav，放进档案根
        ref_dst = d / "ref.wav"
        if ref.resolve() != ref_dst.resolve():
            shutil.copy2(ref, ref_dst)
        prof.ref_audio = "ref.wav"
        self.save(prof)   # 档案现在是自包含的

        manifest = {
            "pack_version": PACK_VERSION,
            "name": prof.name,
            "display_name": prof.display,
            "version": prof.version,
            "source": prof.source,
            "notes": prof.notes,
            "created": prof.created,
            "gpt_weights": new_gpt,
            "sovits_weights": new_sovits,
            "ref_audio": "ref.wav",
            "prompt_text": prof.prompt_text,
        }

        out = out_dir / f"{prof.name}{PACK_SUFFIX}"
        skip = {"tts_infer.yaml", "__pycache__"}
        if not include_train:
            skip |= {"train.list", "train.yaml"}
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for f in sorted(d.rglob("*")):
                if f.is_dir() or f.name in skip or f.suffix == ".pyc":
                    continue
                z.write(f, f"voice/{f.relative_to(d)}")
        return out

    def install(self, pack_path: Path, overwrite: bool = False, say=print) -> VoiceProfile:
        """从声库包安装。解压到 voices/<名字>/ 并做基本校验。"""
        pack_path = Path(pack_path)
        if not pack_path.exists():
            raise FileNotFoundError(f"声库包不存在：{pack_path}")
        with zipfile.ZipFile(pack_path) as z:
            names = z.namelist()
            if "manifest.json" not in names:
                raise ValueError("不是有效的声库包：缺少 manifest.json")
            man = json.loads(z.read("manifest.json").decode("utf-8"))
            if int(man.get("pack_version", 0)) != PACK_VERSION:
                raise ValueError(
                    f"声库包格式版本 {man.get('pack_version')} 与当前程序（{PACK_VERSION}）不匹配"
                )
            name = man["name"]
            dest = self.dir_of(name)
            if dest.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"声库「{name}」已存在。加 --overwrite 覆盖，或先 voice delete {name}"
                    )
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            for n in names:
                if n == "manifest.json" or n.endswith("/"):
                    continue
                if not n.startswith("voice/"):
                    continue      # 防目录穿越：只接受 voice/ 下的条目
                rel = n[len("voice/"):]
                if ".." in Path(rel).parts:
                    continue
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(n) as src, open(target, "wb") as fh:
                    shutil.copyfileobj(src, fh)

        if not (dest / "voice.json").exists():
            shutil.rmtree(dest, ignore_errors=True)
            raise ValueError(f"声库包不完整：缺少 voice/voice.json（包名 {name}）")
        try:
            prof = self.load(name)
        except Exception as e:
            shutil.rmtree(dest, ignore_errors=True)
            raise ValueError(f"声库包解压后无法读取：{e}")
        # 权重和参考音频都在档案里了，把解析逻辑走一遍确认
        g, s = prof.weight_paths(dest)
        miss = [str(p) for p in (g, s) if p is None or not p.exists()]
        if miss:
            say(f"  !! 安装后缺少权重：{miss}")
        # **必须**生成 tts_infer.yaml：声库包不带它，缺了 TTS 服务就起不来，
        # 表现是"装好了声库但每句合成都失败"。先用档案里的 gsv_root 兜底，
        # 程序启动时 ensure_voice_configs() 会把它改成真正的运行时位置。
        try:
            prof.write_api_config(dest)
            say("  已生成 tts_infer.yaml")
        except Exception as e:
            say(f"  !! 生成 tts_infer.yaml 失败：{e}")
        say(f"  已安装声库「{prof.display}」-> {dest}")
        return prof

    # ---------------------------------------------------------------- #
    def write_train_yaml(self, prof: VoiceProfile) -> Path:
        import yaml

        d = self.dir_of(prof.name)
        out = d / "train.yaml"
        out.write_text(
            yaml.safe_dump(asdict(prof.train), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return out

    def read_train_yaml(self, name: str) -> Dict[str, Any]:
        import yaml

        p = self.dir_of(name) / "train.yaml"
        if not p.exists():
            return {}
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def profiles_root_for(project_root: Path) -> Path:
    return Path(project_root) / "voices"
