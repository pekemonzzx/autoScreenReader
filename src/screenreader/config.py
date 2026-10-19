"""集中配置。

设计要点：
  * 全部字段都有合理默认值，配置文件缺失也能跑起来。
  * 坐标一律是**相对显示器左上角的物理像素**（capture.py 已做 DPI 感知）。
  * TTS 后端可替换：现在用 GPT-SoVITS 的 HTTP API，P3 之后可以换成进程内 ONNX。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

Region = List[int]  # [left, top, width, height]


# --------------------------------------------------------------------------- #
@dataclass
class CaptureConfig:
    backend: str = "auto"           # auto | dxgi | mss
    monitor_index: int = 0
    region: Optional[Region] = None  # None = 整个显示器


@dataclass
class OcrConfig:
    det_limit_side_len: int = 960
    use_cls: bool = True
    intra_threads: int = 0           # 0 = 用标定/启发值
    autotune: bool = True            # 首次启动标定线程数（结果缓存）
    # 每轮最多读多少行。OCR 耗时与行数近似线性，区域选大了必须靠它兜底。
    max_lines: int = 25
    min_score: float = 0.5
    # **画面没变时跳过 OCR**（省 CPU 最有效的一项，说明见 EngineConfig）。
    # 放进配置是为了跨会话记住用户的选择。
    skip_unchanged_frame: bool = True


@dataclass
class SegmentConfig:
    # 切句标点（命中即切）
    split_chars: str = "。！？!?；;…"
    # 次要切分点，只在句子过长时用
    soft_split_chars: str = "，,、"
    min_chars: int = 4               # 短于这个的碎片并入相邻句
    max_chars: int = 28              # 长于这个继续切
    merge_gap_chars: int = 0


@dataclass
class TtsConfig:
    backend: str = "gsv_api"         # gsv_api | onnx（P3 再加）
    api_url: str = "http://127.0.0.1:9880"
    # 声库名（voices/<名字>/）。设了它就会用声库档案里的
    # 参考音频 / prompt_text / 权重，config 里这两项将被忽略。
    voice: str = ""
    # TTS 推理运行时的位置（首次启动按需下载的那个）。
    # 空 = 用声库档案里的 train.gsv_root（开发机就是这样，训练和推理共用整合包）。
    # 目标机上由 `screenreader runtime install` 自动写入。
    runtime_root: str = ""
    ref_audio: str = ""
    prompt_text: str = ""
    prompt_lang: str = "zh"
    text_lang: str = "zh"
    speed_factor: float = 1.0
    streaming: bool = True           # 实测流式首包 667ms vs 非流式 1786ms
    split_method: str = "cut5"
    sample_steps: int = 32
    timeout_s: float = 60.0


@dataclass
class AudioConfig:
    # 输出设备名（子串匹配，大小写不敏感）。直播场景填 "CABLE Input"。
    device: str = ""
    samplerate: int = 0              # 0 = 跟随音频自身采样率
    blocksize: int = 0
    latency: str = "low"             # low | high
    # 播放音量百分比。**100 = 原始输出**，可以调到 300（放大 3 倍）。
    # 用户反馈默认音量偏小，所以允许往上放大；
    # 超过 100 时会走软限幅，避免硬削波把声音弄破。
    volume: int = 100


@dataclass
class SnapConfig:
    """「截图朗读」的设置。

    场景：连续但**位置不一样**的段落 —— 用户随手框一段就让它读一遍，
    读完这段临时区域自动消失，常驻区域恢复原状。
    """

    # 全局快捷键。支持键盘组合键和鼠标侧键：
    #   "ctrl+alt+s" / "ctrl+shift+f9" / "mouse4" / "mouse5"
    # 走 Windows 低级钩子，游戏在前台时也生效。
    hotkey: str = "ctrl+alt+s"
    # 临时区域用哪种模式识别。**默认整段**：用户要的是"读一整个段落"，
    # 整段模式会合并折行、按标点断句，听起来才连贯。
    mode: str = "paragraph"
    # 读完自动删除临时区域（用户要求的默认行为）
    auto_remove: bool = True


@dataclass
class CacheConfig:
    enabled: bool = True
    dir: str = ""                    # 空 = <程序目录>/data/tts_cache
    max_entries: int = 500
    max_mb: int = 512
    # **条目多久没被用就自动删（秒）；0 = 不自动过期。默认 30 秒。**
    #
    # 用户要求"语音生成 30 秒后自动清理"。读屏生成的音频绝大多数只播一次，
    # 留着只占磁盘。清的是磁盘缓存，卡片内存里的音频不受影响。
    ttl_s: float = 30.0
    # 程序关闭时清空全部缓存（用户要求）
    clear_on_exit: bool = True


@dataclass
class PipelineConfig:
    # 同一段文字重复出现时不再重复朗读（读屏最大的浪费就在这里）
    skip_unchanged: bool = True
    # 两轮读屏之间的最小间隔（秒），防止游戏满载时疯狂抢占
    min_interval_s: float = 0.5
    # 合成队列深度：1 = 严格串行（最不影响游戏），2-3 = 吞吐更好
    synth_workers: int = 1
    # 播放前预缓冲多少块（流式模式下用小值换低延迟）
    prebuffer_chunks: int = 2
    # 新内容到来时是否打断当前朗读。
    # 实测教训：设 True 会让"每轮都打断上一轮"，结果永远只读出第一句
    # （44 次读屏只合成出 2 句）。读屏场景应该**排队**而不是打断。
    interrupt_on_new: bool = False
    # 排队上限：积压超过这个数，**优先丢弃已经滚出屏幕的**（见 pipeline._trim_backlog）。
    # 注意它必须明显大于"一轮能切出的句数"，否则一轮之内就会被自己截断。
    max_backlog: int = 8
    # 只朗读相对上一轮**新增**的行（弹幕滚动场景的关键优化）
    dedupe_lines: bool = True
    # **自动朗读**：每张卡片生成就自动顺序播一遍（各只播一次）。
    # 放在 pipeline 里而不是 engine 里，是为了跨会话记住用户的选择。
    auto_read_all: bool = False
    # 行相似度阈值：OCR 对同一行两次识别的结果可能有细微差异，不能直接字符串相等
    similarity: float = 0.85
    # 变化行占比超过这个值就当作整屏刷新，全部朗读（切场景时避免只念零散几行）
    refresh_ratio: float = 0.6


@dataclass
class AppConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    snap: SnapConfig = field(default_factory=SnapConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    log_level: str = "INFO"
    # 框选出来的识别区域。以前完全没存，重启就得重新框一遍 ——
    # 现在整段/单行是**逐区域**配置的，不存下来等于每次都要重设。
    # 存成原始 dict（含 mode），由 board.Region 负责解释。
    regions: List[Dict[str, Any]] = field(default_factory=list)
    # 悬浮窗不透明度（百分比 20~100）。用户要求可调。
    # 放在顶层而不是嵌套，是因为它只有一个值，且老配置里没有它时
    # 缺失会回落到 100（全不透明），行为与以前一致。
    overlay_opacity: int = 100

    # ---------------------------------------------------------------- #
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        """读配置。**path 为 None 时会去读默认路径**（data/config.json）。

        注意这跟"返回默认值"是两回事——曾经 load(None) 直接返回全默认配置，
        于是 ref_audio / prompt_text 是空串，TTS 服务一律返回 400，
        排查了半天。想要纯默认值请用 AppConfig()。
        """
        p = Path(path) if path else default_config_path()
        if not p.exists():
            log.info("配置文件不存在，使用默认值：%s", p)
            return cls()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("配置文件解析失败，使用默认值（%s）：%s", p, e)
            return cls()
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AppConfig":
        cfg = cls()
        for f in fields(cls):
            sub = raw.get(f.name)
            if sub is None:
                continue
            if isinstance(sub, dict) and hasattr(getattr(cfg, f.name), "__dataclass_fields__"):
                section = getattr(cfg, f.name)
                valid = {x.name for x in fields(section)}
                for k, v in sub.items():
                    if k in valid:
                        setattr(section, k, v)
                    else:
                        log.warning("配置项 %s.%s 不存在，已忽略", f.name, k)
            else:
                setattr(cfg, f.name, sub)
        return cfg

    def save(self, path: Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("配置已写入 %s", p)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ---------------------------------------------------------------- #
    # 识别区域的存取
    # ---------------------------------------------------------------- #
    def save_regions(self, regions, path: Optional[Path] = None) -> None:
        """把当前区域（含单行/整段模式）写回配置文件。

        **只改 regions 这一个键，不做整份覆盖**。
        以前这里直接 `self.save()` 把内存里的整份配置写回去，结果：
        谁要是拿一个默认 `AppConfig()` 调它（比如测试），
        就会把真的 `data/config.json` 里的声库、参考音频全冲成空 —— 实际踩过一次。
        改成"读出来 -> 改 regions -> 写回去"就不可能误伤别的字段。
        """
        out = []
        for r in regions:
            # **临时区域（「截图朗读」用）不落盘** —— 它读完就删，
            # 存进配置只会在每次用完之后留一条垃圾。
            if getattr(r, "temp", False):
                continue
            out.append({
                "id": r.id,
                "name": r.name,
                "rect": list(r.rect),
                "monitor_index": r.monitor_index,
                "enabled": r.enabled,
                "color": r.color,
                "mode": r.mode,
            })
        self.regions = out
        p = Path(path) if path else default_config_path()
        try:
            raw: Dict[str, Any] = {}
            if p.exists():
                try:
                    loaded = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        raw = loaded
                except Exception as e:
                    log.warning("原有配置解析失败，只写区域（%s）：%s", p, e)
            raw["regions"] = out
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning("保存识别区域失败：%s", e)

    def load_regions(self) -> List[Dict[str, Any]]:
        """读回区域列表（过滤掉字段不全的脏数据）。"""
        out = []
        for raw in (self.regions or []):
            if not isinstance(raw, dict):
                continue
            rect = raw.get("rect")
            if not (isinstance(rect, (list, tuple)) and len(rect) == 4):
                continue
            item = {
                "id": str(raw.get("id") or ""),
                "name": str(raw.get("name") or "区域"),
                "rect": tuple(int(v) for v in rect),
                "monitor_index": int(raw.get("monitor_index") or 0),
                "enabled": bool(raw.get("enabled", True)),
                "color": str(raw.get("color") or "#4ea1ff"),
                # 老配置没这个字段 -> 默认单行，保持原行为
                "mode": str(raw.get("mode") or "line"),
            }
            if item["mode"] not in ("line", "paragraph"):
                item["mode"] = "line"
            out.append(item)
        return out

    # ---------------------------------------------------------------- #
    def apply_log_level(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def cache_dir(self) -> Path:
        if self.cache.dir:
            return Path(self.cache.dir)
        return _app_data_base() / "data" / "tts_cache"

    def voices_roots(self) -> List[Path]:
        """声库可能在的目录，按优先级排列。

        默认在 `<程序目录>/voices`。但**如果运行时被装到了别处**（用户在安装
        向导里换了盘，比如把 9GB 的运行时放到 D 盘），声库就该跟着运行时走：

            用户选 D:\\X\\gsv  ->  声库期望在 D:\\X\\voices

        以前这里写死 `<程序目录>/voices`，导致用户选了别的安装位置之后，
        声库悄悄装回程序目录 —— 用户看到的现象就是
        「声库压缩包没有正确解压到安装目录/voices/ 下」。
        """
        base = _app_data_base()
        out: List[Path] = [base / "voices"]

        rt = (self.tts.runtime_root or "").strip()
        if rt:
            p = Path(rt)
            # 运行时根一般叫 gsv，声库放在它旁边
            sib = (p.parent if p.name.lower() == "gsv" else p) / "voices"
            try:
                same = sib.resolve() == (base / "voices").resolve()
            except Exception:
                same = str(sib).lower() == str(base / "voices").lower()
            if not same:
                out.append(sib)
        return out

    def voices_dir_for_runtime(self, runtime_root) -> Path:
        """给定一个运行时位置，声库应该装到哪。

        规则和 `voices_roots()` 一致：运行时在程序目录里 -> 装到
        `<程序目录>/voices`；运行时在别处（用户换盘了）-> 装到它旁边的
        `voices`，这样用户翻自己选的安装目录就能看到。
        """
        base = _app_data_base()
        if not runtime_root:
            return base / "voices"
        p = Path(runtime_root)
        sib = (p.parent if p.name.lower() == "gsv" else p) / "voices"
        try:
            inside = base.resolve() == p.resolve() or base.resolve() in p.resolve().parents
        except Exception:
            inside = str(p).lower().startswith(str(base).lower())
        return (base / "voices") if inside else sib

    def voices_root(self) -> Path:
        """声库档案目录：默认 `<程序目录>/voices/`。

        如果配置里的运行时在程序目录之外，且那边已经有声库，就用那边的 ——
        这样"运行时装在 D 盘，声库也在 D 盘"才成立。
        """
        cands = self.voices_roots()
        for p in cands[1:]:
            try:
                if p.is_dir() and any(p.glob("*/voice.json")):
                    return p
            except Exception:
                continue
        return cands[0]

    # ---------------------------------------------------------------- #
    def apply_voice(self, voices_root: Optional[Path] = None,
                    auto_select: bool = True) -> Optional[str]:
        """如果配置里指定了声库（tts.voice），就用档案里的推理参数覆盖。

        这样"换声库"只要改一个名字；参考音频和 prompt_text 由档案给出，
        不会出现两者配串（那会让每句话开头多出听不懂的音）。

        **auto_select=True 时，没指定声库会自动选第一个可用的。**
        这个很关键：目标机上用户把声库包解压进 voices\\ 了，
        但 config.json 里的 tts.voice 还是空的 —— 界面下拉框**显示**着某个音色，
        实际 ref_audio 是空的，于是每句合成都报"没有配置参考音频"。
        （实测日志里 200 多条同一错误就是这个。）
        """
        root = Path(voices_root) if voices_root else self.voices_root()
        name = (self.tts.voice or "").strip()

        if name and not (root / name / "voice.json").exists():
            log.warning("配置里指定的声库 %r 不存在，将自动另选", name)
            name = ""

        if not name and auto_select:
            try:
                from .voice import VoiceStore

                names = VoiceStore(root).list_names()
                if names:
                    name = names[0]
                    log.info("配置里没有指定声库，自动选用 %r", name)
            except Exception as e:
                log.debug("扫描声库失败：%s", e)

        self.tts.voice = name
        if not name:
            return None

        try:
            from .voice import VoiceProfile

            prof = VoiceProfile.load(root / name)
        except Exception as e:
            log.error("声库 %r 加载失败，继续用 config.json 里的设置：%s", name, e)
            return None

        kw = prof.to_tts_kwargs(root / name)
        if self.tts.ref_audio and Path(self.tts.ref_audio).resolve() != Path(kw["ref_audio"]).resolve():
            log.info("已按声库 %r 覆盖 config.json 里的 ref_audio", name)
        self.tts.ref_audio = kw["ref_audio"]
        self.tts.prompt_text = kw["prompt_text"]
        self.tts.speed_factor = kw["speed_factor"]
        if prof.api_url:
            self.tts.api_url = prof.api_url
        log.info("使用声库「%s」（%s）", prof.display, prof.version)
        return name


def _app_data_base() -> Path:
    """程序自己的数据根目录：打包后是 exe 所在目录，开发期是仓库根目录。"""
    import sys

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    """默认配置文件路径。

    支持用环境变量 `SCREENREADER_CONFIG` 覆盖 —— **测试必须用它**。
    曾经因为测试直接实例化真实 MainWindow，触发保存区域，
    把真的 data/config.json 里的声库、参考音频冲成了空，排查了半天。
    """
    override = os.environ.get("SCREENREADER_CONFIG")
    if override:
        return Path(override)
    return _app_data_base() / "data" / "config.json"
