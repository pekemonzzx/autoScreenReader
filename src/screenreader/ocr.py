"""OCR 模块。

方案 v2.1 §3.1：引擎 RapidOCR（PP-OCR 系 ONNX 模型），**默认跑在 CPU 上**，
刻意不占显存，把 GPU 全部留给游戏与 TTS（见方案 §3.6）。

────────────────────────────────────────────────────────────────────────
P1 实测踩到的三个坑（都在本文件里做了修正，勿回退）
────────────────────────────────────────────────────────────────────────

1) `Det.limit_type` 默认是 `min` / `736`，会把**短边放大到 736**。
   实测 900x300 的截屏区域被放大成 2208x736（像素量 ×6），det 从 67ms 涨到 310ms。
   → 改为 `max` / 960：长边超过 960 才缩小，小图原样送入。

2) `EngineConfig.onnxruntime.intra_op_num_threads` 默认 `auto`，
   在 Intel 大小核（如 i7-14700KF：8 P-core + 12 E-core）上会把线程池铺满
   全部逻辑核，栅栏同步要等最慢的 E-core 线程，实测慢 3-4 倍。
   → 必须显式限制线程数，并在目标机上自动标定（见 tune_intra_threads）。

3) `enable_cpu_mem_arena` 默认 `false`，每次 run 都重新分配全部工作内存。
   叠加安全软件（本机装了火绒）的内存挂钩后开销明显。
   → 打开 arena。

效果（本机 i7-14700KF，关/开 cls 均实测）：

    | 区域         | 默认配置 | 调优后 |
    | 520x140      | 198ms    |  62ms  |
    | 900x300      | 546ms    | 147ms  |
    | 1280x720     | 825ms    | 288ms  |
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 默认参数（P1 实测确定，勿随意改动）
# --------------------------------------------------------------------------- #
def default_params(det_limit_side_len: int = 960, use_cls: bool = True) -> Dict[str, Any]:
    return {
        # 坑 1：不要让 det 放大输入
        "Det.limit_type": "max",
        "Det.limit_side_len": det_limit_side_len,
        # 竖排/翻转文本需要 cls；实测代价仅 ~30ms，保留
        "Global.use_cls": use_cls,
        # RapidOCR 默认按 info 级别往 stdout 刷一堆模型加载日志，压到 error
        "Global.log_level": "error",
        # 坑 3：打开内存池
        "EngineConfig.onnxruntime.enable_cpu_mem_arena": True,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    }


# 候选线程数。上限刻意压到 16 —— 超过后栅栏同步的代价会反过来吃掉收益。
THREAD_CANDIDATES: Tuple[int, ...] = (4, 6, 8, 10, 12, 16)


def _app_data_dir() -> Path:
    """程序自己的数据目录。

    打包后是 exe 所在目录，开发期是仓库根目录。
    优先用它而不是 AppData：绿色包本身就该自包含，而且受限环境下
    Python 子进程可能没有 AppData 的写权限（本机实测 WinError 5）。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "data"
    return Path(__file__).resolve().parents[2] / "data"


def _cache_path() -> Optional[Path]:
    """标定结果缓存路径。逐个候选位置尝试，全失败就返回 None（只是不缓存，不影响功能）。"""
    candidates = [
        os.environ.get("SCREENREADER_DATA"),
        str(_app_data_dir()),
        os.environ.get("LOCALAPPDATA") and str(Path(os.environ["LOCALAPPDATA"]) / "screenreader"),
        os.environ.get("APPDATA") and str(Path(os.environ["APPDATA"]) / "screenreader"),
        tempfile.gettempdir(),
    ]
    for base in candidates:
        # 只接受绝对路径：某些环境（沙箱/精简容器）里 gettempdir() 可能返回相对路径，
        # 那会在工作目录里凭空建出 screenreader/ 目录。
        if not base or not os.path.isabs(base):
            continue
        try:
            d = Path(base)
            d.mkdir(parents=True, exist_ok=True)
            # 确认真的可写（安全软件/权限策略可能只允许建目录不允许写文件）
            probe = d / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            try:
                probe.unlink()  # 安全软件可能短暂锁住新文件，删不掉不算失败
            except OSError:
                pass
            return d / "ocr_threads.json"
        except Exception as e:
            log.debug("缓存候选目录不可用 %s: %s", base, e)
            continue
    log.warning("找不到可写的缓存目录，OCR 线程数标定结果将不会被缓存")
    return None


def _machine_key(det_limit: int, shape: Sequence[int]) -> str:
    """缓存键 = 机器指纹 + det 上限 + 图像尺寸。

    分两段是为了能"前缀匹配"：换一个区域尺寸时，如果这台机器标定过别的尺寸，
    可以直接复用那个值，而不是重新花十秒扫一遍。
    """
    return f"{_machine_prefix(det_limit)}|{shape[0]}x{shape[1]}"


def _machine_prefix(det_limit: int) -> str:
    return "|".join(
        [
            platform.node(),
            str(os.cpu_count()),
            platform.processor() or platform.machine(),
            f"det{det_limit}",
        ]
    )


def _heuristic_threads() -> int:
    """标定失败时的兜底：不超过 12，且不超过逻辑核数的 3/7。

    14700K(28 逻辑核) -> 12；16 逻辑核 -> 6；8 逻辑核 -> 3。
    """
    n = os.cpu_count() or 4
    return max(2, min(12, n * 3 // 7))


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #
@dataclass
class TextLine:
    text: str
    score: float
    box: Any  # Nx2 顶点数组
    top: float = 0.0
    left: float = 0.0


@dataclass
class OcrResult:
    lines: List[TextLine] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def texts(self) -> List[str]:
        return [ln.text for ln in self.lines]

    def as_text(self, sep: str = "\n") -> str:
        return sep.join(self.texts)

    def __bool__(self) -> bool:
        return bool(self.lines)


# --------------------------------------------------------------------------- #
# 线程数标定
# --------------------------------------------------------------------------- #
def tune_intra_threads(
    image: np.ndarray,
    det_limit_side_len: int = 960,
    candidates: Sequence[int] = THREAD_CANDIDATES,
    repeat: int = 3,
    use_cache: bool = True,
) -> int:
    """在真实图片上标定最优 intra_op 线程数，结果缓存到本机。

    目标机配置不一，这一步必须在目标机上跑一次（几秒钟），
    否则会一直吃 Intel 大小核带来的 3-4 倍性能损失。
    """
    key = _machine_key(det_limit_side_len, image.shape)
    prefix = _machine_prefix(det_limit_side_len)
    cache = _cache_path()

    if use_cache and cache is not None and cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if key in data:
                n = int(data[key])
                log.info("OCR 线程数（缓存命中）：%d", n)
                return n
            # 同一台机器、同一个 det 上限，但区域尺寸不同 —— 直接复用，别重扫
            reused = [(k, int(v)) for k, v in data.items() if k.startswith(prefix)]
            if reused:
                n = reused[-1][1]
                log.info("OCR 线程数（复用同机器其它尺寸的标定值）：%d", n)
                data[key] = n
                try:
                    cache.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                except Exception:
                    pass
                return n
        except Exception:
            pass

    best_n, best_ms = None, float("inf")
    for n in candidates:
        try:
            engine = OcrEngine(
                det_limit_side_len=det_limit_side_len, intra_threads=n, warmup_runs=1
            )
            engine.read(image)  # 再预热一次，排除首次分配
            ts = []
            for _ in range(repeat):
                t = time.perf_counter()
                engine.read(image)
                ts.append((time.perf_counter() - t) * 1000)
            p50 = sorted(ts)[len(ts) // 2]
        except Exception as e:
            log.warning("线程数 %d 标定失败：%s", n, e)
            continue
        log.info("OCR 线程标定 intra=%d -> p50=%.1fms", n, p50)
        if p50 < best_ms:
            best_n, best_ms = n, p50

    if best_n is None:
        best_n = _heuristic_threads()
        log.warning("线程标定全部失败，使用启发值 %d", best_n)

    if use_cache and cache is not None:
        try:
            data = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
            data[key] = best_n
            cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    log.info("OCR 线程标定结果：intra_op_num_threads=%d（p50=%.1fms）", best_n, best_ms)
    return best_n


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
class OcrEngine:
    """RapidOCR 的薄封装，统一 2.x / 3.x 的返回结构，并带上 P1 实测的调优参数。"""

    def __init__(
        self,
        det_limit_side_len: int = 960,
        use_cls: bool = True,
        intra_threads: Optional[int] = None,
        device: str = "cpu",
        engine: Optional[str] = None,
        warmup_runs: int = 0,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        from rapidocr import RapidOCR

        self.det_limit_side_len = det_limit_side_len
        self.device = device
        self.intra_threads = intra_threads or _heuristic_threads()

        params = default_params(det_limit_side_len, use_cls)
        params["EngineConfig.onnxruntime.intra_op_num_threads"] = self.intra_threads
        if device == "dml":
            params["EngineConfig.onnxruntime.use_dml"] = True
        if extra_params:
            params.update(extra_params)

        kwargs: Dict[str, Any] = {"params": params}
        if engine:
            kwargs["engine"] = engine
        try:
            self._engine = RapidOCR(**kwargs)
        except TypeError:
            # 老版本签名不接受 engine/params
            self._engine = RapidOCR()

        self._warmup_runs = warmup_runs
        log.info(
            "OCR 就绪：RapidOCR device=%s intra_threads=%d det_limit=%d cls=%s",
            device,
            self.intra_threads,
            det_limit_side_len,
            use_cls,
        )

    @classmethod
    def auto(
        cls,
        probe_image: np.ndarray,
        det_limit_side_len: int = 960,
        use_cls: bool = True,
        **kwargs: Any,
    ) -> "OcrEngine":
        """先标定线程数，再用最优线程数建引擎。首次启动约 3-6 秒，之后走缓存。"""
        n = tune_intra_threads(probe_image, det_limit_side_len)
        eng = cls(
            det_limit_side_len=det_limit_side_len,
            use_cls=use_cls,
            intra_threads=n,
            **kwargs,
        )
        eng.warmup(probe_image, runs=2)
        return eng

    def warmup(self, image: np.ndarray, runs: int = 2) -> None:
        for _ in range(runs):
            self.read(image)

    # ------------------------------------------------------------------ #
    def read(self, image: np.ndarray) -> OcrResult:
        t0 = time.perf_counter()
        raw = self._engine(image)
        elapsed = (time.perf_counter() - t0) * 1000

        lines = self._parse(raw)
        if lines:
            heights = [
                max(b[:, 1]) - min(b[:, 1])
                for b in (ln.box for ln in lines)
                if b is not None and len(b)
            ]
            tol = (float(np.median(heights)) * 0.6) if heights else 10.0
            lines.sort(key=lambda ln: (round(ln.top / max(tol, 1.0)), ln.left))

        return OcrResult(lines=lines, elapsed_ms=elapsed)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse(raw: Any) -> List[TextLine]:
        """兼容 RapidOCR 3.x（RapidOCROutput）与 2.x（(list, elapse) 元组）。"""
        boxes = txts = scores = None

        if hasattr(raw, "txts"):  # 3.x
            boxes, txts, scores = raw.boxes, raw.txts, raw.scores
        elif isinstance(raw, tuple):  # 2.x
            result = raw[0]
            if result:
                boxes = [r[0] for r in result]
                txts = [r[1] for r in result]
                scores = [r[2] for r in result]

        if txts is None:
            return []

        out: List[TextLine] = []
        for i in range(len(txts)):
            box = boxes[i] if boxes is not None else None
            arr = np.asarray(box, dtype=float) if box is not None else None
            top = float(arr[:, 1].min()) if arr is not None and arr.size else float(i * 10)
            left = float(arr[:, 0].min()) if arr is not None and arr.size else 0.0
            out.append(
                TextLine(
                    text=str(txts[i]),
                    score=float(scores[i]) if scores is not None else 0.0,
                    box=arr,
                    top=top,
                    left=left,
                )
            )
        return out


def warmup(engine: OcrEngine, image: np.ndarray, runs: int = 2) -> None:
    """兼容旧调用：首次调用含 ONNX 会话初始化，必须排除在基准之外。"""
    engine.warmup(image, runs)
