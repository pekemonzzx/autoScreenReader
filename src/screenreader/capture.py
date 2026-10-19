"""截屏模块。

方案 v2.1 §3.3：
  - 主路径 DXGI Desktop Duplication（dxcam）——独占全屏 / 无边框窗口都能抓到
  - 回退路径 mss（GDI）——仅本地 Console 会话可靠，远控虚拟显示器下易黑屏
  - 进程 DPI-aware，区域坐标一律按「物理像素」处理
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

Region = Tuple[int, int, int, int]  # (left, top, width, height)，相对显示器左上角


# --------------------------------------------------------------------------- #
# DPI
# --------------------------------------------------------------------------- #
def enable_dpi_awareness() -> None:
    """让进程拿到物理像素坐标，避免 125%/150% 缩放导致框选区域错位。"""
    if sys.platform != "win32":
        return
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            log.warning("DPI 感知设置失败，缩放非 100%% 时区域坐标可能偏移")


# --------------------------------------------------------------------------- #
# 显示器枚举
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MonitorInfo:
    index: int
    left: int
    top: int
    width: int
    height: int

    def __str__(self) -> str:
        return f"[{self.index}] {self.width}x{self.height} @ ({self.left},{self.top})"


def list_monitors() -> List[MonitorInfo]:
    """枚举物理显示器。索引与 dxcam 的 output_idx 对齐（基于 mss 的枚举顺序）。"""
    import mss

    out: List[MonitorInfo] = []
    with mss.mss() as sct:
        # monitors[0] 是所有屏幕的并集，跳过
        for i, m in enumerate(sct.monitors[1:]):
            out.append(MonitorInfo(i, m["left"], m["top"], m["width"], m["height"]))
    return out


# --------------------------------------------------------------------------- #
# 截屏器
# --------------------------------------------------------------------------- #
class ScreenCapturer:
    """统一截屏接口，内部按 backend 选择 DXGI 或 mss。

    注意：mss 与 dxcam 的实例都有线程亲和性，不要跨线程共享同一个 ScreenCapturer。
    """

    def __init__(
        self,
        monitor_index: int = 0,
        backend: str = "auto",
        region: Optional[Region] = None,
    ) -> None:
        self.monitors = list_monitors()
        if not self.monitors:
            raise RuntimeError("未枚举到任何显示器")
        if not 0 <= monitor_index < len(self.monitors):
            raise ValueError(
                f"monitor_index={monitor_index} 越界，可用 0..{len(self.monitors) - 1}"
            )

        self.monitor = self.monitors[monitor_index]
        self.region: Region = region or (0, 0, self.monitor.width, self.monitor.height)
        self.backend = self._resolve_backend(backend)
        self._last_frame: Optional[np.ndarray] = None

        self._cam = None
        self._sct = None
        if self.backend == "dxgi":
            self._init_dxgi()
        else:
            self._init_mss()

        log.info(
            "截屏后端=%s 显示器=%s 区域=%s", self.backend, self.monitor, self.region
        )

    # -- 初始化 ------------------------------------------------------------ #
    def _resolve_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend
        try:
            import dxcam  # noqa: F401

            return "dxgi"
        except Exception as e:  # pragma: no cover
            log.warning("dxcam 不可用（%s），回退 mss", e)
            return "mss"

    def _init_dxgi(self) -> None:
        import dxcam

        self._cam = dxcam.create(
            output_idx=self.monitor.index, output_color="BGR"
        )
        if self._cam is None:
            raise RuntimeError(
                f"dxcam.create(output_idx={self.monitor.index}) 失败"
            )

    def _init_mss(self) -> None:
        import mss

        self._sct = mss.mss()

    # -- 抓取 -------------------------------------------------------------- #
    def _grab_dxgi(self) -> Optional[np.ndarray]:
        left, top, width, height = self.region
        abs_box = (
            self.monitor.left + left,
            self.monitor.top + top,
            self.monitor.left + left + width,
            self.monitor.top + top + height,
        )
        frame = self._cam.grab(region=abs_box)
        if frame is None:
            return None
        # 形状自检：dxcam 的 region 语义在不同版本/多屏下可能是「输出内相对坐标」，
        # 一旦尺寸对不上就退回"抓全屏再裁剪"，保证正确性优先。
        if frame.shape[1] != width or frame.shape[0] != height:
            full = self._cam.grab()
            if full is None:
                return None
            return np.ascontiguousarray(full[top : top + height, left : left + width])
        return frame

    def _grab_mss(self) -> Optional[np.ndarray]:
        left, top, width, height = self.region
        shot = self._sct.grab(
            {
                "left": self.monitor.left + left,
                "top": self.monitor.top + top,
                "width": width,
                "height": height,
            }
        )
        # mss 返回 BGRA
        return np.asarray(shot)[:, :, :3]

    def grab(
        self, retries: int = 3, retry_delay: float = 0.002
    ) -> Optional[np.ndarray]:
        """抓一帧 BGR。

        dxcam 在画面无变化时会返回 None，重试若干次后返回上一帧缓存
        （屏幕阅读场景下"内容没变"用旧帧是正确的语义）。
        """
        for i in range(retries):
            frame = (
                self._grab_dxgi() if self.backend == "dxgi" else self._grab_mss()
            )
            if frame is not None:
                self._last_frame = frame
                return frame
            if i + 1 < retries:
                time.sleep(retry_delay)
        return self._last_frame

    def close(self) -> None:
        if self._cam is not None:
            try:
                self._cam.release()
            except Exception:
                pass
            self._cam = None
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None

    def __enter__(self) -> "ScreenCapturer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def selfcheck(region: Optional[Region] = None, backend: str = "auto") -> dict:
    """抓一帧并做基本健康检查，返回诊断信息。

    用于安装包首次启动自检 + 远程部署排障（方案 §4.2 第 6 条）。
    """
    info: dict = {"ok": False}
    try:
        with ScreenCapturer(backend=backend, region=region) as cap:
            t0 = time.perf_counter()
            frame = cap.grab()
            dt = (time.perf_counter() - t0) * 1000
            if frame is None:
                info["error"] = "抓帧返回 None（DXGI 可能被独占全屏或远控虚拟显示器阻断）"
                return info
            mean = float(frame.mean())
            info.update(
                ok=mean > 1.0,  # 全黑帧多半意味着抓屏被阻断
                backend=cap.backend,
                monitor=str(cap.monitor),
                region=cap.region,
                shape=list(frame.shape),
                mean_luma=round(mean, 2),
                first_frame_ms=round(dt, 1),
            )
            if mean <= 1.0:
                info["warning"] = (
                    "抓到的是全黑帧：检查是否被远控虚拟显示器接管、"
                    "或使用了独占全屏且 DXGI 不可用"
                )
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info
