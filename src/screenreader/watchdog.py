"""显存探测与自适应分档（方案 §3.7）。

背景：目标机是**游戏直播机**——游戏 + OBS 编码 + 本服务抢同一块 GPU。
"配置要求不能太高" + "多台机器配置不一" 这两条决定了**不能把显存档位写死**。

────────────────────────────────────────────────────────────────────────
显存需求实测（RTX 4080 SUPER，v2ProPlus fp16，2026-10-14）
────────────────────────────────────────────────────────────────────────
    模型加载后常驻          2007 MB
    单次推理峰值增量         789 MB
    ------------------------------------
    TTS 总计               ≈ 2796 MB   （约 2.8 GB）

OCR 刻意走 CPU，显存占用为 0（方案 §3.1）。
所以判据是：**游戏跑起来之后的空闲显存，能不能再放下这 2.8GB**。

注意 `reserve_mb` 的语义是"给游戏/OBS 继续增长留的安全余量"，
**不是**游戏已经占掉的总量——把两者叠加会把游戏算两遍。
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

# 实测 TTS 峰值占用，供分档与文档引用
TTS_VRAM_MB = 2796
# 给游戏 / OBS 继续增长留的安全余量（不是游戏已占用总量）
DEFAULT_RESERVE_MB = 768


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    util_pct: int

    def __str__(self) -> str:
        return f"[{self.index}] {self.name}  {self.used_mb}/{self.total_mb} MB 占用, 空闲 {self.free_mb} MB, 利用率 {self.util_pct}%"


@dataclass
class Tier:
    key: str
    label: str
    streaming: bool
    sample_steps: int
    synth_workers: int
    max_lines: int
    note: str


TIERS: List[Tier] = [
    Tier("A", "显存充裕（fp16 常驻，全速）", True, 32, 2, 25, "游戏外的空闲显存 ≥4.0GB"),
    Tier("B", "显存偏紧（降采样步数）", True, 16, 1, 20, "空闲 3.1-4.0GB"),
    Tier("C", "显存紧张（最少步数 + 收紧行数）", True, 8, 1, 12, "空闲 2.2-3.1GB，刚够放模型"),
    Tier("D", "显存告急（建议关掉游戏再读屏）", True, 8, 1, 8, "空闲 <2.2GB，放不下 2.8GB 的模型"),
]

# 分档阈值：budget = 空闲显存 - 安全余量，单位 MB
THRESHOLDS = (("A", 3300), ("B", 2300), ("C", 1400))


def _nvidia_smi_path() -> Optional[str]:
    p = shutil.which("nvidia-smi")
    if p:
        return p
    import os

    cand = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
    return cand if os.path.exists(cand) else None


def query_gpus() -> List[GpuInfo]:
    """用 nvidia-smi 查显存。没有 N 卡或查询失败时返回空列表。"""
    exe = _nvidia_smi_path()
    if not exe:
        return []
    try:
        # 必须走 proc.run_hidden：这是 GUI 程序，直接 subprocess 会**每 10 秒
        # 闪一次黑窗**（nvidia-smi 是控制台程序）。用户实测反馈过这个问题。
        from .proc import run_hidden

        out = run_hidden(
            [
                exe,
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
        )
        if out.returncode != 0:
            log.debug("nvidia-smi 返回 %s: %s", out.returncode, out.stderr.strip()[:200])
            return []
        gpus: List[GpuInfo] = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_mb=int(float(parts[2])),
                    used_mb=int(float(parts[3])),
                    free_mb=int(float(parts[4])),
                    util_pct=int(float(parts[5])),
                )
            )
        return gpus
    except Exception as e:
        log.debug("nvidia-smi 查询失败：%s", e)
        return []


def decide_tier(
    gpus: List[GpuInfo],
    reserve_mb: int = DEFAULT_RESERVE_MB,
    index: int = 0,
) -> Tier:
    """按「游戏跑起来之后的空闲显存 - 安全余量」决定档位。

    实测 TTS（v2ProPlus fp16）常驻 2007MB + 推理峰值增量 789MB ≈ 2.8GB，
    见本模块顶部注释。
    """
    if not gpus:
        # 没有 N 卡：TTS 只能在 CPU 上跑（慢，但不崩）
        return Tier("D", "无可用 NVIDIA 显卡", False, 8, 1, 8, "TTS 将走 CPU，延迟会明显上升")
    gpu = next((g for g in gpus if g.index == index), gpus[0])
    budget = gpu.free_mb - reserve_mb
    for key, need in THRESHOLDS:
        if budget >= need:
            return next(t for t in TIERS if t.key == key)
    return TIERS[-1]


class GpuWatchdog:
    """周期性复查显存，档位**只降不升**（避免来回抖动）。"""

    def __init__(
        self,
        reserve_mb: int = DEFAULT_RESERVE_MB,
        interval_s: float = 10.0,
        on_change: Optional[Callable[[Tier, GpuInfo], None]] = None,
        gpu_index: int = 0,
    ) -> None:
        self.reserve_mb = reserve_mb
        self.interval_s = interval_s
        self.on_change = on_change
        self.gpu_index = gpu_index
        self.tier: Optional[Tier] = None
        self.gpu: Optional[GpuInfo] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._order = {t.key: i for i, t in enumerate(TIERS)}

    def sample(self) -> tuple:
        gpus = query_gpus()
        t = decide_tier(gpus, self.reserve_mb, self.gpu_index)
        g = next((x for x in gpus if x.index == self.gpu_index), gpus[0] if gpus else None)
        return t, g

    def start(self) -> Tier:
        self.tier, self.gpu = self.sample()
        log.info("显存分档：%s — %s", self.tier.key, self.tier.label)
        if self.gpu:
            log.info("  %s", self.gpu)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gpu-watchdog", daemon=True)
        self._thread.start()
        return self.tier

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            new_tier, gpu = self.sample()
            if self.tier is None:
                continue
            # 只降不升
            if self._order[new_tier.key] > self._order[self.tier.key]:
                log.warning(
                    "显存收紧，降档 %s -> %s（%s）",
                    self.tier.key,
                    new_tier.key,
                    new_tier.label,
                )
                self.tier, self.gpu = new_tier, gpu
                if self.on_change:
                    try:
                        self.on_change(new_tier, gpu)
                    except Exception as e:
                        log.warning("档位回调异常：%s", e)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
