"""screenreader —— 截屏 → OCR → 分句 TTS → 直播软件 的桌面读屏工具。"""

__version__ = "0.2.0"

from .audio import DeviceInfo, Player, find_output_device, list_output_devices
from .cache import AudioCache, make_key, wav_from_pcm, wav_info
from .capture import (
    MonitorInfo,
    ScreenCapturer,
    enable_dpi_awareness,
    list_monitors,
    selfcheck,
)
from .config import AppConfig, default_config_path
from .ocr import OcrEngine, OcrResult, TextLine, default_params, tune_intra_threads
from .pipeline import ReadResult, ScreenReaderPipeline
from .segment import SegmentOptions, normalize_text, split_text
from .tts import StreamFormat, TtsBackend, create_backend
from .watchdog import GpuInfo, GpuWatchdog, Tier, decide_tier, query_gpus

__all__ = [
    # capture
    "MonitorInfo",
    "ScreenCapturer",
    "enable_dpi_awareness",
    "list_monitors",
    "selfcheck",
    # ocr
    "OcrEngine",
    "OcrResult",
    "TextLine",
    "default_params",
    "tune_intra_threads",
    # segment
    "SegmentOptions",
    "split_text",
    "normalize_text",
    # tts
    "TtsBackend",
    "StreamFormat",
    "create_backend",
    # audio
    "Player",
    "DeviceInfo",
    "list_output_devices",
    "find_output_device",
    # cache
    "AudioCache",
    "make_key",
    "wav_from_pcm",
    "wav_info",
    # watchdog
    "GpuInfo",
    "GpuWatchdog",
    "Tier",
    "decide_tier",
    "query_gpus",
    # pipeline / config
    "ScreenReaderPipeline",
    "ReadResult",
    "AppConfig",
    "default_config_path",
]
