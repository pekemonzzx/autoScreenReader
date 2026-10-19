"""TTS 后端。"""

from __future__ import annotations

from typing import Any, Dict

from .base import StreamFormat, TtsBackend


def create_backend(cfg: Any) -> TtsBackend:
    """按配置创建后端。cfg 是 screenreader.config.TtsConfig。"""
    name = getattr(cfg, "backend", "gsv_api")
    common: Dict[str, Any] = dict(
        ref_audio=cfg.ref_audio,
        prompt_text=cfg.prompt_text,
        prompt_lang=cfg.prompt_lang,
        text_lang=cfg.text_lang,
        speed_factor=cfg.speed_factor,
    )
    if name == "gsv_api":
        from .gsv_api import GsvApiBackend

        return GsvApiBackend(
            api_url=cfg.api_url,
            split_method=cfg.split_method,
            sample_steps=cfg.sample_steps,
            timeout_s=cfg.timeout_s,
            **common,
        )
    if name == "onnx":
        raise NotImplementedError(
            "ONNX 后端属于 P3 计划（LunaVox / GPT-SoVITS ONNX 导出），尚未实现。"
        )
    raise ValueError(f"未知的 TTS 后端：{name}")


__all__ = ["TtsBackend", "StreamFormat", "create_backend"]
