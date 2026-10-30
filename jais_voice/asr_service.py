"""Pipecat STT service backed by a locally loaded Jais-ASR model.

Subclasses ``SegmentedSTTService``: pipecat's VAD emits whole speech
segments and each is transcribed once (good fit for a 6B LLM-style ASR).
Audio arrives as raw 16-bit PCM at the pipeline sample rate.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger
from typing_extensions import override

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

from .hub import ModelHub


class JaisASRService(SegmentedSTTService):
    """Arabic speech-to-text using Jais-ASR (``IFM/Jais-ASR``)."""

    def __init__(self, *, hub: ModelHub, **kwargs):
        """Args: hub - loaded-shares models; kwargs forwarded to SegmentedSTTService."""
        super().__init__(
            settings=STTSettings(language=Language.AR),
            sample_rate=hub.config.sample_rate_in,
            **kwargs,
        )
        self._hub = hub

    @property
    @override
    def wants_wav_segments(self) -> bool:
        # The Jais-ASR processor consumes raw PCM, not WAV containers.
        return False

    def can_generate_metrics(self) -> bool:
        return True

    @override
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe one VAD speech segment.

        Args:
            audio: Raw 16-bit signed little-endian PCM (mono) at ``sample_rate``.
        """
        await self.start_processing_metrics()

        # 16-bit PCM -> float32 in [-1, 1]; truncate overlong segments.
        waveform = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        max_samples = int(self._hub.config.asr.max_segment_secs * self.sample_rate)
        if waveform.shape[0] > max_samples:
            logger.warning(
                "STT segment too long (%.1fs), truncating to last %.1fs",
                waveform.shape[0] / self.sample_rate,
                self._hub.config.asr.max_segment_secs,
            )
            waveform = waveform[-max_samples:]

        try:
            result = await self._hub.transcribe(waveform, self.sample_rate)
        except Exception as e:  # keep the pipeline alive on transient errors
            logger.exception("Jais-ASR transcription failed")
            await self.stop_processing_metrics()
            yield ErrorFrame(error=f"Jais-ASR transcription error: {e}")
            return

        await self.stop_processing_metrics()

        if result is None or not result.text.strip():
            return  # empty/non-speech segment: nothing to push

        logger.debug("Transcription: [{}]", result.text)
        yield TranscriptionFrame(
            result.text,
            self._user_id,
            time_now_iso8601(),
            Language.AR,
        )


__all__ = ["JaisASRService"]
