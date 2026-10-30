"""Pipecat TTS service backed by Audar-TTS-V1-Flash + NeuCodec.

The Audar backbone draws discrete speech tokens and the codec decodes them to
a 24 kHz waveform (float32). We re-chunk into ~250 ms frames so the audio
flows smoothly into pipecat's 10 ms transport chunker.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

from .hub import ModelHub


class JaisAudarTTSService(TTSService):
    """Arabic voice cloning TTS using the demo registry or a custom voice."""

    def __init__(self, *, hub: ModelHub, **kwargs):
        cfg = hub.config.tts
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(
                model=cfg.repo,
                voice=cfg.voice,
                language=Language.AR,
            ),
            sample_rate=hub.config.sample_rate_out,
            **kwargs,
        )
        self._hub = hub
        self._resampler = create_stream_resampler()
        self._voice_ready = False

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame):
        """Load TTS + prepare the reference voice before the first synthesis."""
        await super().start(frame)
        await self._hub.prepare_tts_voice()
        self._voice_ready = True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize ``text`` and yield audio frames."""
        try:
            await self.start_tts_usage_metrics(text)
            if not self._voice_ready:
                await self._hub.prepare_tts_voice()
                self._voice_ready = True

            model, tok, codec = await self._hub.load_tts()
            wav = await asyncio.to_thread(
                self._hub.synthesize_codes, model, tok, codec, text
            )

            # float32 [-1, 1] @ 24 kHz -> int16 PCM bytes at the pipeline rate.
            audio = np.clip(wav, -1.0, 1.0).astype(np.float32)
            pcm = (audio * 32767.0).astype(np.int16).tobytes()
            if self.sample_rate != 24000:
                pcm = await self._resampler.resample(pcm, 24000, self.sample_rate)

            chunk_bytes = int(self.sample_rate * self._hub.config.tts_chunk_secs) * 2
            for offset in range(0, len(pcm), chunk_bytes):
                yield TTSAudioRawFrame(
                    audio=pcm[offset:offset + chunk_bytes],
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as e:
            logger.exception("Audar-TTS synthesis failed")
            yield ErrorFrame(error=f"Audar-TTS synthesis error: {e}")


__all__ = ["JaisAudarTTSService"]
