"""Pipecat LLM service backed by a locally loaded Jais-3 model.

Follows the same frame flow as ``pipecat.services.openai.base_llm``:
``process_frame`` handles ``LLMContextFrame`` by wrapping generation in
LLMFullResponseStart/End frames and streaming text chunks downstream, where
the TTS service aggregates them into spoken sentences.
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Mapping

import torch
from loguru import logger
from typing_extensions import override

from pipecat.frames.frames import (
    Frame,
    FrameDirection,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

from .config import LlmConfig
from .hub import ModelHub

# Reasoning residue: Jais-3 takes reasoning_effort via its chat template and,
# for voice latency, we default to "low" (no reasoning). Belt-and-suspenders:
# strip any leftover think/thought tags before pushing text downstream.

_THINK_TAG_RE = re.compile(r"<\/?\s*(?:think|thought)\b[^>]*>", re.IGNORECASE)


class JaisLLMService(LLMService):
    """Arabic chat LLM using Jais-3 with voice-friendly sampling defaults."""

    def __init__(self, *, hub: ModelHub, **kwargs):
        cfg = hub.config.llm
        super().__init__(
            settings=LLMSettings(
                model=cfg.repo,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                max_tokens=cfg.max_new_tokens,
            ),
            **kwargs,
        )
        self._hub = hub
        self._sampling = cfg

    def can_generate_metrics(self) -> bool:
        return True

    # -- frame flow ----------------------------------------------------- #

    @override
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        try:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self._process_context(frame.context)
        except Exception as e:
            logger.exception("LLM completion error")
            await self.push_error(error_msg=f"Error during completion: {e}", exception=e)
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    async def _process_context(self, context: LLMContext):
        """Trim context and stream a completion, chunk by chunk."""
        await self.start_ttfb_metrics()

        messages = self._trim_messages(context)
        model, tokenizer = await self._hub.load_llm()
        inputs = await asyncio.to_thread(
            self._build_inputs, tokenizer, messages, model.device
        )
        if inputs is None:
            logger.warning("Could not build LLM inputs; skipping completion")
            await self.stop_ttfb_metrics()
            return

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        thread, error_box = self._spawn_generation(model, inputs, streamer)
        emitted = False
        try:
            while True:
                piece = await asyncio.to_thread(next, streamer, None)
                if piece is None:
                    break
                piece = _THINK_TAG_RE.sub("", piece)
                if piece:
                    if not emitted:
                        emitted = True
                        await self.stop_ttfb_metrics()
                    await self._push_llm_text(piece)
            if error_box[0] is not None:
                raise error_box[0]
            if not emitted:
                logger.warning("LLM produced no visible answer text")
        finally:
            # Detach the streamer so the generator ends even on cancel.
            try:
                streamer.end()
            except Exception:
                pass
            thread.join(timeout=0.0)
            await self.stop_ttfb_metrics()

    # -- generation helpers --------------------------------------------- #

    def _trim_messages(self, context: LLMContext) -> list[dict]:
        """Keep system/developer messages plus the latest N user/assistant turns."""
        keep = self._hub.config.llm.max_context_messages
        messages = [
            m if isinstance(m, Mapping) else dict(m) for m in context.get_messages()
        ]
        system = [m for m in messages if m.get("role") in ("system", "developer")]
        rest = [m for m in messages if m.get("role") not in ("system", "developer")]
        return system + rest[-keep:] if system else rest[-keep:]

    def _build_inputs(self, tokenizer, messages: list[dict], device):
        """Tokenize a chat turn with the model's template (dict or tensor both OK)."""
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
                reasoning_effort=self._hub.config.llm.reasoning_effort,
            )
        except TypeError:
            # Template may not accept this kwarg; retry with the plain call.
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        if isinstance(inputs, Mapping) or hasattr(inputs, "keys"):
            inputs.pop("token_type_ids", None)
            return inputs.to(device)
        if isinstance(inputs, torch.Tensor):
            return inputs.to(device)
        return None

    def _spawn_generation(
        self, model, inputs, streamer
    ) -> tuple[threading.Thread, list[Exception | None]]:
        """Start ``model.generate`` on a daemon thread; error surfaces via the box."""
        error_box: list[Exception | None] = [None]
        thread = threading.Thread(
            target=self._generate_run,
            args=(model, inputs, streamer, error_box),
            daemon=True,
        )
        thread.start()
        return thread, error_box

    def _generate_run(self, model, inputs, streamer, error_box):
        try:
            gen_kwargs = dict(
                streamer=streamer,
                do_sample=True,
                temperature=self._sampling.temperature,
                top_p=self._sampling.top_p,
                top_k=self._sampling.top_k,
                repetition_penalty=self._sampling.repetition_penalty,
                max_new_tokens=self._sampling.max_new_tokens,
            )
            with torch.inference_mode():
                if hasattr(inputs, "keys"):  # BatchEncoding: input_ids + extras
                    model.generate(**inputs, **gen_kwargs)
                else:
                    model.generate(inputs, **gen_kwargs)
        except Exception as e:
            error_box[0] = e
        finally:
            try:
                streamer.end()  # make next(streamer, None) return None
            except Exception:
                pass


__all__ = ["JaisLLMService"]
