"""Real-time Arabic voice chatbot entrypoint.

Architecture (cascade):

    transport.input() -> STT (Jais-ASR) -> user context aggregator
    -> LLM (Jais-3)   -> TTS (Audar-TTS) -> transport.output()
    -> assistant context aggregator

VAD is attached to the user context aggregator (pipecat 1.15 style).

Run:
    python app.py -t webrtc        # built-in browser UI
"""

from __future__ import annotations

import asyncio
import threading

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.params import PipelineParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

from jais_voice.config import JaisConfig
from jais_voice.hub import ModelHub

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}

# Models survive across browser connections; reload only on bot restart.
_SHARED_HUB: ModelHub | None = None
_HUB_LOCK = threading.Lock()


def _get_hub(config: JaisConfig) -> ModelHub:
    """One ModelHub per process, so models survive/reuse across connections."""
    global _SHARED_HUB
    with _HUB_LOCK:
        if _SHARED_HUB is None:
            _SHARED_HUB = ModelHub(config)
    return _SHARED_HUB


async def _warm_in_loop(hub: ModelHub) -> None:
    """Warm-up fallback running on the server's event loop."""
    try:
        await hub.warmup()
        logger.info("Model warm-up (server loop) finished")
    except Exception as e:
        logger.warning("Model warm-up failed: {}", e)


def _background_preload() -> None:
    """Load all models + voice reference while the server is booting.

    Colab users click Connect within seconds; without this the first
    connection would block for minutes on model load. The hub singleton
    is shared, so the pipeline reuses whatever this thread already loaded.
    """
    global _SHARED_HUB
    config = JaisConfig.from_env()
    hub = _get_hub(config)
    try:
        asyncio.run(hub.warmup())
        logger.info("Background model warm-up finished")
    except Exception as e:
        # Fall through to the services' lazy loading path.
        logger.warning("Background model warm-up failed: {}", e)


async def bot(runner_args: RunnerArguments):
    """Main bot entrypoint called by the pipecat runner on every connection."""
    config = JaisConfig.from_env()
    hub = _get_hub(config)

    # Kick a warm-up in the background if the module-level one hasn't run yet
    # (e.g. the bot was imported after the server booted).
    if not hub._warmed_up:
        asyncio.get_running_loop().create_task(_warm_in_loop(hub))

    from jais_voice.asr_service import JaisASRService
    from jais_voice.llm_service import JaisLLMService
    from jais_voice.tts_service import JaisAudarTTSService

    transport: BaseTransport = await create_transport(runner_args, transport_params)

    stt = JaisASRService(hub=hub)
    llm = JaisLLMService(hub=hub)
    tts = JaisAudarTTSService(hub=hub)

    context = LLMContext(
        messages=[{"role": "system", "content": config.llm.system_prompt}]
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),     # web mic frames
            stt,                   # Arabic speech segments -> text
            user_aggregator,       # merges transcripts into LLM context
            llm,                   # Arabic chat completion (streaming)
            tts,                   # text -> 24 kHz speech
            transport.output(),    # speaker frames
            assistant_aggregator,  # records bot turns in context
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=config.sample_rate_in,
            audio_out_sample_rate=config.sample_rate_out,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        # Long model loads + slow T4 turns must not trip the watchdogs.
        setup_timeout_secs=600.0,
        start_timeout_secs=600.0,
        idle_timeout_secs=None,  # silence should not kill the bot
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation with a short Arabic greeting.
        context.add_message({"role": "developer", "content": config.llm.intro_prompt})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


if __name__ == "__main__":
    # Warm the GPU models while uvicorn boots, so the first Connect is fast.
    threading.Thread(target=_background_preload, daemon=True).start()

    from pipecat.runner.run import main

    main()
