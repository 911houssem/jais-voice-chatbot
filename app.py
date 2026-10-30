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

import os

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


async def bot(runner_args: RunnerArguments):
    """Main bot entrypoint called by the pipecat runner on every connection."""
    global _SHARED_HUB
    config = JaisConfig.from_env()
    if _SHARED_HUB is None:
        _SHARED_HUB = ModelHub(config)

    from jais_voice.asr_service import JaisASRService
    from jais_voice.llm_service import JaisLLMService
    from jais_voice.tts_service import JaisAudarTTSService

    hub = _SHARED_HUB
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
    from pipecat.runner.run import main

    main()
