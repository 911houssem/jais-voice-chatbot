"""Environment-driven configuration for the Jais Arabic voice chatbot.

All knobs are read from environment variables (see ``.env.example``). Config
values are plain dataclasses; the model-loading code lives in ``hub.py``.

Defaults target a 16 GB GPU (e.g. Colab T4):
- STT: quantized Jais-ASR checkpoint (W4A16-AWQ, ~3.4 GB VRAM)
- LLM: Jais-3-0.9B in fp16 (~1.8 GB)
- TTS: Audar-TTS-V1-Flash + NeuCodec (~2 GB combined)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Model repositories (as published on Hugging Face).
ASR_REPO = "IFM/Jais-ASR"
ASR_REPO_AWQ = "IFM/Jais-ASR-W4A16-AWQ"
LLM_REPO = "inception42/Jais-3-0.9B"
TTS_REPO = "audarai/Audar-TTS-V1-Flash"
TTS_CODEC_REPO = "neuphonic/neucodec"

# Default Arabic persona prompt. Kept short for low latency and voice output.
DEFAULT_SYSTEM_PROMPT = (
    "أنت مساعد صوتي عربي ودود يُدعى جيس. تحدث دائماً بالعربية الفصحى "
    "بأسلوب طبيعي وقصير جداً، جملة واحدة أو جملتان كحد أقصى، لأن ردودك "
    "تُقرأ بصوت عالٍ. لا تستخدم ملاكمة أو قوائم أو رموزاً. إذا سُئلت "
    "بإنجليزية، أجب نفسك بالعربية."
)

# Developer message queued on first client connection (kick-starts the bot).
DEFAULT_INTRO_PROMPT = (
    "قدّم نفسك باختصار شديد في جملة واحدة ومرّحباً المستخدم."
)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


@dataclass
class AsrConfig:
    """Speech-to-text (Jais-ASR) settings.

    ``repo="auto"`` picks the fp16 checkpoint when the GPU has >= 20 GB VRAM,
    otherwise the W4A16-AWQ quantized checkpoint (needs ``compressed-tensors``).
    """

    repo: str = "auto"
    dtype: str = "auto"  # "auto" -> fp16 (T4 has no bf16)
    device: str = "cuda"
    max_segment_secs: float = 35.0  # truncate very long user turns
    max_new_tokens: int = 256


@dataclass
class LlmConfig:
    """Chat LLM (Jais-3) settings.

    The model is a reasoning model. ``reasoning_effort="low"`` (per its card,
    "no reasoning") is the right default for voice agents: `high` is more
    accurate but far too slow to speak comfortably.
    """

    repo: str = LLM_REPO
    dtype: str = "fp16"
    device: str = "cuda"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    intro_prompt: str = DEFAULT_INTRO_PROMPT
    reasoning_effort: str = "low"
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.0
    max_new_tokens: int = 200  # short, spoken answers
    max_context_messages: int = 16  # trim history beyond system + latest N


@dataclass
class TtsConfig:
    """Text-to-speech (Audar-TTS-V1-Flash + NeuCodec) settings."""

    repo: str = TTS_REPO
    codec_repo: str = TTS_CODEC_REPO
    device: str = "cuda"
    voice: str = "demo_female_1"  # built-in demo voice id or "custom"
    ref_wav_path: str | None = None  # required when voice == "custom"
    ref_text: str | None = None  # required when voice == "custom"
    ref_trim_secs: float = 12.0  # trim reference clip to <= this length
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_new_tokens: int = 1500
    min_new_tokens: int = 50


@dataclass
class JaisConfig:
    """Aggregate configuration for the whole pipeline."""

    asr: AsrConfig = field(default_factory=AsrConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    sample_rate_in: int = 16000   # pipeline audio-in rate == Jais-ASR native
    sample_rate_out: int = 24000  # pipeline audio-out rate == TTS native
    tts_chunk_secs: float = 0.25  # yield decoded audio in ~250 ms pieces

    @classmethod
    def from_env(cls) -> "JaisConfig":
        """Read every ``JAIS_*`` / ``PIPECAT_*`` environment variable."""
        asr = AsrConfig(
            repo=os.getenv("JAIS_ASR_MODEL", "auto"),
            dtype=os.getenv("JAIS_ASR_DTYPE", "auto"),
            device=os.getenv("JAIS_ASR_DEVICE", "cuda"),
            max_segment_secs=_get_float("JAIS_ASR_MAX_SEGMENT_SECS", 35.0),
            max_new_tokens=_get_int("JAIS_ASR_MAX_NEW_TOKENS", 256),
        )
        llm = LlmConfig(
            repo=os.getenv("JAIS_LLM_MODEL", LLM_REPO),
            dtype=os.getenv("JAIS_LLM_DTYPE", "fp16"),
            device=os.getenv("JAIS_LLM_DEVICE", "cuda"),
            system_prompt=os.getenv("JAIS_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT,
            intro_prompt=os.getenv("JAIS_INTRO_PROMPT") or DEFAULT_INTRO_PROMPT,
            reasoning_effort=os.getenv("JAIS_LLM_REASONING_EFFORT", "low"),
            temperature=_get_float("JAIS_LLM_TEMPERATURE", 0.2),
            top_p=_get_float("JAIS_LLM_TOP_P", 0.95),
            top_k=_get_int("JAIS_LLM_TOP_K", 20),
            repetition_penalty=_get_float("JAIS_LLM_REPETITION_PENALTY", 1.0),
            max_new_tokens=_get_int("JAIS_LLM_MAX_NEW_TOKENS", 200),
            max_context_messages=_get_int("JAIS_LLM_MAX_CONTEXT_MESSAGES", 16),
        )
        tts = TtsConfig(
            repo=os.getenv("JAIS_TTS_MODEL", TTS_REPO),
            codec_repo=os.getenv("JAIS_TTS_CODEC_MODEL", TTS_CODEC_REPO),
            device=os.getenv("JAIS_TTS_DEVICE", "cuda"),
            voice=os.getenv("JAIS_TTS_VOICE", "demo_female_1"),
            ref_wav_path=os.getenv("JAIS_TTS_REF_WAV") or None,
            ref_text=os.getenv("JAIS_TTS_REF_TEXT") or None,
            ref_trim_secs=_get_float("JAIS_TTS_REF_TRIM_SECS", 12.0),
            temperature=_get_float("JAIS_TTS_TEMPERATURE", 0.7),
            top_k=_get_int("JAIS_TTS_TOP_K", 40),
            top_p=_get_float("JAIS_TTS_TOP_P", 0.9),
            repetition_penalty=_get_float("JAIS_TTS_REPETITION_PENALTY", 1.1),
            max_new_tokens=_get_int("JAIS_TTS_MAX_NEW_TOKENS", 1500),
            min_new_tokens=_get_int("JAIS_TTS_MIN_NEW_TOKENS", 50),
        )
        return cls(
            asr=asr,
            llm=llm,
            tts=tts,
            sample_rate_in=_get_int("JAIS_SAMPLE_RATE_IN", 16000),
            sample_rate_out=_get_int("JAIS_SAMPLE_RATE_OUT", 24000),
            tts_chunk_secs=_get_float("JAIS_TTS_CHUNK_SECS", 0.25),
        )
