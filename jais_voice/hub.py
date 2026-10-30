"""Model hub: lazy, single-copy loading of STT/LLM/TTS models on GPU.

``ModelHub`` is a per-process singleton created once per bot connection in
``app.py``. Each model loads on first use (guarded by an ``asyncio.Lock``) so
heavy I/O and CUDA work happen off the event loop via ``asyncio.to_thread``.

All three models live on the same GPU; fp16 is the default dtype because the
target GPU (T4, compute capability 7.5) has no bf16 support.
"""

from __future__ import annotations

import asyncio
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from loguru import logger

from .config import ASR_REPO, ASR_REPO_AWQ, JaisConfig

# Built-in demo voices published on the Audar-TTS model card. Clips live at:
# https://huggingface.co/audarai/Audar-TTS-V1-Flash/resolve/main/samples/<id>_source.wav
DEMO_VOICE_IDS = (
    "demo_male_1",
    "demo_male_2",
    "demo_male_3",
    "demo_female_1",
    "demo_female_2",
    "demo_female_3",
)
_DEMO_CLIP_URL = (
    "https://huggingface.co/audarai/Audar-TTS-V1-Flash/resolve/main/samples"
    "/{voice}_source.wav"
)
_CACHE_DIR = Path(os.path.expanduser("~/.cache/jais-voice"))

# Jais-ASR emits this marker (per its model card) when it detects no speech.
_NO_TRANSCRIPT_MARKER = "[IFM-NoTranscript]"

# Speech tokens emitted by the Audar-TTS decoder.
_SPEECH_RE = re.compile(r"<\|speech_(\d+)\|>")


@dataclass
class TranscriptionResult:
    """Normalized STT result across card quirks."""

    text: str
    language: str | None = None
    refused: bool = False


class ModelHub:
    """Owns the Jais-ASR / Jais-3 / Audar-TTS models for one process."""

    def __init__(self, config: JaisConfig):
        self.config = config
        self._asr = None
        self._llm = None
        self._tts = None
        self._ref_codes = None
        self._ref_text = None
        self._warmed_up = False
        self._locks = {
            "asr": asyncio.Lock(),
            "llm": asyncio.Lock(),
            "tts": asyncio.Lock(),
        }
        self._ref_lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Device / dtype helpers
    # ------------------------------------------------------------------ #

    def available_vram_gb(self) -> float | None:
        """Free VRAM in GiB, or ``None`` when CUDA is unavailable."""
        try:
            free, _total = torch.cuda.mem_get_info()
            return free / (1024**3)
        except Exception:
            return None

    def _resolve_dtype(self, requested: str, *, warn_only: bool = False) -> torch.dtype:
        """Resolve a dtype string, guarding bf16 on GPUs without bf16 support."""
        if requested in ("", "auto"):
            requested = "fp16"  # conservative default for T4
        name = requested.lower()
        if name == "bf16" and torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major < 8:  # e.g. T4/RTX 20xx: bf16 unsupported in cuBLAS ops
                logger.warning(
                    "bf16 requested but current GPU (capability {}) lacks bf16; "
                    "falling back to fp16.",
                    major,
                )
                name = "fp16"
        if name not in ("fp16", "bf16", "float16", "float32", "fp32"):
            raise ValueError(f"Unsupported dtype: {requested!r}")
        return {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }[name]

    def _resolve_asr_repo(self) -> str:
        """Auto-select the ASR checkpoint by available VRAM."""
        repo = self.config.asr.repo
        if repo != "auto":
            return repo
        vram = self.available_vram_gb()
        if vram is None:
            return ASR_REPO_AWQ  # no CUDA info -> prefer the small model
        if vram >= 20.0:
            logger.info("ASR: auto -> %s (fp16 needs ~11.4 GB; %.1f GB free)",
                        ASR_REPO, vram)
            return ASR_REPO
        logger.info("ASR: auto -> %s (need ~3.4 GB; %.1f GB free)",
                    ASR_REPO_AWQ, vram)
        return ASR_REPO_AWQ

    # ------------------------------------------------------------------ #
    # Jais-ASR (STT)
    # ------------------------------------------------------------------ #

    def _asr_dtype_str(self, repo: str) -> str:
        configured = self.config.asr.dtype
        if configured != "auto":
            return configured
        # Quantized checkpoints manage their own precision; fp16 otherwise.
        return "auto" if "AWQ" in repo or "INT" in repo else "fp16"

    async def load_asr(self):
        """Load ``IFM/Jais-ASR`` (or the AWQ variant) + processor."""
        async with self._locks["asr"]:
            if self._asr is not None:
                return self._asr
            repo = self._resolve_asr_repo()
            dtype_str = self._asr_dtype_str(repo)
            dtype = self._resolve_dtype(dtype_str) if dtype_str != "auto" else None

            def _load():
                from transformers import AutoModelForCausalLM, AutoProcessor

                logger.info("Loading Jais-ASR model %s (dtype=%s)...", repo, dtype_str)
                processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
                if getattr(processor, "tokenizer", None) is not None:
                    processor.tokenizer.padding_side = "left"
                model_kwargs = {
                    "trust_remote_code": True,
                    **({"dtype": dtype} if dtype is not None else {}),
                }
                try:
                    # Quantized checkpoints must stay compressed, otherwise
                    # transformers decompresses them (~11 GB -> OOM on 16 GB).
                    if dtype is None:
                        model = AutoModelForCausalLM.from_pretrained(
                            repo, run_compressed=True, **model_kwargs
                        )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(repo, **model_kwargs)
                except (TypeError, ValueError):
                    logger.warning("run_compressed=True rejected by transformers; retrying")
                    model = AutoModelForCausalLM.from_pretrained(repo, **model_kwargs)
                model = model.to(self.config.asr.device)
                model.eval()
                return model, processor

            self._asr = await asyncio.to_thread(_load)
            logger.info("Jais-ASR loaded: %s", repo)
            return self._asr

    async def transcribe(
        self, waveform: np.ndarray, sample_rate: int
    ) -> TranscriptionResult | None:
        """Transcribe one VAD speech segment.

        ``waveform`` is a mono float32 array in [-1, 1]; the processor
        resamples to 16 kHz internally (per the model card quickstart).
        """
        model, processor = await self.load_asr()
        async with self._locks["asr"]:

            def _run() -> TranscriptionResult | None:
                tokenizer = getattr(processor, "tokenizer", None) or processor
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": [{"type": "audio"}]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = processor(
                    text=[prompt],
                    audio=[waveform],
                    sampling_rate=[sample_rate],
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
                output = model.generate(**inputs, do_sample=False, max_new_tokens=256)
                result = processor.post_process_transcription(
                    output, input_lengths=inputs["input_ids"].shape[1]
                )
                if isinstance(result, dict):
                    text = result.get("text") or ""
                    refused = bool(result.get("refused", False))
                    language = result.get("language")
                else:  # attribute-style result object
                    text = getattr(result, "text", "") or ""
                    refused = bool(getattr(result, "refused", False))
                    language = getattr(result, "language", None)
                text = (text or "").strip()
                if not text or text == _NO_TRANSCRIPT_MARKER or refused:
                    return None
                return TranscriptionResult(text=text, language=language, refused=refused)

            return await asyncio.to_thread(_run)

    # ------------------------------------------------------------------ #
    # Jais-3 (LLM)
    # ------------------------------------------------------------------ #

    async def load_llm(self):
        """Load ``Jais-3`` + tokenizer."""
        async with self._locks["llm"]:
            if self._llm is not None:
                return self._llm
            cfg = self.config.llm
            dtype = self._resolve_dtype(cfg.dtype)

            def _load():
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info("Loading Jais-3 model %s (dtype=%s)...", cfg.repo, cfg.dtype)
                tokenizer = AutoTokenizer.from_pretrained(cfg.repo, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.repo,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ).to(cfg.device)
                model.eval()
                return model, tokenizer

            self._llm = await asyncio.to_thread(_load)
            logger.info("Jais-3 loaded: %s", cfg.repo)
            return self._llm

    # ------------------------------------------------------------------ #
    # Audar-TTS + NeuCodec
    # ------------------------------------------------------------------ #

    async def load_tts(self):
        """Load the Audar-TTS backbone and its NeuCodec decoder."""
        async with self._locks["tts"]:
            if self._tts is not None:
                return self._tts
            cfg = self.config.tts
            dtype = self._resolve_dtype("fp16", warn_only=True)

            def _load():
                from neucodec import NeuCodec
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info("Loading Audar-TTS model %s...", cfg.repo)
                is_repo = not (cfg.repo.startswith(("/", ".")) or os.path.isdir(cfg.repo))
                kwargs = {"subfolder": "transformers"} if is_repo else {}
                tok = AutoTokenizer.from_pretrained(cfg.repo, **kwargs)
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.repo, dtype=dtype, **kwargs
                ).to(cfg.device)
                model.eval()
                logger.info("Loading codec %s...", cfg.codec_repo)
                codec = NeuCodec.from_pretrained(cfg.codec_repo).eval().to(cfg.device)
                return model, tok, codec

            self._tts = await asyncio.to_thread(_load)
            logger.info("Audar-TTS loaded: %s + %s", cfg.repo, cfg.codec_repo)
            return self._tts

    def synthesize_codes(self, model, tokenizer, codec, text: str) -> np.ndarray:
        """Run the official Audar-TTS inference flow; returns float32 @ 24 kHz.

        Mirrors ``audar_tts/audar_tts_infer.py`` from the official repository.
        Called in a worker thread by the TTS service.
        """
        cfg = self.config.tts
        prompt = (
            "user: Convert the text to speech:"
            f"<|REF_TEXT_START|>{self._ref_text}<|REF_TEXT_END|>"
            f"<|REF_SPEECH_START|>{self._codes_to_str()}<|REF_SPEECH_END|>"
            f"<|TARGET_TEXT_START|>{text}<|TARGET_TEXT_END|>"
            "\nassistant:<|TARGET_CODES_START|>"
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(
            model.device
        )
        eos = tokenizer.convert_tokens_to_ids("<|TARGET_CODES_END|>")
        with torch.inference_mode():
            out = model.generate(
                ids,
                max_new_tokens=cfg.max_new_tokens,
                min_new_tokens=cfg.min_new_tokens,
                eos_token_id=eos,
                pad_token_id=151643,
                do_sample=True,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                repetition_penalty=cfg.repetition_penalty,
            )
            text_out = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
        codes = [int(x) for x in _SPEECH_RE.findall(text_out)]
        if not codes:
            raise RuntimeError(
                "TTS emitted no speech tokens (model output was:\n%r)" % text_out[:200]
            )
        with torch.inference_mode():
            wav = codec.decode_code(
                torch.tensor(codes)[None, None, :].to(next(codec.parameters()).device)
            )
        return wav.cpu().numpy()[0, 0, :]

    def _codes_to_str(self) -> str:
        return "".join(f"<|speech_{c}|>" for c in self._ref_codes)

    # ------------------------------------------------------------------ #
    # TTS voice reference (demo registry or custom file)
    # ------------------------------------------------------------------ #

    def _load_ref_wav(self, path: Path) -> np.ndarray:
        """Load a reference clip as 16 kHz mono float32, trimmed."""
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = soxr.resample(wav, sr, 16000)
        max_samples = int(self.config.tts.ref_trim_secs * 16000)
        if wav.shape[0] > max_samples:
            wav = wav[:max_samples]
        return np.ascontiguousarray(wav, dtype=np.float32)

    def _resolve_ref_path(self) -> Path:
        """Demo voice -> cached HF sample clip; custom voice -> given file."""
        cfg = self.config.tts
        if cfg.voice == "custom":
            if not cfg.ref_wav_path:
                raise ValueError(
                    'JAIS_TTS_REF_WAV (and JAIS_TTS_REF_TEXT) must be set when '
                    'JAIS_TTS_VOICE="custom".'
                )
            path = Path(cfg.ref_wav_path)
            if not path.is_file():
                raise FileNotFoundError(f"Reference WAV not found: {path}")
            return path
        if cfg.voice not in DEMO_VOICE_IDS:
            raise ValueError(
                f"Unknown voice {cfg.voice!r}. Pick one of {DEMO_VOICE_IDS} "
                'or set JAIS_TTS_VOICE="custom".'
            )
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _CACHE_DIR / f"{cfg.voice}_source.wav"
        if not path.is_file():
            url = _DEMO_CLIP_URL.format(voice=cfg.voice)
            logger.info("Downloading demo voice clip: %s", url)
            urllib.request.urlretrieve(url, str(path))
        return path

    async def _ensure_ref_codes(self):
        """Encode the reference clip once; bootstrap its transcript via ASR.

        The official voice registry ships clips without transcripts, so we
        transcribe the reference clip with the loaded Jais-ASR model — through
        the same locked ``transcribe()`` path used for user speech.
        """
        if self._ref_codes is not None:
            return
        async with self._ref_lock:
            if self._ref_codes is not None:  # raced during download/encode
                return
            cfg = self.config.tts

            def _prepare_wav():
                return self._load_ref_wav(self._resolve_ref_path())

            wav = await asyncio.to_thread(_prepare_wav)
            # Transcript: env override wins, otherwise ASR bootstrap.
            if cfg.ref_text:
                text = cfg.ref_text.strip()
            else:
                logger.info("Bootstrapping reference transcript with Jais-ASR...")
                result = await self.transcribe(wav, 16000)
                if result is None:
                    raise RuntimeError(
                        "Could not transcribe the reference clip. Provide the "
                        "transcript explicitly via JAIS_TTS_REF_TEXT (and the clip "
                        "via JAIS_TTS_REF_WAV)."
                    )
                text = result.text
            codec = (await self.load_tts())[2]
            self._ref_codes = await asyncio.to_thread(self._encode_reference, codec, wav)
            self._ref_text = text
            logger.info(
                "TTS reference ready (voice=%s, %d codes, transcript: %r)",
                cfg.voice,
                len(self._ref_codes),
                self._ref_text[:60],
            )

    @staticmethod
    def _encode_reference(codec, wav_16k: np.ndarray) -> list[int]:
        wav = torch.from_numpy(wav_16k.astype(np.float32))[None, None, :]
        device = next(codec.parameters()).device
        return codec.encode_code(wav.to(device)).squeeze().tolist()

    async def prepare_tts_voice(self):
        """Public hook used by the TTS service before first synthesis."""
        await self.tts_ready()

    async def warmup(self):
        """Preload every model + voice reference (background in app.py)."""
        await self.load_tts()
        await self.load_llm()
        await self.load_asr()
        await self._ensure_ref_codes()
        self._warmed_up = True
        try:
            free, _total = torch.cuda.mem_get_info()
            logger.info("Warm-up complete; VRAM free %.1f GB", free / (1024**3))
        except Exception:
            pass

    async def tts_ready(self):
        """Load TTS models + reference, keeping things ordered."""
        await self.load_tts()
        await self._ensure_ref_codes()
