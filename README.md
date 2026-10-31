# Jais Voice Chatbot — Arabic real-time voice assistant 🇸🇦🎙️

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/911houssem/jais-voice-chatbot/blob/main/notebooks/jais_voice_chatbot_colab.ipynb)

**EN** | **عربي (بالعربية أدناه)**

A fully open, **self-hosted Arabic voice chatbot** built on [pipecat](https://github.com/pipecat-ai/pipecat) with Saudi-made open models:

| Stage | Model | Role |
|---|---|---|
| STT | **Jais-ASR** (`IFM/Jais-ASR` / AWQ quantized) | Arabic speech → text |
| LLM | **Jais-3-0.9B** (`inception42/Jais-3-0.9B`) | Arabic chat |
| TTS | **Audar-TTS-V1-Flash** + NeuCodec | Text → expressive Arabic voice |

Everything runs locally on one GPU (a Colab **T4 16 GB** is enough), the browser UI comes from the pipecat runner (`pipecat-ai-prebuilt`), and the whole loop is: ميك → نص → رد عربي → صوت.

---

## ⭐ Features

- **One-click Colab**: `notebooks/jais_voice_chatbot_colab.ipynb` — install, load, serve, and expose a WebRTC URL on a free T4.
- **Real-time cascade pipeline**: Silero VAD → segmented STT → streaming LLM → streaming TTS.
- **Environment-driven config**: every knob (models, dtypes, sampling, voices) via `JAIS_*` env vars — see `.env.example`.
- **Smart GPU fit**: `JAIS_ASR_MODEL=auto` picks the fp16 checkpoint on ≥ 20 GB VRAM and the W4A16-AWQ quantized checkpoint otherwise; bf16 requests fall back to fp16 automatically on T4-class GPUs.
- **Voice cloning bootstrap**: the built-in demo voices ship without transcripts, so the reference transcript is bootstrapped by transcribing the reference clip with Jais-ASR itself. Custom voices: set `JAIS_TTS_REF_WAV` + `JAIS_TTS_REF_TEXT`.
- **Latency-minded defaults**: Jais-3 runs at `reasoning_effort="low"`, T = 0.2, ≤ 200 output tokens; TTS emits ~250 ms audio chunks.

## 🚀 Quickstart (local, CUDA GPU)

```bash
git clone <this-repo> && cd jais-voice-chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt torch               # torch: pick your CUDA wheel
pip install --no-deps neucodec==0.0.6               # TTS codec (torchtune is stubbed, see _vendor)
cp .env.example .env                                # optional tweaks

python app.py -t webrtc
# open http://localhost:8080/client and start talking Arabic 🎙️
```

## ☁️ Google Colab

Open `notebooks/jais_voice_chatbot_colab.ipynb` and run cells top-to-bottom; it:

1. checks the GPU / VRAM and adapts defaults for T4,
2. clones this repo and installs pinned deps (incl. `compressed-tensors`),
3. downloads all three models to the local HF cache,
4. starts `python app.py -t webrtc --host 0.0.0.0`,
5. exposes the runner port and prints the **browser URL** to click and talk.

## 🧠 Architecture

```
Browser (WebRTC, pipecat prebuilt UI)
        │ opus mic frames                 ▲ opus speaker frames
 transport.input()                        transport.output()
        │                                        ▲
 Silero VAD segments          🎧 TTS: Audar-TTS-V1-Flash
        ▼ (16 kHz, raw PCM)      (24 kHz, NeuCodec decode,
 🔤 STT: Jais-ASR                 0.25 s chunks)
 (fps16/AWQ, LLM-style ASR)                   ▲
        │ TranscriptionFrame(ar)              │ LLMTextFrame chunks
 user context aggregator ─────────────▶ 🤖 LLM: Jais-3-0.9B
                                          (reasoning_effort=low, streamer)
```

- `jais_voice/config.py` — every `JAIS_*` setting in one dataclass tree.
- `jais_voice/hub.py` — lazy, lock-guarded loading; official Audar-TTS inference flow (reference encode once → prompt → speech tokens → codec decode).
- `jais_voice/asr_service.py`, `llm_service.py`, `tts_service.py` — thin pipecat <q>`SegmentedSTTService` / `LLMService` / `TTSService</q> wrappers that only map frames to/from the models.
- `app.py` — runner-entry `bot()`; new pipecat 1.15 wiring: `PipelineWorker` + `WorkerRunner`, VAD in `LLMUserAggregatorParams`, RTVI auto-wired.

## 🌡️ Latency on a T4 (rough)

- STT (6B AWQ): ~3–8 s per short utterance — the heaviest stage.
- LLM first token: ~1–2 s; streaming into TTS hides the rest.
- TTS per chunk: ~0.5–1 s; audio starts playing when the first sentence streams in.
- Target end-to-end ≈ 5–11 s on T4, ~2–4 s on a 24 GB RTX (fp16 ASR + faster clocks).

## 🔧 Environment variables

Complete list with defaults lives in `.env.example`. Highlights:

| Var | Default | Notes |
|---|---|---|
| `JAIS_ASR_MODEL` | `auto` | or an HF repo id |
| `JAIS_ASR_DTYPE` | `auto` | `fp16`/`bf16`/`fp32`/`auto` |
| `JAIS_LLM_MODEL` | `inception42/Jais-3-0.9B` | |
| `JAIS_LLM_REASONING_EFFORT` | `low` | `high` = smarter, much slower |
| `JAIS_LLM_TEMPERATURE` | `0.2` | card's non-thinking default |
| `JAIS_TTS_VOICE` | `demo_female_1` | or `demo_male_1..3`, `demo_female_1..3`, `custom` |
| `JAIS_TTS_REF_WAV` / `JAIS_TTS_REF_TEXT` | – | required for `voice=custom` |

## ⚖️ Licenses

- **Code**: MIT (see `LICENSE`).
- **Models**: Jais-ASR / Jais-3 are Apache-2.0. Audar-TTS-V1-Flash ships under the **AudarAI Open v1.0** license — check its model card before production use.

---

# جايس شات صوتي — مساعد عربي في الوقت الحقيقي 🇸🇦🎙️

مساعد صوتي عربي مفتوح المصدر يعمل بالكامل على كرت الشاشة عندك (يكفي T4 في كولاب)، مبنٍ فوق إطار العمل **Pipecat** ويستطيع:

| المرحلة | النموذج | الوظيفة |
|---|---|---|
| تحويل الكلام لنص | **Jais-ASR** | عربي صوت → نص |
| الذكاء الحواري | **Jais-3-0.9B** | محادثة عربية |
| تحويل النص لكلام | **Audar-TTS-V1-Flash** + NeuCodec | نص → صوت عربي معبّر |

## التشغيل على كولاب
افتح الدفتر `notebooks/jais_voice_chatbot_colab.ipynb` وشغّل الخلايا بالترتيب؛ آخر خلية تعطيك رابط المتصفح للتحدث مباشرة.

## التشغيل محلياً
```bash
pip install -r requirements.txt
pip install --no-deps neucodec==0.0.6
python app.py -t webrtc
# ثم افتح http://localhost:8080/client وتكلم بالعربية 🎙️
```
> لماذا `--no-deps`؟ حزمة `neucodec` تعتمد على `torchtune`/`torchao` — مكدس ثقيل
> متشابك الإصدارات يكسر الحل عند pip. الغالب الوحيد الذي تستورده هو واجهة
> `RotaryPositionalEmbeddings`، وهو مسوّل عنه (BSD-3 كما في الأصل) في
> `jais_voice/_vendor/torchtune`.

## ملاحظات
- إعدادات كل شيء عبر متغيرات البيئة (`.env.example`)؛ الافتراضات مضبوطة لكرت 16GB مثل T4.
- ترخيص الأكواد MIT، أما نماذج جايس فلها Apache-2.0 ونموذج Audar-TTS له ترخيص AudarAI الخاص — راجعه قبل الاستخدام التجاري.
