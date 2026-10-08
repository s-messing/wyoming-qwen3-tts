import logging
import time
import zlib
from collections.abc import AsyncGenerator, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
from wyoming.audio import AudioChunk

from .audio import CHANNELS, QWEN_LANGUAGES, SAMPLE_RATE, SAMPLE_WIDTH, to_pcm
from .de_normalize import normalize
from .scheduler import GpuWorker, Request
from .textclean import clean_text
from .voice import Voice

if TYPE_CHECKING:
    from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

CODEC_HZ = 12.5  # Qwen3-TTS 12Hz tokenizer: codec frames per second of audio


def token_budget(text: str) -> int:
    """Cap generation length so a derailed generation stops early.

    German speech runs ~13-15 characters per second; allow roughly double plus slack.
    """
    seconds = 3.0 + 0.15 * len(text)
    return int(seconds * CODEC_HZ)


def prepare_text(text: str, language: str) -> str:
    return normalize(clean_text(text), language)


class Qwen3Engine:
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cuda",
        temperature: float = 0.5,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        chunk_size: int = 4,
        seed: int | None = 42,
    ) -> None:
        self.model_id = model_id
        self.model_tag = model_id.rstrip("/").split("/")[-1]
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.chunk_size = chunk_size
        self.seed = seed

        self.model: Any = None
        self.worker = GpuWorker()
        self._prompts: dict[Voice, Any] = {}

    async def load(self) -> None:
        _LOGGER.info("Loading %s (device=%s)", self.model_id, self.device)
        start = time.perf_counter()
        await self.worker.call(self._load_sync)
        _LOGGER.info("Model loaded and warmed up in %.1fs", time.perf_counter() - start)

    def _load_sync(self) -> None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        if self.device == "cuda" and not torch.cuda.is_available():
            _LOGGER.warning("CUDA is not available. Have you passed a GPU into the container?")
        self.model = FasterQwen3TTS.from_pretrained(self.model_id, device=self.device)
        self.model.warmup()

    async def get_prompt(self, voice: Voice) -> Any:
        prompt = self._prompts.get(voice)
        if prompt is None:
            prompt = await self.worker.call(lambda: self._load_prompt_sync(voice))
            self._prompts[voice] = prompt
        return prompt

    def _load_prompt_sync(self, voice: Voice) -> Any:
        """Load the cached voice clone prompt, or build it from <name>.wav + <name>.txt and cache it."""
        import torch

        if self.model is None:
            raise RuntimeError("Model not loaded")
        cache = voice.wav.with_name(f"{voice.name}.{self.model_tag}.pt")
        source_mtime = max(voice.wav.stat().st_mtime, voice.txt.stat().st_mtime)
        if cache.exists() and cache.stat().st_mtime >= source_mtime:
            _LOGGER.info("Loading voice prompt %s", cache.name)
            return torch.load(cache, weights_only=False, map_location=self.device)

        _LOGGER.info("Building voice prompt for '%s' -> %s", voice.name, cache.name)
        prompt = self.model.model.create_voice_clone_prompt(ref_audio=str(voice.wav), ref_text=voice.ref_text)
        try:
            torch.save(prompt, cache)
        except OSError as err:
            _LOGGER.warning("Could not cache voice prompt %s: %s", cache, err)
        return prompt

    def _generate(self, text: str, prompt: Any, language: str) -> Iterator[np.ndarray]:
        """Runs on the GPU worker thread."""
        import torch

        if self.seed is not None:
            # Same text -> same audio; recurring assistant phrases always sound identical
            torch.manual_seed(zlib.crc32(f"{self.seed}:{text}".encode()))
        for chunk, sr, _timing in self.model.generate_voice_clone_streaming(
            text=text,
            language=QWEN_LANGUAGES[language],
            voice_clone_prompt=prompt,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            chunk_size=self.chunk_size,
            max_new_tokens=token_budget(text),
        ):
            if sr != SAMPLE_RATE:
                raise RuntimeError(f"Unexpected sample rate {sr}, expected {SAMPLE_RATE}")
            yield chunk

    async def synthesize_stream(
        self,
        request: Request,
        text: str,
        voice: Voice,
        language: str,
    ) -> AsyncGenerator[bytes, None]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        prompt = await self.get_prompt(voice)
        spoken = prepare_text(text, language)
        if not spoken:
            return
        _LOGGER.debug("Synthesizing: %r -> %r (lang=%s, voice=%s)", text, spoken, language, voice.name)
        async for chunk in self.worker.stream(request, lambda: self._generate(spoken, prompt, language)):
            yield to_pcm(chunk)

    async def stream_to_handler(
        self,
        handler: "AsyncEventHandler",
        request: Request,
        text: str,
        voice: Voice,
        language: str,
    ) -> float | None:
        first_audio_time: float | None = None
        start = time.perf_counter()

        async for chunk in self.synthesize_stream(request, text, voice, language):
            if first_audio_time is None:
                first_audio_time = time.perf_counter() - start
                _LOGGER.debug("First audio chunk: %.3fs", first_audio_time)
            await handler.write_event(AudioChunk(audio=chunk, rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS).event())

        return first_audio_time
