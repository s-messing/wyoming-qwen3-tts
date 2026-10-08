from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from wyoming_qwen3_tts import SERVICE_NAME

from .audio import SUPPORTED_LANGUAGES

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QWEN3_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    uri: str = Field(default="tcp://0.0.0.0:10200", description="Server URI")
    assets: Path = Field(default=Path("./assets"), description="Assets directory path")
    log_level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")
    zeroconf: str | None = Field(
        default=SERVICE_NAME,
        description="Zeroconf service name (enables discovery if set)",
    )
    model: str = Field(default=DEFAULT_MODEL, description="Qwen3-TTS Base model (HF id or local path)")
    design_model: str = Field(default=DESIGN_MODEL, description="Qwen3-TTS VoiceDesign model for the design command")
    offline: bool = Field(default=False, description="Do not download models, use the local HF cache only")
    language: str = Field(default="de", description="Language used when the request does not specify one")
    temperature: float = Field(default=0.5, gt=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(default=50, ge=1, le=1000, description="Top-k sampling")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, description="Top-p nucleus sampling")
    repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0, description="Repetition penalty")
    chunk_size: int = Field(default=4, ge=1, le=48, description="Codec steps per streamed audio chunk (12.5 steps = 1s)")
    min_segment_chars: int = Field(
        default=20,
        ge=1,
        le=500,
        description="Minimum characters before synthesizing a segment",
    )
    seed: int | None = Field(
        default=42,
        description="Seed for reproducible synthesis: same text, same audio (None for random)",
    )

    @field_validator("seed", mode="before")
    @classmethod
    def _empty_seed(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("language")
    @classmethod
    def _supported_language(cls, value: str) -> str:
        if value not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{value}', use one of {sorted(SUPPORTED_LANGUAGES)}")
        return value
