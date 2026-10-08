from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wyoming-qwen3-tts")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

SERVICE_NAME = "wyoming-qwen3-tts"
