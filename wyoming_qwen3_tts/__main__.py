#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import logging
import os
import signal
from functools import partial
from pathlib import Path

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer, AsyncTcpServer

from . import SERVICE_NAME, __version__
from .audio import SUPPORTED_LANGUAGES
from .config import Settings
from .handler import Qwen3EventHandler
from .voice import Voice, scan_voices

_LOGGER = logging.getLogger(__name__)


def parse_args(defaults: Settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wyoming Qwen3-TTS server")
    parser.add_argument("--uri", default=defaults.uri)
    parser.add_argument("--assets", type=Path, default=defaults.assets)
    parser.add_argument(
        "--zeroconf",
        default=defaults.zeroconf,
        help="Zeroconf service name (enables discovery)",
    )
    parser.add_argument(
        "--log-level",
        default=defaults.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--model", default=defaults.model, help="Qwen3-TTS Base model (HF id or local path)")
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=defaults.offline)
    parser.add_argument(
        "--language",
        default=defaults.language,
        choices=sorted(SUPPORTED_LANGUAGES),
        help="Language used when the request does not specify one",
    )
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--top-p", type=float, default=defaults.top_p)
    parser.add_argument("--repetition-penalty", type=float, default=defaults.repetition_penalty)
    parser.add_argument("--chunk-size", type=int, default=defaults.chunk_size)
    parser.add_argument("--min-segment-chars", type=int, default=defaults.min_segment_chars)
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Seed for reproducible synthesis: same text, same audio (omit for random)",
    )
    parser.add_argument("--version", action="version", version=__version__)

    commands = parser.add_subparsers(dest="command", metavar="{serve,design,pick}")
    commands.add_parser("serve", help="Run the Wyoming server (default)")

    design = commands.add_parser("design", help="Generate voice candidates from a description")
    design.add_argument("--name", required=True, help="Voice name, shown in Home Assistant")
    design.add_argument("--description", required=True, help="Voice description, e.g. 'A calm male voice in his forties ...'")
    design.add_argument("--candidates", type=int, default=3)
    design.add_argument("--text", help="Reference sentence to speak (default: a German assistant greeting)")
    design.add_argument("--design-model", default=defaults.design_model)

    pick = commands.add_parser("pick", help="Turn a candidate into the voice <name>")
    pick.add_argument("--name", required=True)
    pick.add_argument("--candidate", type=int, required=True)
    return parser.parse_args()


def build_info(voices: list[Voice]) -> Info:
    languages = sorted(SUPPORTED_LANGUAGES)
    return Info(
        tts=[
            TtsProgram(
                name=SERVICE_NAME,
                description="Qwen3-TTS text-to-speech with designed and cloned voices",
                attribution=Attribution(name="Qwen", url="https://github.com/QwenLM/Qwen3-TTS"),
                installed=True,
                voices=[
                    TtsVoice(
                        name=voice.name,
                        description=f"Voice: {voice.name}",
                        version=None,
                        attribution=Attribution(name="", url=""),
                        installed=True,
                        languages=languages,
                    )
                    for voice in voices
                ],
                version=__version__,
                supports_synthesize_streaming=True,
            )
        ]
    )


def setup_environment(args: argparse.Namespace) -> Path:
    voices_path: Path = args.assets / "voices"
    voices_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.assets / "hf"))
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    return voices_path


async def serve(args: argparse.Namespace, voices_path: Path) -> None:
    from .engine import Qwen3Engine

    _LOGGER.info("Starting %s service", SERVICE_NAME)
    _LOGGER.info("Assets path: %s", args.assets)

    voices = scan_voices(voices_path)
    if not voices:
        _LOGGER.warning("No voices found in %s. Create one with the 'design' and 'pick' commands.", voices_path)

    engine = Qwen3Engine(
        model_id=args.model,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    await engine.load()
    for voice in voices:
        await engine.get_prompt(voice)
    _LOGGER.info(
        "Synthesis settings: lang=%s, temp=%.2f, top_k=%d, top_p=%.2f, rep_penalty=%.2f, chunk_size=%d, seed=%s",
        args.language,
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
        args.chunk_size,
        args.seed,
    )

    wyoming_info = build_info(voices)
    server = AsyncServer.from_uri(args.uri)

    if args.zeroconf:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// URI")

        from wyoming.zeroconf import HomeAssistantZeroconf

        zeroconf_host = None if server.host in ("0.0.0.0", "::") else server.host
        zeroconf = HomeAssistantZeroconf(
            name=args.zeroconf,
            port=server.port,
            host=zeroconf_host,
        )
        await zeroconf.register_server()
        _LOGGER.info(
            "Zeroconf discovery enabled: %s -> %s:%s",
            args.zeroconf,
            zeroconf.host,
            server.port,
        )

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def handle_signal() -> None:
        _LOGGER.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, handle_signal)

    _LOGGER.info("Server ready on %s", args.uri)

    try:
        server_task = asyncio.create_task(
            server.run(
                partial(
                    Qwen3EventHandler,
                    wyoming_info,
                    engine,
                    voices_path,
                    args.language,
                    args.min_segment_chars,
                )
            )
        )
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        _, pending = await asyncio.wait(
            [server_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    finally:
        engine.worker.stop()
        _LOGGER.info("Shutdown complete")


def run() -> None:
    args = parse_args(Settings())
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    voices_path = setup_environment(args)

    if args.command == "design":
        from .design import design

        paths = design(
            voices_path,
            name=args.name,
            description=args.description,
            model_id=args.design_model,
            language=args.language,
            candidates=args.candidates,
            text=args.text,
        )
        print("\nListen to the candidates, then pick one:")
        for i, path in enumerate(paths, start=1):
            print(f"  {i}: {path}")
        print(f"\n  wyoming-qwen3-tts pick --name {args.name} --candidate <n>")
    elif args.command == "pick":
        from .design import pick

        target = pick(voices_path, args.name, args.candidate)
        print(f"Voice '{args.name}' created: {target}. Restart the server to load it.")
    else:
        asyncio.run(serve(args, voices_path))


if __name__ == "__main__":
    run()
