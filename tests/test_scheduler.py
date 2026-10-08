import asyncio
import threading
import time
from collections.abc import Iterator

from wyoming_qwen3_tts.scheduler import GpuWorker, Request


def fake_segment(log: list[str], label: str, chunks: int = 3, delay: float = 0.01) -> Iterator[str]:
    """Stands in for one sentence of GPU generation."""
    log.append(label)
    for i in range(chunks):
        time.sleep(delay)
        yield f"{label}:{i}"


async def consume(worker: GpuWorker, request: Request, log: list[str], labels: list[str]) -> list[str]:
    """Submit segments one after another, like the Wyoming handler does."""
    out: list[str] = []
    for label in labels:
        out.extend([item async for item in worker.stream(request, lambda label=label: fake_segment(log, label))])
    return out


async def test_stream_yields_items_in_order() -> None:
    worker = GpuWorker()
    log: list[str] = []
    out = await consume(worker, Request(), log, ["a1", "a2"])
    assert out == ["a1:0", "a1:1", "a1:2", "a2:0", "a2:1", "a2:2"]
    worker.stop()


async def test_new_request_first_segment_jumps_ahead_and_round_robin() -> None:
    worker = GpuWorker()
    log: list[str] = []
    gate = threading.Event()

    def blocker() -> Iterator[str]:
        log.append("blocker")
        gate.wait(5)
        yield "x"

    # Occupy the worker so the following jobs queue up
    blocking = asyncio.create_task(_drain(worker.stream(Request(name="blocker"), blocker)))
    await asyncio.sleep(0.05)

    started = Request(name="a")
    started.started = True  # request A already produced audio
    a = asyncio.create_task(_drain(worker.stream(started, lambda: fake_segment(log, "a2"))))
    await asyncio.sleep(0.01)
    b = asyncio.create_task(_drain(worker.stream(Request(name="b"), lambda: fake_segment(log, "b1"))))
    await asyncio.sleep(0.01)

    gate.set()
    await asyncio.gather(blocking, a, b)
    assert log == ["blocker", "b1", "a2"]  # B's first segment before A's follow-up
    worker.stop()


async def test_two_requests_interleave_per_segment() -> None:
    worker = GpuWorker()
    log: list[str] = []
    ra, rb = Request(name="a"), Request(name="b")
    await asyncio.gather(
        consume(worker, ra, log, ["a1", "a2", "a3"]),
        consume(worker, rb, log, ["b1", "b2", "b3"]),
    )
    # Both get their first segment early; neither waits for the other's whole answer
    assert set(log[:2]) == {"a1", "b1"}
    assert log.index("b1") < log.index("a3")
    assert log.index("a1") < log.index("b3")
    worker.stop()


async def test_cancel_drops_pending_segment_and_stops_running_one() -> None:
    worker = GpuWorker()
    log: list[str] = []
    produced: list[int] = []

    def long_segment() -> Iterator[int]:
        log.append("long")
        for i in range(100):
            time.sleep(0.01)
            produced.append(i)
            yield i

    running = asyncio.create_task(_drain(worker.stream(Request(), long_segment)))
    await asyncio.sleep(0.05)
    pending = asyncio.create_task(_drain(worker.stream(Request(), lambda: fake_segment(log, "pending"))))
    await asyncio.sleep(0.01)

    pending.cancel()
    running.cancel()
    await asyncio.gather(running, pending, return_exceptions=True)
    await asyncio.sleep(0.1)

    assert "pending" not in log
    assert len(produced) < 100
    worker.stop()


async def test_call_runs_on_worker_thread() -> None:
    worker = GpuWorker(name="gpu-test")
    assert await worker.call(lambda: threading.current_thread().name) == "gpu-test"
    worker.stop()


async def test_errors_propagate() -> None:
    worker = GpuWorker()

    def broken() -> Iterator[str]:
        raise RuntimeError("cuda on fire")
        yield "unreachable"

    try:
        await _drain(worker.stream(Request(), broken))
    except RuntimeError as err:
        assert "cuda on fire" in str(err)
    else:
        raise AssertionError("expected RuntimeError")
    worker.stop()


async def _drain(stream: object) -> list[object]:
    return [item async for item in stream]  # type: ignore[attr-defined]
