"""Single GPU worker thread with fair, sentence-level scheduling across concurrent requests.

faster-qwen3-tts replays one CUDA graph over a batch-1 static KV cache, so only one
generation can run at a time and a running generation cannot be paused for another.
To keep several rooms responsive, requests are interleaved per segment (sentence):

1. queued calls (model load, warmup, prompt building)
2. the first segment of a request that has not produced audio yet (fast first audio)
3. round-robin across requests by least recently served
"""

import asyncio
import concurrent.futures
import itertools
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, TypeVar

_LOGGER = logging.getLogger(__name__)

T = TypeVar("T")
_DONE = object()
_counter = itertools.count(1)


@dataclass(eq=False)
class Request:
    """One synthesis request (a Wyoming streaming session or a single Synthesize)."""

    name: str = ""
    started: bool = False
    last_served: int = 0


@dataclass(eq=False)
class _Job:
    request: Request | None  # None for calls
    work: Callable[[], Any]
    loop: asyncio.AbstractEventLoop | None = None
    queue: "asyncio.Queue[Any] | None" = None
    future: "concurrent.futures.Future[Any] | None" = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    seq: int = field(default_factory=lambda: next(_counter))

    def sort_key(self) -> tuple[int, int, int, int]:
        if self.request is None:
            return (0, 0, 0, self.seq)
        return (1, int(self.request.started), self.request.last_served, self.seq)


class GpuWorker:
    def __init__(self, name: str = "gpu-worker") -> None:
        self._cond = threading.Condition()
        self._jobs: list[_Job] = []
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    async def call(self, fn: Callable[[], T]) -> T:
        """Run fn on the worker thread ahead of any queued segment."""
        job = _Job(request=None, work=fn, future=concurrent.futures.Future())
        self._enqueue(job)
        assert job.future is not None
        result: T = await asyncio.wrap_future(job.future)
        return result

    async def stream(self, request: Request, work: Callable[[], Iterator[T]]) -> AsyncIterator[T]:
        """Run the generator returned by work() on the worker thread and yield its items."""
        loop = asyncio.get_running_loop()
        job = _Job(request=request, work=work, loop=loop, queue=asyncio.Queue())
        self._enqueue(job)
        assert job.queue is not None
        try:
            while True:
                item = await job.queue.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # Consumer finished, failed or was cancelled (e.g. client disconnect)
            job.cancelled.set()
            with self._cond:
                if job in self._jobs:
                    self._jobs.remove(job)

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    def _enqueue(self, job: _Job) -> None:
        with self._cond:
            self._jobs.append(job)
            self._cond.notify()

    def _next_job(self) -> _Job | None:
        with self._cond:
            while not self._jobs and not self._stopped:
                self._cond.wait()
            if self._stopped:
                return None
            job = min(self._jobs, key=_Job.sort_key)
            self._jobs.remove(job)
            if job.request is not None:
                job.request.started = True
                job.request.last_served = next(_counter)
            return job

    def _run(self) -> None:
        while (job := self._next_job()) is not None:
            if job.future is not None:
                self._run_call(job)
            else:
                self._run_stream(job)

    @staticmethod
    def _run_call(job: _Job) -> None:
        assert job.future is not None
        if not job.future.set_running_or_notify_cancel():
            return
        try:
            job.future.set_result(job.work())
        except BaseException as err:
            job.future.set_exception(err)

    @staticmethod
    def _run_stream(job: _Job) -> None:
        assert job.loop is not None and job.queue is not None
        loop, queue = job.loop, job.queue

        def put(item: Any) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:  # event loop closed
                job.cancelled.set()

        if job.cancelled.is_set():
            return
        gen = None
        try:
            gen = job.work()
            for item in gen:
                if job.cancelled.is_set():
                    _LOGGER.debug("Segment cancelled, stopping generation")
                    break
                put(item)
            put(_DONE)
        except BaseException as err:
            put(err)
        finally:
            close = getattr(gen, "close", None)
            if close is not None:
                close()
