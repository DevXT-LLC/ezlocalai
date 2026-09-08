"""Bounded sync-to-async audio bridge with explicit native cancellation."""

import asyncio
import concurrent.futures
import threading
import anyio


async def stream_native_audio(model, **kwargs):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue(maxsize=2)
    cancelled = threading.Event()

    def send(item):
        pending = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        while not cancelled.is_set():
            try:
                pending.result(timeout=0.1)
                return True
            except concurrent.futures.TimeoutError:
                continue
        pending.cancel()
        return False

    def produce():
        iterator = model.generate_voice_clone_stream(**kwargs)
        try:
            for chunk in iterator:
                if cancelled.is_set() or not send(("audio", chunk)):
                    return
            send(("done", None))
        except Exception as error:
            if not cancelled.is_set():
                send(("error", error))
        finally:
            iterator.close()

    task = asyncio.create_task(asyncio.to_thread(produce))
    complete = False
    try:
        while True:
            kind, item = await queue.get()
            if kind == "done":
                complete = True
                return
            if kind == "error":
                raise item
            yield item
    finally:
        cancelled.set()
        # Starlette disconnects cancel an AnyIO scope repeatedly, not just one
        # asyncio await. Finish process teardown before the GPU lease is freed.
        with anyio.CancelScope(shield=True):
            if not complete:
                await asyncio.to_thread(model.close)
            await asyncio.shield(task)
