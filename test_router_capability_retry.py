import pathlib
import sys
import unittest
from unittest.mock import AsyncMock, patch


ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Router import WorkerInfo, WorkerRegistry
import router_app


def _worker(worker_id: str, capability: str = "embedding") -> WorkerInfo:
    return WorkerInfo(
        worker_id=worker_id,
        label=worker_id,
        url=f"http://{worker_id}.local",
        capabilities=[capability],
        models=[],
        cap_slots={
            capability: {
                "capacity": 1,
                "in_flight": 0,
                "queued": 0,
                "available": 1,
            }
        },
    )


class RouterCapabilityRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_worker_500_fails_over(self):
        registry = WorkerRegistry(ttl_seconds=60)
        first = registry.register(_worker("first"))
        second = registry.register(_worker("second"))
        responses = [
            router_app.Response(content=b'{"error":"busy"}', status_code=500),
            router_app.Response(content=b'{"data":[]}', status_code=200),
        ]
        selected = []

        async def pick(_capability, _model, exclude=None):
            excluded = set(exclude or ())
            worker = first if first.worker_id not in excluded else second
            selected.append(worker.worker_id)
            return worker

        async def proxy(worker, _path, _payload, **kwargs):
            registry.release_in_flight(worker.worker_id, kwargs["reservation_id"])
            return responses.pop(0)

        with (
            patch("router_app.get_registry", return_value=registry),
            patch("router_app._pick", side_effect=pick),
            patch("router_app._proxy_json", side_effect=proxy),
        ):
            worker, response = await router_app._proxy_json_capability_with_retry(
                capability="embedding",
                path="/v1/embeddings",
                payload={"input": "hello"},
                model=None,
            )

        self.assertIs(worker, second)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(selected, ["first", "second"])
        self.assertEqual(first.total_errors, 1)

    async def test_single_worker_is_requeued_after_transient_failure(self):
        registry = WorkerRegistry(ttl_seconds=60)
        worker = registry.register(_worker("only"))
        responses = [
            router_app.Response(content=b'{"error":"saturated"}', status_code=429),
            router_app.Response(content=b'{"data":[]}', status_code=200),
        ]

        async def proxy(worker_arg, _path, _payload, **kwargs):
            registry.release_in_flight(worker_arg.worker_id, kwargs["reservation_id"])
            return responses.pop(0)

        with (
            patch("router_app.get_registry", return_value=registry),
            patch("router_app._pick", AsyncMock(return_value=worker)) as pick,
            patch("router_app._proxy_json", side_effect=proxy),
            patch("router_app.asyncio.sleep", AsyncMock()),
        ):
            selected, response = await router_app._proxy_json_capability_with_retry(
                capability="embedding",
                path="/v1/embeddings",
                payload={"input": "hello"},
                model=None,
            )

        self.assertIs(selected, worker)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(pick.await_count, 2)

    async def test_client_error_is_returned_without_retry(self):
        registry = WorkerRegistry(ttl_seconds=60)
        worker = registry.register(_worker("only"))
        response = router_app.Response(
            content=b'{"detail":"invalid input"}', status_code=400
        )

        async def proxy(worker_arg, _path, _payload, **kwargs):
            registry.release_in_flight(worker_arg.worker_id, kwargs["reservation_id"])
            return response

        with (
            patch("router_app.get_registry", return_value=registry),
            patch("router_app._pick", AsyncMock(return_value=worker)) as pick,
            patch("router_app._proxy_json", side_effect=proxy),
        ):
            selected, result = await router_app._proxy_json_capability_with_retry(
                capability="embedding",
                path="/v1/embeddings",
                payload={"input": ""},
                model=None,
            )

        self.assertIs(selected, worker)
        self.assertIs(result, response)
        self.assertEqual(pick.await_count, 1)

    async def test_multipart_transcription_reuses_upload_for_failover(self):
        registry = WorkerRegistry(ttl_seconds=60)
        first = registry.register(_worker("first", capability="stt"))
        second = registry.register(_worker("second", capability="stt"))
        responses = [
            router_app.Response(content=b'{"error":"model busy"}', status_code=503),
            router_app.Response(content=b'{"text":"hello"}', status_code=200),
        ]
        uploads = []

        async def pick(_capability, _model, exclude=None):
            return first if first.worker_id not in set(exclude or ()) else second

        async def proxy(worker, _path, **kwargs):
            uploads.append(kwargs["files"]["file"][1])
            registry.release_in_flight(worker.worker_id, kwargs["reservation_id"])
            return responses.pop(0)

        with (
            patch("router_app.get_registry", return_value=registry),
            patch("router_app._pick", side_effect=pick),
            patch("router_app._proxy_multipart", side_effect=proxy),
        ):
            worker, response = await router_app._proxy_multipart_capability_with_retry(
                capability="stt",
                path="/v1/audio/transcriptions",
                files={"file": ("voice.wav", b"audio-data", "audio/wav")},
                fields={"model": "whisper-1"},
                model="whisper-1",
            )

        self.assertIs(worker, second)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(uploads, [b"audio-data", b"audio-data"])

    async def test_dispatch_race_returns_to_queue_without_spending_retry(self):
        registry = WorkerRegistry(ttl_seconds=60)
        worker = registry.register(_worker("only"))
        worker.extra["llm_unload_dependent_capabilities"] = ["embedding"]
        response = router_app.Response(content=b'{"data":[]}', status_code=200)
        original_reserve = registry.try_reserve_in_flight
        reservations = [None]

        def reserve(*args, **kwargs):
            if reservations:
                return reservations.pop()
            return original_reserve(*args, **kwargs)

        async def proxy(worker_arg, _path, _payload, **kwargs):
            registry.release_in_flight(worker_arg.worker_id, kwargs["reservation_id"])
            return response

        with (
            patch("router_app.get_registry", return_value=registry),
            patch.object(registry, "try_reserve_in_flight", side_effect=reserve),
            patch("router_app._pick", AsyncMock(return_value=worker)) as pick,
            patch("router_app._proxy_json", side_effect=proxy) as proxy_mock,
            patch("router_app.asyncio.sleep", AsyncMock()),
        ):
            _, result = await router_app._proxy_json_capability_with_retry(
                capability="embedding",
                path="/v1/embeddings",
                payload={"input": "hello"},
                model=None,
            )

        self.assertIs(result, response)
        self.assertEqual(pick.await_count, 2)
        self.assertEqual(proxy_mock.await_count, 1)


if __name__ == "__main__":
    unittest.main()
