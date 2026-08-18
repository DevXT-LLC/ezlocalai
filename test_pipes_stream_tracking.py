import os
import sys
import threading
import types
import unittest
from unittest import mock


sys.modules.setdefault("xllamacpp", types.SimpleNamespace())

from Pipes import Pipes


class PipesStreamTrackingTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _pipe(response):
        pipe = Pipes.__new__(Pipes)
        pipe._llm_temporarily_unavailable = False
        pipe.should_use_fallback = mock.Mock(return_value=(False, ""))
        pipe._cancel_context_reset = mock.Mock()
        pipe._resolve_slot_model = mock.Mock(return_value="test/model")
        pipe._inference_count = 0
        pipe._model_inference_counts = {}
        pipe._inference_count_lock = threading.Lock()
        pipe.llm = types.SimpleNamespace(is_vision=False)
        pipe.resource_manager = mock.Mock()
        pipe.current_context = None
        pipe._optimal_context = 4096
        pipe._schedule_context_reset = mock.Mock()
        pipe._get_response_internal = mock.AsyncMock(return_value=(response, None))
        return pipe

    async def test_stream_remains_in_flight_until_consumed(self):
        pipe = self._pipe(iter([{"token": "one"}, {"token": "two"}]))
        pipe.available_models = ["test/model"]
        pipe.persistent_llms = {"test/model": types.SimpleNamespace(n_parallel=1)}
        pipe.model_configs = {"test/model": {"n_parallel": 1}}
        pipe._resolve_source_model = lambda model: model
        pipe._is_vision_model = lambda model: False

        response, _ = await pipe.get_response({"stream": True})

        self.assertEqual(pipe._inference_count, 1)
        self.assertEqual(pipe._model_inference_counts, {"test/model": 1})
        pipe.resource_manager.mark_model_in_use.assert_called_once_with(mock.ANY, True)
        with (
            mock.patch.dict(
                os.environ,
                {
                    "TTS_ENABLED": "false",
                    "STT_ENABLED": "false",
                    "EMBEDDING_ENABLED": "false",
                },
            ),
            mock.patch("Pipes.is_image_enabled", return_value=False),
            mock.patch("Pipes.is_video_enabled", return_value=False),
            mock.patch("Pipes.is_music_enabled", return_value=False),
        ):
            active_snapshot = pipe.get_slot_capacity_snapshot()
        self.assertEqual(active_snapshot["cap_slots"]["text"]["in_flight"], 1)
        self.assertEqual(active_snapshot["model_slots"]["test/model"]["in_flight"], 1)

        self.assertEqual(list(response), [{"token": "one"}, {"token": "two"}])
        self.assertEqual(pipe._inference_count, 0)
        self.assertEqual(pipe._model_inference_counts, {})
        self.assertEqual(pipe.resource_manager.mark_model_in_use.call_count, 2)
        self.assertEqual(
            pipe.resource_manager.mark_model_in_use.call_args_list[-1].args[1], False
        )

    async def test_closing_unstarted_stream_releases_inference(self):
        pipe = self._pipe(iter([{"token": "unused"}]))

        response, _ = await pipe.get_response({"stream": True})
        self.assertEqual(pipe._inference_count, 1)

        response.close()
        response.close()

        self.assertEqual(pipe._inference_count, 0)
        self.assertEqual(pipe.resource_manager.mark_model_in_use.call_count, 2)

    async def test_stream_error_releases_inference(self):
        def failing_stream():
            yield {"token": "first"}
            raise RuntimeError("generation failed")

        pipe = self._pipe(failing_stream())
        response, _ = await pipe.get_response({"stream": True})

        self.assertEqual(next(response), {"token": "first"})
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            next(response)

        self.assertEqual(pipe._inference_count, 0)
        self.assertEqual(pipe._model_inference_counts, {})

    async def test_non_streaming_response_finishes_immediately(self):
        payload = {"choices": [{"message": {"content": "done"}}]}
        pipe = self._pipe(payload)

        response, _ = await pipe.get_response({"stream": False})

        self.assertIs(response, payload)
        self.assertEqual(pipe._inference_count, 0)
        self.assertEqual(pipe._model_inference_counts, {})
        self.assertEqual(pipe.resource_manager.mark_model_in_use.call_count, 2)


if __name__ == "__main__":
    unittest.main()
