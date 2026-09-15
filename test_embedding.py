import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import xllamacpp as xlc

from ezlocalai.Embedding import Embedding


class EmbeddingParamsTests(unittest.TestCase):
    def _embedding(self):
        embedding = Embedding.__new__(Embedding)
        embedding.context_length = 10000
        embedding.batch_size = 512
        embedding.ubatch_size = 512
        embedding.n_parallel = 1
        embedding.kv_cache_type = "f16"
        embedding.model_alias = "Qwen3-Embedding-0.6B"
        return embedding

    def test_full_context_is_passed_to_native_params(self):
        self.assertEqual(
            self._embedding()._build_params("model.gguf", 0, 0).n_ctx, 10000
        )

    def test_native_context_error_becomes_actionable_client_error(self):
        embedder = self._embedding()
        embedder.server = Mock()
        embedder.server.handle_embeddings.return_value = {
            "error": {
                "code": 400,
                "type": "exceed_context_size_error",
                "message": "request (11002 tokens) exceeds the available context size (10240 tokens)",
            }
        }
        with self.assertRaisesRegex(ValueError, "11002 tokens.*") as error:
            embedder.get_embeddings("input")
        self.assertIn("Split the input", str(error.exception))
        self.assertIn("10000", str(error.exception))

    def test_client_errors_are_not_confused_with_server_failures(self):
        for code, expected in (
            (400, ValueError),
            (422, ValueError),
            (500, RuntimeError),
        ):
            embedder = self._embedding()
            embedder.server = Mock()
            embedder.server.handle_embeddings.return_value = {
                "error": {"code": code, "message": "test error"}
            }
            with self.assertRaises(expected):
                embedder.get_embeddings("input")

    def test_successful_inputs_are_forwarded_without_truncation(self):
        embedder = self._embedding()
        embedder.server = Mock()
        response = {
            "data": [{"embedding": [0.5, 0.5]}],
            "usage": {"prompt_tokens": 8302},
        }
        embedder.server.handle_embeddings.return_value = response
        text = "test " * 8300
        self.assertIs(embedder.get_embeddings(text), response)
        embedder.server.handle_embeddings.assert_called_once_with(
            {"input": text, "model": embedder.model_alias}
        )

    def test_cpu_fallback_disables_all_cuda_compute_offload(self):
        params = self._embedding()._build_params("model.gguf", 0, 0)

        self.assertEqual(params.n_gpu_layers, 0)
        self.assertTrue(params.no_kv_offload)
        self.assertTrue(params.no_op_offload)
        self.assertEqual(
            params.flash_attn_type,
            xlc.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_DISABLED,
        )

    def test_gpu_embedding_keeps_accelerated_compute_enabled(self):
        params = self._embedding()._build_params("model.gguf", 0, 12)

        self.assertEqual(params.n_gpu_layers, 12)
        self.assertFalse(params.no_kv_offload)
        self.assertFalse(params.no_op_offload)
        self.assertEqual(
            params.flash_attn_type,
            xlc.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED,
        )


class EmbeddingEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_error_is_http_400_and_releases_embedder(self):
        # Exercise the actual endpoint body without app.py's model-loading imports.
        from fastapi import HTTPException

        tree = ast.parse(Path(__file__).with_name("app.py").read_text())
        endpoint = next(
            n
            for n in tree.body
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "embedding"
        )
        endpoint.decorator_list = []
        endpoint.args.defaults = [ast.Constant(None)]
        for arg in endpoint.args.args:
            arg.annotation = None
        embedder = Embedding.__new__(Embedding)
        embedder.context_length = 10000
        embedder.model_alias = "embedding"
        embedder.server = Mock()
        embedder.server.handle_embeddings.return_value = {
            "error": {
                "code": 400,
                "type": "exceed_context_size_error",
                "message": "input too long",
            }
        }
        pipe = SimpleNamespace(
            acquire_embedder=AsyncMock(return_value=embedder),
            release_embedder=AsyncMock(),
        )
        resource = Mock()
        pipes = SimpleNamespace(
            get_embedding_server_client=lambda: SimpleNamespace(is_configured=False),
            should_use_ezlocalai_fallback=lambda: (False, ""),
            get_fallback_client=lambda: None,
            ModelType=SimpleNamespace(EMBEDDING="embedding"),
            get_resource_manager=lambda: resource,
        )
        namespace = {
            "asyncio": asyncio,
            "HTTPException": HTTPException,
            "pipe": pipe,
            "getenv": lambda key: "true",
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[endpoint], type_ignores=[])),
                "app.py",
                "exec",
            ),
            namespace,
        )
        request = SimpleNamespace(
            input="input",
            model="embedding",
            dimensions=None,
            model_dump=lambda **kwargs: {},
        )
        with patch.dict("sys.modules", {"Pipes": pipes}):
            with self.assertRaises(HTTPException) as error:
                await namespace["embedding"](request)
        self.assertEqual(error.exception.status_code, 400)
        self.assertIn("Split the input", error.exception.detail)
        pipe.release_embedder.assert_awaited_once_with(embedder)
        resource.mark_model_in_use.assert_called_with("embedding", False)


if __name__ == "__main__":
    unittest.main()
