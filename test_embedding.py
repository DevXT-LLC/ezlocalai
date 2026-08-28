import unittest

import xllamacpp as xlc

from ezlocalai.Embedding import Embedding


class EmbeddingParamsTests(unittest.TestCase):
    def _embedding(self):
        embedding = Embedding.__new__(Embedding)
        embedding.context_length = 8192
        embedding.batch_size = 512
        embedding.ubatch_size = 512
        embedding.n_parallel = 1
        embedding.kv_cache_type = "f16"
        return embedding

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


if __name__ == "__main__":
    unittest.main()
