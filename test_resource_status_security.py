"""Keep CUDA diagnostics in logs without leaking driver errors in API status."""

import ast
import logging
from pathlib import Path
from typing import Dict, List
import unittest
from unittest import mock

import torch


class ResourceStatusSecurityTests(unittest.TestCase):
    def test_gpu_failure_keeps_other_gpu_stats_and_logs_private_details(self):
        # Compile the real method without initializing inference backends.
        tree = ast.parse(Path(__file__).with_name("Pipes.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "ResourceManager"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "get_torch_cuda_memory_stats"
        )
        namespace = {"torch": torch, "logging": logging, "List": List, "Dict": Dict}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), "Pipes.py", "exec"),
            namespace,
        )
        with mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(
            torch.cuda, "device_count", return_value=2
        ), mock.patch.object(
            torch.cuda,
            "mem_get_info",
            side_effect=[RuntimeError("private /server/model path"), (2**30, 2**31)],
        ), mock.patch.object(
            torch.cuda, "memory_allocated", return_value=0
        ), mock.patch.object(
            torch.cuda, "memory_reserved", return_value=0
        ), mock.patch.object(
            torch.cuda, "max_memory_allocated", return_value=0
        ), self.assertLogs(
            level="ERROR"
        ) as logs:
            stats = namespace["get_torch_cuda_memory_stats"](None)
        self.assertEqual(
            stats[0], {"index": 0, "error": "CUDA memory statistics unavailable"}
        )
        self.assertEqual(stats[1]["free_gb"], 1)
        self.assertEqual(stats[1]["total_gb"], 2)
        self.assertNotIn("private", str(stats))
        self.assertIn("private /server/model path", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
