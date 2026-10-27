"""Offline CLI guards for the optional standalone CUDA compute diagnostic.

Build the binary separately, then set GPU_COMPUTE_PROBE_BIN or pass --probe-bin.
Without a built binary this suite skips. Every invocation hides CUDA devices;
no test invokes --run, acquires a maintenance lease or changes worker settings.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import unittest


class GPUComputeProbeOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("GPU_COMPUTE_PROBE_BIN")
        if not configured:
            raise unittest.SkipTest("Set GPU_COMPUTE_PROBE_BIN to a separately built compute probe")
        cls.binary = Path(configured).expanduser().resolve()
        if not cls.binary.is_file():
            raise RuntimeError("Configured compute probe does not exist: " + str(cls.binary))

    def invoke(self, arguments, success=False, peak=None):
        self.assertNotIn("--run", arguments)
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void"}
        result = subprocess.run([str(self.binary)] + arguments, env=environment,
                                text=True, capture_output=True, timeout=15, check=False)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["schema"], "ezlocalai_gpu_compute_probe_v1")
        self.assertEqual(output["status"], "plan" if success else "error")
        if success:
            self.assertIs(output["cuda_called"], False)
            self.assertEqual(output["planned_peak_explicit_device_bytes"], peak)
        else:
            self.assertTrue(output["error"])
        return output

    def plan(self, *extra, success=False, peak=None):
        return self.invoke(["--plan", "--max-bytes", "805306368", *extra], success, peak)

    def test_default_plan_has_bounded_peak_without_cuda(self):
        self.plan(success=True, peak=570425344)

    def test_int8_4096_plan_has_bounded_peak_without_cuda(self):
        self.plan("--sizes", "4096", "--dtypes", "int8", success=True, peak=134217728)

    def test_mode_is_required(self):
        self.invoke([])

    def test_allocation_cap_is_required(self):
        self.invoke(["--plan"])

    def test_insufficient_allocation_cap(self):
        self.invoke(["--plan", "--max-bytes", "1"])

    def test_negative_allocation_cap(self):
        self.invoke(["--plan", "--max-bytes", "-1"])

    def test_unsupported_dimension(self):
        self.plan("--sizes", "1024")

    def test_duplicate_dimension(self):
        self.plan("--sizes", "8192,8192")

    def test_unsupported_dtype(self):
        self.plan("--dtypes", "fp8")

    def test_reserve_cannot_be_removed(self):
        self.plan("--min-free-bytes", "1")

    def test_allocation_budget_overflow(self):
        self.invoke(["--plan", "--max-bytes", "18446744073709551615"])

    def test_reserve_budget_overflow(self):
        self.plan("--min-free-bytes", "18446744073709551615")

    def test_zero_samples(self):
        self.plan("--samples", "0")

    def test_excessive_iterations(self):
        self.plan("--iterations", "1001")

    def test_unknown_option(self):
        self.plan("--unknown", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--probe-bin")
    options, remaining = parser.parse_known_args()
    if options.probe_bin:
        os.environ["GPU_COMPUTE_PROBE_BIN"] = options.probe_bin
    unittest.main(argv=[__file__, *remaining])
