#!/usr/bin/env python3
"""CPU-only checks against the pinned audit source; never loads a GPU backend."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    os.environ.get(
        "EXPERIMENTAL_PREFILL_SOURCE",
        "/home/josh/repos/xtsys/ezlocalai/build/validation/llama-v090-cuda-stage/source",
    )
)
KERNEL = Path("ggml/src/ggml-cuda/gated-delta-net-mma.cu")
OPS = Path("tests/test-backend-ops.cpp")
ENV = "GGML_CUDA_EXPERIMENTAL_GDN_SLICE64"
HASHES = {
    KERNEL: "bda5c4530c74a62ca1993d317de852635520dde7904cb199088a5dab1e9ae751",
    OPS: "9e258a11b819e947570b5a08e0ae5786dc29b19ea5198ac7bb8ef4a7da51d9d4",
}

# Compile the real selector, mocking only device information and CUDA initialization.
STUBS = r"""
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#define GGML_ASSERT assert
#define GGML_CUDA_MAX_DEVICES 2
#define GGML_CUDA_CC_AMPERE_GA10X 860
#define GGML_CUDA_CC_ADA_LOVELACE 890
#define GGML_CUDA_CC_BLACKWELL 1200
#define GGML_CUDA_CC_DGX_SPARK 1210
#define GGML_CUDA_CC_IS_RDNA3_5(cc) ((cc) == 1150)
#define GGML_UNUSED_VARS(x) (void)(x)
struct device_info { int cc, nsm; };
struct cuda_info { device_info devices[2]; } info;
const cuda_info & ggml_cuda_info() { return info; }
bool ampere_mma_available(int cc) { return cc >= 800; }
bool fail64, fail_all;
int initialized = 0;
template <int C, int V, int BLOCKS = 1> bool init(int) {
    static_assert(C == 16);
    initialized |= V == 64 ? 4 : V == 128 ? 8 : BLOCKS;
    return !fail_all && !(V == 64 && fail64);
}
"""

MAIN = r"""
int main(int argc, char ** argv) {
    assert(argc == 11);
    ggml_cuda_gdn_mma_args a{};
    info.devices[0] = {std::atoi(argv[1]), std::atoi(argv[8])};
    a.H = std::atoll(argv[2]);
    a.H_k = std::atoll(argv[3]);
    a.n_tokens = std::atoll(argv[4]);
    a.n_seqs = std::atoll(argv[5]);
    a.rq3 = std::atoll(argv[6]);
    a.eligible = std::atoi(argv[7]);
    fail64 = std::atoi(argv[9]);
    fail_all = std::atoi(argv[10]);
    const auto path = select_gdn_mma_path(0, a);
    const char * old = std::getenv("GGML_CUDA_EXPERIMENTAL_GDN_SLICE64");
    setenv("GGML_CUDA_EXPERIMENTAL_GDN_SLICE64", old && !std::strcmp(old, "1") ? "0" : "1", 1);
    assert(select_gdn_mma_path(0, a) == path);
    std::cout << static_cast<int>(path) << ' ' << initialized;
}
"""


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs)


class Slice64Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="experimental-prefill-gdn-check-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.work = Path(cls.temp.name)
        cls.original = {}
        for relative, expected in HASHES.items():
            data = (SOURCE / relative).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise AssertionError(
                    f"Audit source changed: {relative}; re-audit first"
                )
            cls.original[relative] = data.decode()
            target = cls.work / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / relative, target)
        for suffix in ("", "-tests"):
            patch = ROOT / f"patches/experimental-prefill-gdn-slice64{suffix}.patch"
            run("git", "apply", "--check", str(patch), cwd=cls.work)
            run("git", "apply", str(patch), cwd=cls.work)
            run("git", "apply", "--reverse", "--check", str(patch), cwd=cls.work)
        cls.patched = (cls.work / KERNEL).read_text()
        header = (SOURCE / KERNEL.with_suffix(".cuh")).read_text()
        args = header[header.index("struct ggml_cuda_gdn_mma_args") :]
        args = args[: args.index("\n};") + 3]
        cls.executables = {}
        for variant, source in (
            ("baseline", cls.original[KERNEL]),
            ("candidate", cls.patched),
        ):
            selector = source[
                source.index("enum class gdn_path") : source.index("}  // namespace")
            ]
            cpp = cls.work / f"{variant}.cpp"
            cpp.write_text(STUBS + args + "\n" + selector + MAIN)
            for backend, flags in (
                ("cuda", []),
                ("hip", ["-DGGML_USE_HIP"]),
                ("musa", ["-DGGML_USE_MUSA"]),
            ):
                exe = cls.work / f"{variant}-{backend}"
                run("g++", "-std=c++17", "-O0", *flags, str(cpp), "-o", str(exe))
                cls.executables[variant, backend] = exe

    def select(self, value=None, variant="candidate", backend="cuda", **overrides):
        values = dict(
            cc=1200,
            H=32,
            H_k=16,
            tokens=128,
            seqs=1,
            rq3=1,
            eligible=1,
            nsm=170,
            fail64=0,
            fail_all=0,
        )
        values.update(overrides)
        env = os.environ.copy()
        env.pop(ENV, None)
        if value is not None:
            env[ENV] = value
        result = run(
            str(self.executables[variant, backend]), *map(str, values.values()), env=env
        )
        return tuple(map(int, result.stdout.split()))

    def test_default_and_invalid_values_match_baseline(self):
        for heads in (16, 32, 48, 64):
            for nsm in (31, 32, 63, 64, 127, 128, 170):
                baseline = self.select(variant="baseline", H=heads, nsm=nsm)
                for value in (None, "", "0", "true", "01", "1 "):
                    with self.subTest(H=heads, nsm=nsm, value=value):
                        self.assertEqual(self.select(value, H=heads, nsm=nsm), baseline)

    def test_opt_in_all_allowed_architectures_and_head_counts(self):
        for cc in (860, 890, 1200, 1210):
            for heads in (16, 32, 48, 64):
                for seqs in (1, 2, 3):
                    with self.subTest(cc=cc, H=heads, seqs=seqs):
                        self.assertEqual(
                            self.select("1", cc=cc, H=heads, seqs=seqs), (3, 4)
                        )

    def test_all_admission_guards_precede_experiment(self):
        for override in (
            dict(eligible=0),
            dict(H_k=32),
            dict(H=8),
            dict(tokens=63),
            dict(tokens=2**63 - 32),
            dict(rq3=0),
            dict(seqs=0),
            dict(seqs=(2**31 - 1) // (32 * 4) + 1),
            dict(cc=750),
            dict(cc=800),
            dict(cc=900),
            dict(cc=1000),
        ):
            with self.subTest(**override):
                self.assertEqual(self.select("1", **override), (0, 0))
        for tokens in (64, 65, 128, 129, 2049, 2**63 - 33):
            self.assertEqual(self.select("1", tokens=tokens), (3, 4))

    def test_failed_slice64_init_keeps_original_schedule(self):
        for nsm, expected in ((128, (1, 5)), (64, (2, 6)), (63, (4, 12))):
            self.assertEqual(self.select("1", fail64=1, nsm=nsm), expected)
            self.assertEqual(self.select("1", fail_all=1, nsm=nsm), (0, expected[1]))

    def test_hip_and_musa_selection_unchanged(self):
        for backend in ("hip", "musa"):
            for cc in (860, 1150, 1200):
                for tokens in (64, 2047, 2048):
                    expected = self.select(
                        variant="baseline", backend=backend, cc=cc, tokens=tokens
                    )
                    self.assertEqual(
                        self.select("1", backend=backend, cc=cc, tokens=tokens),
                        expected,
                    )

    def test_kernel_math_and_initialization_are_byte_identical(self):
        start, end = "namespace ggml_cuda_gated_delta_net_mma {", "enum class gdn_path"
        original = self.original[KERNEL]
        self.assertEqual(
            original[original.index(start) : original.index(end)],
            self.patched[self.patched.index(start) : self.patched.index(end)],
        )
        self.assertIn(
            "mma(c, al, bh);\n        mma(c, ah, bl);\n        mma(c, ah, bh);",
            self.patched,
        )

    def test_launch_and_null_state_guard(self):
        launch = self.patched[self.patched.index("bool ggml_cuda_gdn_mma_launch") :]
        self.assertLess(
            launch.index("if (!a.state_out)"), launch.index("select_gdn_mma_path")
        )
        self.assertIn(
            "if (path == gdn_path::slice64) {\n"
            "        gdn_single<16, 64><<<a.H * a.n_seqs * 2, dim3(32, 8), "
            "sizeof(shared<16, 64>), stream>>>(a);",
            launch,
        )

    def test_slice_ownership_covers_every_state_element_once(self):
        # Schedule-level model, not a numerical GPU test: each tile owns 16 x 8 values.
        for width in (32, 64, 128):
            seen = set()
            per_warp = (128 // 16) * (width // 8) // 8
            for v0 in range(0, 128, width):
                for warp in range(8):
                    for j in range(per_warp):
                        ti = warp + j * 8
                        row, col = (ti // (width // 8)) * 16, (ti % (width // 8)) * 8
                        for r in range(row, row + 16):
                            for c in range(v0 + col, v0 + col + 8):
                                self.assertNotIn((r, c), seen)
                                seen.add((r, c))
            self.assertEqual(seen, {(r, c) for r in range(128) for c in range(128)})

    def test_operator_patch_adds_eligible_perf_cases_without_changing_tolerances(self):
        patched = (self.work / OPS).read_text()
        original = self.original[OPS]
        begin, end = "struct test_gated_delta_net :", "// GGML_OP_GATED_DELTA_NET +"
        self.assertEqual(
            original[original.index(begin) : original.index(end)],
            patched[patched.index(begin) : patched.index(end)],
        )
        self.assertIn("GGML_TYPE_F32, 16, 128, n_tokens, n_seqs, v_repeat", patched)

    def test_audit_source_still_untouched(self):
        for relative, expected in HASHES.items():
            self.assertEqual(
                hashlib.sha256((SOURCE / relative).read_bytes()).hexdigest(), expected
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
