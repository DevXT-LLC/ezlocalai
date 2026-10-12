#!/usr/bin/env python3
"""Build the pinned xllamacpp from source for custom CUDA architecture targets.

Requires CMake, a C/C++ compiler, Rust/cargo, and (for --cuda/--hip) its SDK.
Example: python scripts/build_xllamacpp.py --cuda --install
Existing source/build directories are reused only at the exact pinned revision.
The DFlash large-image cache fix is included upstream in the pinned llama.cpp
revision (aa94f208); no local patch is applied.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

XLLAMACPP_REVISION = "64fbe0fe7c407d5d6418bc794d89cf553736d9e3"
LLAMA_REVISION = "aa94f2086147cd4bfbcff818ce998dffce5fcb39"


def run(command, **kwargs):
    print(subprocess.list2cmdline([str(x) for x in command]), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def prepare_source(source):
    if not source.exists():
        run(
            [
                "git",
                "clone",
                "--no-checkout",
                "https://github.com/xorbitsai/xllamacpp.git",
                str(source),
            ]
        )
        run(["git", "checkout", "--detach", XLLAMACPP_REVISION], cwd=source)
        run(
            ["git", "submodule", "update", "--init", "--recursive", "--depth", "1"],
            cwd=source,
        )
    else:
        # Source directory exists (e.g. from a BuildKit cache mount). Check if it
        # is already at the expected revision; if not, fetch and re-checkout.
        actual = run(
            ["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True
        ).stdout.strip()
        if actual != XLLAMACPP_REVISION:
            print(f"Source at {actual}, expected {XLLAMACPP_REVISION} - updating")
            run(["git", "fetch", "--all"], cwd=source)
            run(["git", "checkout", "--detach", XLLAMACPP_REVISION], cwd=source)
            run(
                ["git", "submodule", "update", "--init", "--recursive", "--depth", "1"],
                cwd=source,
            )
    # Remove stale untracked files left over from prior builds (e.g. leftover
    # patch files that were copied into the source tree by an older version of
    # this script).  git checkout does not remove untracked content, so a
    # BuildKit cache mount can carry them across builds.
    run(["git", "clean", "-fd"], cwd=source)

    for path, expected in (
        (source, XLLAMACPP_REVISION),
        (source / "thirdparty/llama.cpp", LLAMA_REVISION),
    ):
        actual = run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True
        ).stdout.strip()
        if actual != expected:
            raise RuntimeError(
                f"Refusing to patch {path}: expected {expected}, got {actual}"
            )




def invalidate_extension(source):
    # setuptools does not track updated extra_objects (the native static libs)
    # when deciding whether to relink. Remove only generated binding binaries;
    # keep every object/library so cached source builds remain incremental.
    for suffix in ("so", "pyd"):
        for binary in (source / "build").glob(f"lib*/xllamacpp/xllamacpp*.{suffix}"):
            binary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--cuda", action="store_true")
    backend.add_argument("--hip", action="store_true")
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--cuda-architectures", default="86-real;89-real;120-real")
    parser.add_argument("--source-dir")
    parser.add_argument("--wheel-dir")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    workspace = Path(tempfile.mkdtemp(prefix="ezlocalai-xllamacpp-"))
    source = (
        Path(args.source_dir).resolve() if args.source_dir else workspace / "source"
    )
    wheels = Path(args.wheel_dir).resolve() if args.wheel_dir else workspace / "wheels"
    prepare_source(source)
    invalidate_extension(source)
    env = dict(
        os.environ,
        NPROC=str(args.jobs),
        CARGO_BUILD_JOBS=str(args.jobs),
        CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs),
        XLLAMACPP_RELEASE="1",
    )
    for name in (
        "XLLAMACPP_BUILD_CUDA",
        "XLLAMACPP_BUILD_HIP",
        "XLLAMACPP_BUILD_VULKAN",
    ):
        env.pop(name, None)
    if args.cuda:
        env.update(XLLAMACPP_BUILD_CUDA="1", CUDA_ARCHITECTURES=args.cuda_architectures)
        # The binding's linker omits NCCL even when CMake auto-detects it.
        # Disable that optional multi-GPU collective backend, matching our
        # single-GPU reference build and avoiding an unimportable extension.
        env["CMAKE_ARGS"] = env.get("CMAKE_ARGS", "") + " -DGGML_CUDA_NCCL=OFF"
        # setup.py searches CUDA_PATH/lib; NVIDIA Linux installs use lib64.
        cuda = Path(env.get("CUDA_PATH") or env.get("CUDA_HOME") or "/usr/local/cuda")
        env["CUDA_PATH"] = str(cuda)
        library_dirs = [
            str(cuda / part)
            for part in ("lib64", "lib64/stubs", "lib", "lib/stubs")
            if (cuda / part).is_dir()
        ]
        env["LIBRARY_PATH"] = os.pathsep.join(
            library_dirs + [env.get("LIBRARY_PATH", "")]
        )
    elif args.hip:
        env["XLLAMACPP_BUILD_HIP"] = "1"
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(source),
            "--no-deps",
            "--wheel-dir",
            str(wheels),
        ],
        env=env,
    )
    built = sorted(wheels.glob("xllamacpp-*.whl"), key=lambda p: p.stat().st_mtime)
    if not built:
        raise RuntimeError("xllamacpp build produced no wheel")
    if args.install:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(built[-1]),
            ]
        )

    print(f"Built wheel: {built[-1]}")


if __name__ == "__main__":
    main()
