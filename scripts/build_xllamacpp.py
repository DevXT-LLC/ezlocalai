#!/usr/bin/env python3
"""Build the pinned xllamacpp with the DFlash large-image cache hotfix.

Requires CMake, a C/C++ compiler, Rust/cargo, and (for --cuda/--hip) its SDK.
Example: python scripts/build_xllamacpp.py --cuda --install
Existing source/build directories are reused only at the exact pinned revision.
The patch is applied by xllamacpp's own patch-aware build, including its existing
compatibility patches. No models, target KV precision or sampling are changed.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

XLLAMACPP_REVISION = "c490a681ce9ac3490d74b418fee5515ac5c1343e"
LLAMA_REVISION = "5266f24da75dc449bd56cbed7addb9c8e4a6a73e"
HOTFIX = "dflash-pinned-image-positions-v1"


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
    patch = (
        Path(__file__).resolve().parents[1]
        / "native/patches/dflash-pinned-image-positions.patch"
    )
    destination = source / "patches/llama.cpp/0003-dflash-pinned-image-positions.patch"
    if destination.exists() and destination.read_bytes() != patch.read_bytes():
        raise RuntimeError(f"Refusing to overwrite a different patch: {destination}")
    shutil.copyfile(patch, destination)
    # Inspectable provenance, bundled in the wheel alongside the Python bindings.
    (source / "src/xllamacpp/_ezlocalai_hotfix.py").write_text(
        f"HOTFIX = {HOTFIX!r}\nLLAMA_REVISION = {LLAMA_REVISION!r}\n"
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
        run(
            [
                sys.executable,
                "-c",
                f"from xllamacpp._ezlocalai_hotfix import HOTFIX; assert HOTFIX == {HOTFIX!r}; print(HOTFIX)",
            ]
        )
    print(f"Patched wheel: {built[-1]}")


if __name__ == "__main__":
    main()
