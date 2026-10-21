#pragma once

namespace ezlocalai {
// Native llama.cpp engine: release v0.7.0. This is newer than the Python
// reference's xllamacpp engine (aa94f2086, b11549, in scripts/build_xllamacpp.py).
inline constexpr char llama_cpp_reference_revision[] = "620d2225cc436ab0b001e3200e590e3b286ef9ab";
inline constexpr char xllamacpp_reference_revision[] = "64fbe0fe7c407d5d6418bc794d89cf553736d9e3";
// Versioneer metadata in the pinned binding's source archive.
inline constexpr char xllamacpp_reference_version[] = "2026.10.11549";
inline constexpr int llm_default_context_checkpoints = 32;
inline constexpr int llm_default_checkpoint_min_step = 8192;
}
