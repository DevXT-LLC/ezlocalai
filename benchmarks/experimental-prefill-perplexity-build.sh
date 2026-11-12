#!/usr/bin/env bash
# Compiler-only addition to the completed CUDA 12.8 build; no dependency rebuild.
set -euo pipefail
cd /validation/build
test ! -e bin/llama-perplexity
test ! -e bin/libllama-perplexity-impl.so
sha256sum bin/* CMakeCache.txt > /validation/perplexity-existing.sha256
verify_existing() {
    status=$?
    trap - EXIT
    sha256sum -c /validation/perplexity-existing.sha256 \
        > /validation/perplexity-existing-check.log || status=1
    exit "$status"
}
trap verify_existing EXIT
date -u +%FT%TZ | tee /validation/perplexity-start.txt
make -f tools/perplexity/CMakeFiles/llama-perplexity-impl.dir/build.make -j4 \
    tools/perplexity/CMakeFiles/llama-perplexity-impl.dir/build \
    2>&1 | tee /validation/perplexity-impl-build.log
make -f tools/perplexity/CMakeFiles/llama-perplexity.dir/build.make -j4 \
    tools/perplexity/CMakeFiles/llama-perplexity.dir/build \
    2>&1 | tee /validation/perplexity-exe-build.log
test -x bin/llama-perplexity
sha256sum bin/llama-perplexity bin/libllama-perplexity-impl.so \
    | tee /validation/perplexity-artifact-sha256.txt
date -u +%FT%TZ | tee /validation/perplexity-finish.txt
