# Deployment Compose reference

`compose_profiles.json` captures Docker Compose's resolved environment
for the Python repository's CUDA, GB10, CPU, Raspberry Pi, ROCm and Jetson profiles at commit
`0f7e9f45ac0d67ea806ee19874f98bb8f47c7544`. Each profile records its source
file SHA-256. Capture date: 2026-10-20.

The capture ran `docker compose --env-file /dev/null -f PROFILE config
--no-env-resolution --format json` with a clean environment containing only
PATH and the case inputs. No production environment file or credentials were
read. Each declared source setting is exercised unset, explicitly empty, and
with the synthetic value `parity_SETTING_NAME`. Fixed values in the source
profile remain fixed under overrides.

The host-only C++ test checks 1,875 resolved settings across the six profiles and
all three cases. Native implementation settings may be additional. It also
checks host architecture selection, default fleet architecture gates, and
that native ARM64 builds do not silently exclude package tests. Both CPU profiles
must hide CUDA and omit GPU reservations. ROCm retains its AMD device mappings
and groups; Jetson retains the NVIDIA container runtime. Docker Compose
is required; a running Docker daemon or GPU is not.

One intentional default correction is recorded alongside the unchanged source
capture: the Python Pi profile names `unsloth/Qwen3.5-0.6B-GGUF`, which returned
HTTP 401 without credentials during verification. The native profile uses the
published [Qwen3.5-0.8B model](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF).
The runner checks this correction only for unset/empty `DEFAULT_MODEL`
(two comparisons); explicit overrides must still match the source. It fails
if the recorded source value changes or the correction lacks an explanation.

Run from a configured root build:

```sh
cmake --build build --target test_compose_profiles
ctest --test-dir build -R '^compose_profiles$' --output-on-failure
```

These comparisons establish declared Compose defaults and interpolation
semantics. They do not establish every runtime environment variable's behavior
or GPU execution on unavailable hardware.
