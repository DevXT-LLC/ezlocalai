# ezlocalai Benchmark Results

## Benchmark: Full Code Analysis (Pipes.py - ~1000 lines, ~10k tokens)

### Before Simplification (2025-11-25)

Configuration:
- DEFAULT_MODEL: `unsloth/Qwen3-VL-4B-Instruct-GGUF@64000,unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF@128000`
- GPU_LAYERS: -1 (auto-calibration)
- VRAM_BUDGET: 20 (manual)

#### 4B Model (64k fixed context, 37/37 GPU layers)

```text
Prompt tokens: 8,886
Completion tokens: 615
Total time: 7.8s
Generation speed: 78.5 tok/s overall

LLM Timings:
  Prompt processing: 13,553.1 tok/s
  Generation: 148.6 tok/s
```

#### 30B Model (128k fixed context, 5/49 GPU layers)

```text
Prompt tokens: 8,886
Completion tokens: 547
Total time: 127.9s
Generation speed: 4.3 tok/s overall

LLM Timings:
  Prompt processing: 4.5 tok/s
  Generation: 4.3 tok/s
```

### Key Observations (Before)

- 30B with 128k fixed context = only 5 GPU layers fit = very slow
- 4B with 64k fixed context = 37/37 GPU layers = very fast
- The fixed context wastes VRAM on unused KV cache capacity

---

## After Simplification (Dynamic Context)

Configuration:
- DEFAULT_MODEL: `unsloth/Qwen3-VL-4B-Instruct-GGUF,unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`
- No GPU_LAYERS env (always auto)
- No VRAM_BUDGET env (auto-detected from GPU)
- Context is dynamic (rounded up to nearest 32k)

#### 4B Model (dynamic 32k context, 37/37 GPU layers)

```text
Prompt tokens: 9,798
Completion tokens: 632
Total time: 6.1s
Generation speed: 103.8 tok/s overall

LLM Timings:
  Prompt processing: 14,163.0 tok/s
  Generation: 151.5 tok/s
```

#### 30B Model (dynamic 32k context, 38/49 GPU layers)

```text
Prompt tokens: 9,798
Completion tokens: 484
Total time: 47.6s
Generation speed: 10.2 tok/s overall

LLM Timings:
  Prompt processing: 1,015.8 tok/s
  Generation: 14.1 tok/s
```

### Improvements After Simplification

| Metric | 4B Before | 4B After | Improvement |
|--------|-----------|----------|-------------|
| Total Time | 7.8s | 6.1s | **22% faster** |
| Prompt Speed | 13,553 tok/s | 14,163 tok/s | 5% faster |
| Gen Speed | 148.6 tok/s | 151.5 tok/s | 2% faster |

| Metric | 30B Before | 30B After | Improvement |
|--------|------------|-----------|-------------|
| Total Time | 127.9s | 47.6s | **63% faster** |
| Prompt Speed | 4.5 tok/s | 1,015.8 tok/s | **226x faster** |
| Gen Speed | 4.3 tok/s | 14.1 tok/s | **3.3x faster** |
| GPU Layers | 5/49 (10%) | 38/49 (78%) | **7.6x more** |

### Summary

The dynamic context sizing provides massive speedups:

1. **30B is now 63% faster** overall because it gets 38 GPU layers instead of 5
2. **Prompt processing on 30B is 226x faster** (1,016 vs 4.5 tok/s)
3. **4B is 22% faster** with smaller context overhead
4. **Simpler configuration** - no need to manually tune context or VRAM

The key insight: **fixed large contexts waste VRAM on empty KV cache**. By sizing context dynamically based on actual prompt size, we can fit more model layers on GPU.

