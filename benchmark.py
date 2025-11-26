#!/usr/bin/env python3
"""
ezlocalai Benchmark Script
Tests LLM inference speed, TTS, and image generation performance.
"""

import time
import requests
import json
import sys

BASE_URL = "http://localhost:8091"

def benchmark_chat(prompt: str, max_tokens: int = 100, name: str = "Chat") -> dict:
    """Benchmark a chat completion request."""
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            },
            timeout=300
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            tokens_per_sec = completion_tokens / elapsed if elapsed > 0 else 0
            
            return {
                "name": name,
                "success": True,
                "elapsed": elapsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "tokens_per_sec": tokens_per_sec,
                "content": data["choices"][0]["message"]["content"][:200]
            }
        else:
            return {"name": name, "success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
    except Exception as e:
        return {"name": name, "success": False, "error": str(e)}


def benchmark_tts(text: str, name: str = "TTS") -> dict:
    """Benchmark text-to-speech generation."""
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/v1/audio/speech",
            json={
                "input": text,
                "voice": "default",
                "response_format": "wav"
            },
            timeout=300
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            audio_size = len(response.content)
            return {
                "name": name,
                "success": True,
                "elapsed": elapsed,
                "audio_size_kb": audio_size / 1024,
                "text_length": len(text)
            }
        else:
            return {"name": name, "success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
    except Exception as e:
        return {"name": name, "success": False, "error": str(e)}


def benchmark_image(prompt: str, name: str = "Image") -> dict:
    """Benchmark image generation."""
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/v1/images/generations",
            json={
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024"
            },
            timeout=300
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            return {
                "name": name,
                "success": True,
                "elapsed": elapsed,
                "images": len(data.get("data", []))
            }
        else:
            return {"name": name, "success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
    except Exception as e:
        return {"name": name, "success": False, "error": str(e)}


def get_models():
    """Get available models."""
    try:
        response = requests.get(f"{BASE_URL}/v1/models", timeout=30)
        if response.status_code == 200:
            return response.json().get("data", [])
    except:
        pass
    return []


def run_benchmarks():
    """Run all benchmarks and print results."""
    print("=" * 70)
    print("ezlocalai Benchmark Suite")
    print("=" * 70)
    
    # Check server availability
    try:
        response = requests.get(f"{BASE_URL}/v1/models", timeout=10)
        if response.status_code != 200:
            print("ERROR: Server not responding correctly")
            return
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {BASE_URL}: {e}")
        return
    
    models = get_models()
    print(f"\nAvailable models: {[m.get('id') for m in models]}")
    
    results = []
    
    # Warmup
    print("\n[Warmup] Running warmup request...")
    warmup = benchmark_chat("Hello", max_tokens=10, name="Warmup")
    if warmup["success"]:
        print(f"  Warmup complete in {warmup['elapsed']:.2f}s")
    else:
        print(f"  Warmup failed: {warmup.get('error')}")
    
    # LLM Benchmarks
    print("\n" + "-" * 70)
    print("LLM Inference Benchmarks")
    print("-" * 70)
    
    tests = [
        ("Short response", "What is 2+2? Answer briefly.", 20),
        ("Medium response", "Explain what Python is in 2-3 sentences.", 100),
        ("Long response", "Write a detailed explanation of how neural networks work, including layers, activation functions, and backpropagation.", 300),
        ("Code generation", "Write a Python function to calculate the fibonacci sequence recursively with memoization.", 200),
    ]
    
    for name, prompt, max_tokens in tests:
        print(f"\n  [{name}] {prompt[:50]}...")
        result = benchmark_chat(prompt, max_tokens, name)
        results.append(result)
        
        if result["success"]:
            print(f"    Time: {result['elapsed']:.2f}s")
            print(f"    Tokens: {result['prompt_tokens']} prompt + {result['completion_tokens']} completion = {result['total_tokens']} total")
            print(f"    Speed: {result['tokens_per_sec']:.1f} tokens/sec")
        else:
            print(f"    FAILED: {result.get('error', 'Unknown error')}")
    
    # TTS Benchmark
    print("\n" + "-" * 70)
    print("TTS Benchmarks")
    print("-" * 70)
    
    tts_tests = [
        ("Short TTS", "Hello, how are you today?"),
        ("Medium TTS", "The quick brown fox jumps over the lazy dog. This is a test of the text to speech system."),
        ("Long TTS", "Artificial intelligence has revolutionized the way we interact with technology. From virtual assistants to autonomous vehicles, AI systems are becoming increasingly sophisticated and capable of handling complex tasks that were once thought to be the exclusive domain of human intelligence."),
    ]
    
    for name, text in tts_tests:
        print(f"\n  [{name}] {text[:50]}...")
        result = benchmark_tts(text, name)
        results.append(result)
        
        if result["success"]:
            print(f"    Time: {result['elapsed']:.2f}s")
            print(f"    Audio size: {result['audio_size_kb']:.1f} KB")
            print(f"    Text length: {result['text_length']} chars")
        else:
            print(f"    FAILED: {result.get('error', 'Unknown error')}")
    
    # Image Generation Benchmark (if available)
    print("\n" + "-" * 70)
    print("Image Generation Benchmarks")
    print("-" * 70)
    
    print("\n  [Image Gen] A beautiful sunset over mountains...")
    result = benchmark_image("A beautiful sunset over mountains, digital art", "Image Generation")
    results.append(result)
    
    if result["success"]:
        print(f"    Time: {result['elapsed']:.2f}s")
        print(f"    Images generated: {result['images']}")
    else:
        print(f"    SKIPPED/FAILED: {result.get('error', 'Unknown error')}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    llm_results = [r for r in results if r["name"] not in ["Warmup", "Short TTS", "Medium TTS", "Long TTS", "Image Generation"] and r["success"]]
    if llm_results:
        avg_tokens_sec = sum(r["tokens_per_sec"] for r in llm_results) / len(llm_results)
        total_tokens = sum(r["completion_tokens"] for r in llm_results)
        total_time = sum(r["elapsed"] for r in llm_results)
        print(f"\nLLM Performance:")
        print(f"  Average speed: {avg_tokens_sec:.1f} tokens/sec")
        print(f"  Total tokens generated: {total_tokens}")
        print(f"  Total LLM time: {total_time:.2f}s")
    
    tts_results = [r for r in results if "TTS" in r["name"] and r["success"]]
    if tts_results:
        total_tts_time = sum(r["elapsed"] for r in tts_results)
        total_audio_kb = sum(r["audio_size_kb"] for r in tts_results)
        print(f"\nTTS Performance:")
        print(f"  Total TTS time: {total_tts_time:.2f}s")
        print(f"  Total audio generated: {total_audio_kb:.1f} KB")
    
    img_results = [r for r in results if "Image" in r["name"] and r["success"]]
    if img_results:
        print(f"\nImage Generation:")
        print(f"  Time: {img_results[0]['elapsed']:.2f}s")
    
    print("\n" + "=" * 70)
    return results


if __name__ == "__main__":
    run_benchmarks()
