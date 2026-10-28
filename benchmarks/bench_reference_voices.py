#!/usr/bin/env python3
"""Render selectable reference voices with the existing native Qwen worker.

An explicitly supplied immutable image and local model/runtime paths are used.
This starts only an owned CPU-only container; it never inspects or stops a
production worker. Outputs are demonstrations, not voice-quality scores or
measurements of GPU/live-call latency.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
import uuid
import wave
from array import array


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, help='Immutable local sha256 image ID')
    parser.add_argument('--runtime', type=Path, required=True, help='Matching CPU runtime bundle')
    parser.add_argument('--model-cache', type=Path, required=True, help='HF repository cache including snapshots/blobs')
    parser.add_argument('--revision', required=True)
    parser.add_argument('--voices', type=Path, required=True)
    parser.add_argument('--voice', action='append', required=True, help='Reference WAV basename (repeatable)')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--text', default='Hello. The working agent is running the tests. I can help you steer it.')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', args.image):
        parser.error('--image must be an immutable image ID')
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        parser.error('--revision must be an immutable repository revision')
    if not 1 <= args.threads <= 32 or not math.isfinite(args.timeout) or not 0 < args.timeout <= 600:
        parser.error('invalid thread count or timeout')
    for voice in args.voice:
        if Path(voice).name != voice or not voice.endswith('.wav') or not (args.voices / voice).is_file():
            parser.error('each --voice must name an existing WAV inside --voices')
    runtime, cache, voices, output = [p.resolve() for p in (args.runtime, args.model_cache, args.voices, args.output)]
    model_names = ('qwen-talker-0.6b-base-Q8_0.gguf', 'qwen-tokenizer-12hz-f16.gguf')
    model_files = [cache / 'snapshots' / args.revision / name for name in model_names]
    if not all(p.is_file() for p in model_files) or not (runtime / 'ezlocalai-tts').is_file():
        parser.error('model files or matching runtime missing')
    output.mkdir(parents=True, exist_ok=False)
    owner = 'qwen-voice-probe-' + uuid.uuid4().hex[:12]
    name = 'ezlocalai-' + owner
    report = {'purpose': 'CPU-only native Qwen selectable-voice feasibility', 'text': args.text,
              'image': args.image, 'container': name, 'owner': owner,
              'threads': args.threads, 'model_revision': args.revision,
              'model_sha256': {p.name: digest(p) for p in model_files},
              'runtime_sha256': digest(runtime / 'ezlocalai-tts'), 'cases': [],
              'limits': ['CPU timings do not predict GPU or end-to-end LFM call latency.',
                         'PCM validity/differences do not measure voice similarity, intelligibility or human preference.',
                         'Existing native Qwen uses speaker embeddings, not reference-transcript ICL.',
                         'Streaming vocoder cadence remains 72 frames; no shorter-window quality tradeoff.']}
    command = ['docker', 'run', '--pull=never', '--rm', '-i', '--name', name, '--label', 'devxt.voice_probe=' + owner,
               '--runtime', 'runc', '--network', 'none', '--cpus', str(args.threads), '--memory', '8g',
               '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'CUDA_VISIBLE_DEVICES=-1',
               '-e', 'GGML_BACKEND_PATH=/runtime', '-e', 'LD_LIBRARY_PATH=/runtime',
               '-v', str(runtime) + ':/runtime:ro', '-v', str(cache) + ':/model-cache:ro',
               '-v', str(voices) + ':/voices:ro', '--entrypoint', '/runtime/ezlocalai-tts', args.image,
               '-m', '/model-cache/snapshots/' + args.revision + '/' + model_names[0],
               '-mm', '/model-cache/snapshots/' + args.revision + '/' + model_names[1],
               '-ngl', '0', '--no-mmproj-offload', '--device', 'none', '-t', str(args.threads),
               '-c', '4096', '-b', '512', '-ub', '512', '--temp', '0.9', '--top-k', '50',
               '--top-p', '1', '--repeat-penalty', '1.05', '--seed', '42', '--flash-attn', 'on']
    messages = queue.Queue(maxsize=16)
    process = None
    try:
        with (output / 'worker.log').open('wb') as log:
            started = time.monotonic()
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
            def read():
                try:
                    while True:
                        line = process.stdout.readline(4 * 1024 * 1024 + 1)
                        if not line:
                            messages.put(None)
                            return
                        if len(line) > 4 * 1024 * 1024:
                            raise RuntimeError('oversize worker response')
                        messages.put(json.loads(line))
                except BaseException as error:
                    messages.put(error)
            threading.Thread(target=read, daemon=True).start()
            def receive(deadline):
                value = messages.get(timeout=max(.001, deadline - time.monotonic()))
                if isinstance(value, BaseException):
                    raise value
                if value is None or 'error' in value:
                    raise RuntimeError('native worker failed: ' + str(value))
                return value
            ready = receive(started + args.timeout)
            if ready.get('ready') is not True:
                raise RuntimeError('invalid worker readiness')
            report['startup_seconds'] = time.monotonic() - started
            for index, voice in enumerate(args.voice):
                start = time.monotonic()
                payload = {'text': args.text, 'speaker': '/voices/' + voice, 'stream': True,
                           'language': 'english', 'max_new_tokens': 192}
                process.stdin.write((json.dumps(payload) + '\n').encode())
                process.stdin.flush()
                pcm = bytearray()
                chunks, rate, first = [], None, None
                deadline = start + args.timeout
                while True:
                    event = receive(deadline)
                    if 'pcm_f32' in event:
                        raw = base64.b64decode(event['pcm_f32'], validate=True)
                        if len(raw) % 4 or (rate is not None and rate != event['sample_rate']):
                            raise RuntimeError('invalid PCM or changing sample rate')
                        rate = event['sample_rate']
                        if raw and first is None:
                            first = time.monotonic() - start
                        chunks.append(len(raw))
                        pcm.extend(raw)
                    elif event.get('ok') is True:
                        if event.get('limit_reached') or not pcm:
                            raise RuntimeError('truncated or empty speech')
                        break
                    else:
                        raise RuntimeError('unexpected native event')
                if rate != 24000:
                    raise RuntimeError('unexpected sample rate')
                samples = array('f')
                samples.frombytes(pcm)
                if not all(math.isfinite(x) for x in samples):
                    raise RuntimeError('nonfinite speech')
                peak = max(abs(x) for x in samples)
                if peak < 1e-6:
                    raise RuntimeError('silent speech')
                converted = array('h', (max(-32768, min(32767, round(x * 32767))) for x in samples))
                filename = str(index + 1) + '-' + Path(voice).stem + '.wav'
                with wave.open(str(output / filename), 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(rate)
                    wav.writeframes(converted.tobytes())
                case = {'voice': voice, 'reference_sha256': digest(voices / voice), 'file': filename,
                        'first_audio_seconds': first, 'total_seconds': time.monotonic() - start,
                        'audio_seconds': len(samples) / rate, 'frames': event['frames'],
                        'peak_absolute': peak,
                        'rms': math.sqrt(sum(x*x for x in samples) / len(samples)),
                        'clipped_samples': sum(abs(x) > 1 for x in samples),
                        'pcm_f32_sha256': hashlib.sha256(pcm).hexdigest(), 'chunk_bytes': chunks}
                report['cases'].append(case)
                print(json.dumps({k: case[k] for k in ('voice', 'first_audio_seconds', 'total_seconds', 'audio_seconds')}), flush=True)
            report['status'] = 'passed'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        if process is not None:
            inspected = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
            if inspected.returncode == 0:
                info = json.loads(inspected.stdout)[0]
                if info['Config'].get('Labels', {}).get('devxt.voice_probe') != owner:
                    raise RuntimeError('refusing cleanup of unowned container')
                subprocess.run(['docker', 'rm', '-f', info['Id']], check=True, stdout=subprocess.DEVNULL)
            process.wait(timeout=30)
        (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
