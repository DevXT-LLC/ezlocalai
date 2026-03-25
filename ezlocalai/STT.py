import os
import base64
import uuid
import logging
import time
import webrtcvad
import pyaudio
import wave
import threading
import numpy as np
import gc
import torch
from io import BytesIO
from faster_whisper import WhisperModel

try:
    from faster_whisper import BatchedInferencePipeline
except ImportError:
    BatchedInferencePipeline = None


class VoicePrintStore:
    """
    Stores speaker voice prints (MFCC centroids) keyed by session_id and
    speaker UUID so that the same physical speaker gets a consistent ID
    across multiple transcription chunks within the same session.

    Voice prints are NOT listed as available TTS voices.
    Sessions auto-expire after `ttl_seconds` (default 24 hours).
    """

    def __init__(self, ttl_seconds: int = 86400):
        # { session_id: { speaker_uuid: { "centroid": np.ndarray, "count": int } } }
        self._prints: dict = {}
        self._timestamps: dict = {}  # { session_id: last_access_time }
        self._ttl = ttl_seconds

    def _prune_expired(self):
        """Remove sessions that haven't been accessed within the TTL."""
        now = time.time()
        expired = [sid for sid, ts in self._timestamps.items() if now - ts > self._ttl]
        for sid in expired:
            del self._prints[sid]
            del self._timestamps[sid]
        if expired:
            logging.debug(f"[VoicePrintStore] Pruned {len(expired)} expired session(s)")

    def get_session_prints(self, session_id: str) -> dict:
        """Return stored voice prints for a session, or empty dict."""
        self._prune_expired()
        self._timestamps[session_id] = time.time()
        return self._prints.get(session_id, {})

    def update_speaker(
        self, session_id: str, speaker_uuid: str, centroid: np.ndarray, count: int = 1
    ):
        """Update or create a voice print entry for a speaker in a session."""
        if session_id not in self._prints:
            self._prints[session_id] = {}
        if speaker_uuid in self._prints[session_id]:
            existing = self._prints[session_id][speaker_uuid]
            # Running weighted average of centroid
            old_count = existing["count"]
            new_count = old_count + count
            existing["centroid"] = (
                existing["centroid"] * old_count + centroid * count
            ) / new_count
            existing["count"] = new_count
        else:
            self._prints[session_id][speaker_uuid] = {
                "centroid": centroid.copy(),
                "count": count,
            }
        self._timestamps[session_id] = time.time()

    def match_centroid(
        self, session_id: str, centroid: np.ndarray, threshold: float = 0.3
    ) -> str | None:
        """
        Find the best-matching stored voice print for this centroid.
        Returns the speaker UUID if cosine similarity exceeds (1 - threshold),
        otherwise None (meaning this is a new speaker).
        """
        prints = self.get_session_prints(session_id)
        if not prints:
            return None

        best_uuid = None
        best_similarity = -1.0

        for spk_uuid, data in prints.items():
            stored = data["centroid"]
            # Cosine similarity
            dot = np.dot(centroid, stored)
            norm = np.linalg.norm(centroid) * np.linalg.norm(stored)
            if norm == 0:
                continue
            similarity = dot / norm
            if similarity > best_similarity:
                best_similarity = similarity
                best_uuid = spk_uuid

        if best_similarity >= (1.0 - threshold):
            return best_uuid
        return None


# Global voice print store shared across all STT instances
_voice_print_store = VoicePrintStore()

# Suppress ALSA warnings in Docker containers without sound cards
os.environ.setdefault("PYTHONWARNINGS", "ignore")


def _get_optimal_cpu_threads():
    """Get optimal number of CPU threads for whisper inference."""
    try:
        import multiprocessing

        # Use half of available cores to leave room for other processes
        # but at least 2 threads and at most 8 (diminishing returns beyond that)
        cores = multiprocessing.cpu_count()
        return max(2, min(8, cores // 2))
    except:
        return 4  # Safe default


def _check_cudnn_available():
    """Check if cuDNN is available and working for CTranslate2/faster-whisper."""
    if not torch.cuda.is_available():
        return False
    try:
        # Check if cuDNN is available via torch
        if not torch.backends.cudnn.is_available():
            return False

        # Actually test cuDNN by running a small convolution
        # This catches version mismatch issues that is_available() misses
        test_tensor = torch.randn(1, 1, 8, 8, device="cuda")
        conv = torch.nn.Conv2d(1, 1, 3, padding=1).cuda()
        with torch.backends.cudnn.flags(enabled=True):
            _ = conv(test_tensor)

        # Clean up
        del test_tensor, conv
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        logging.warning(f"[STT] cuDNN test failed: {e}")
        return False


def get_available_vram_mb():
    """Get available VRAM in MB."""
    if torch.cuda.is_available():
        try:
            free_memory = torch.cuda.get_device_properties(
                0
            ).total_memory - torch.cuda.memory_allocated(0)
            return free_memory / (1024 * 1024)
        except:
            return 0
    return 0


class STT:
    def __init__(self, model="base", wake_functions={}):
        # Check if there's enough VRAM for STT (need at least 1GB free)
        available_vram = get_available_vram_mb()
        min_vram_mb = 1000  # 1GB minimum for Whisper

        # Check both VRAM availability and cuDNN availability
        cudnn_available = _check_cudnn_available()

        if (
            torch.cuda.is_available()
            and available_vram >= min_vram_mb
            and cudnn_available
        ):
            device = "cuda"
            # Use float16 for GPU - best balance of speed and accuracy
            # int8_float16 is slightly faster but may reduce accuracy
            compute_type = "float16"
        else:
            device = "cpu"
            # int8 on CPU is much faster than float32 with minimal accuracy loss
            compute_type = "int8"
            if torch.cuda.is_available():
                if not cudnn_available:
                    logging.warning(
                        "[STT] cuDNN not available, using CPU for Whisper (faster-whisper requires cuDNN for GPU)"
                    )
                elif available_vram < min_vram_mb:
                    logging.debug(
                        f"[STT] Only {available_vram:.0f}MB VRAM available, using CPU (need {min_vram_mb}MB)"
                    )

        # Get optimal CPU threads (only used when device is CPU)
        cpu_threads = _get_optimal_cpu_threads() if device == "cpu" else 0

        logging.info(
            f"[STT] Loading Whisper {model} on {device} with compute_type={compute_type}"
        )
        self.device = device
        self.model_name = model
        self.compute_type = compute_type

        try:
            self.w = WhisperModel(
                model,
                download_root="models",
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
            # Create batched inference pipeline for faster batch processing
            self.batched_model = (
                BatchedInferencePipeline(model=self.w)
                if BatchedInferencePipeline
                else self.w
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError, OSError) as e:
            error_str = str(e).lower()
            # Check for various GPU-related errors including cuDNN issues
            is_gpu_error = any(
                x in error_str
                for x in [
                    "out of memory",
                    "cuda",
                    "cudnn",
                    "libcudnn",
                    "invalid handle",
                ]
            )
            if is_gpu_error and device == "cuda":
                logging.warning(f"[STT] GPU loading failed, falling back to CPU: {e}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = "cpu"
                self.compute_type = "int8"
                cpu_threads = _get_optimal_cpu_threads()
                self.w = WhisperModel(
                    model,
                    download_root="models",
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=cpu_threads,
                )
                self.batched_model = (
                    BatchedInferencePipeline(model=self.w)
                    if BatchedInferencePipeline
                    else self.w
                )
            else:
                raise

        self.audio = pyaudio.PyAudio()
        self.wake_functions = wake_functions

    def _diarize_segments(
        self, file_path, segments, num_speakers=None, session_id=None
    ):
        """
        Assign speaker labels to transcription segments using audio feature clustering.

        Uses MFCC features from librosa and agglomerative clustering from scipy
        to identify different speakers in the audio.

        When a session_id is provided, speaker voice prints (MFCC centroids) are
        persisted so that the same physical speaker receives a consistent UUID-based
        ID across multiple chunks within the same recording session.

        Args:
            file_path: Path to the audio file
            segments: List of segment dicts with 'start' and 'end' keys
            num_speakers: Optional number of speakers (auto-detect if None)
            session_id: Optional session identifier for cross-chunk speaker
                        consistency. When provided, speakers are assigned
                        persistent UUIDs instead of SPEAKER_00 numbering.

        Returns:
            segments with 'speaker' key added to each segment
        """
        import librosa
        from scipy.cluster.hierarchy import linkage, fcluster, inconsistent
        from scipy.spatial.distance import pdist

        try:
            audio, sr = librosa.load(file_path, sr=16000)
        except Exception as e:
            logging.warning(f"[STT] Failed to load audio for diarization: {e}")
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"
            return segments

        segment_features = []
        valid_indices = []

        for i, seg in enumerate(segments):
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            segment_audio = audio[start_sample:end_sample]

            # Skip very short segments (< 100ms)
            if len(segment_audio) < int(sr * 0.1):
                continue

            try:
                mfccs = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=20)
                feature = np.concatenate(
                    [np.mean(mfccs, axis=1), np.std(mfccs, axis=1)]
                )
                segment_features.append(feature)
                valid_indices.append(i)
            except Exception as e:
                logging.debug(f"[STT] Failed to extract features for segment {i}: {e}")

        if len(segment_features) < 2:
            # Only one segment — try to match against stored voice prints
            if session_id and len(segment_features) == 1:
                spk_uuid = _voice_print_store.match_centroid(
                    session_id, segment_features[0]
                )
                if spk_uuid is None:
                    spk_uuid = uuid.uuid4().hex[:12]
                    _voice_print_store.update_speaker(
                        session_id, spk_uuid, segment_features[0]
                    )
                else:
                    _voice_print_store.update_speaker(
                        session_id, spk_uuid, segment_features[0]
                    )
                for seg in segments:
                    seg["speaker"] = f"SPK_{spk_uuid}"
            else:
                for seg in segments:
                    seg["speaker"] = "SPEAKER_00"
            return segments

        features = np.array(segment_features)
        distances = pdist(features, metric="cosine")
        linkage_matrix = linkage(distances, method="average")

        if num_speakers and num_speakers > 0:
            labels = fcluster(linkage_matrix, t=num_speakers, criterion="maxclust")
        else:
            # Auto-detect number of speakers using inconsistency coefficient
            depth = min(3, len(segment_features) - 1)
            incon = inconsistent(linkage_matrix, d=depth)
            # Count merges with high inconsistency from the top of the tree
            detected = 1
            for i in range(len(incon) - 1, -1, -1):
                if incon[i, 3] > 1.0:
                    detected += 1
                else:
                    break
            detected = max(1, min(detected, len(segment_features)))
            logging.info(f"[STT] Auto-detected {detected} speaker(s)")
            labels = fcluster(linkage_matrix, t=max(1, detected), criterion="maxclust")

        if session_id:
            # --- Voice-print-based persistent speaker IDs ---
            # Compute a centroid for each cluster
            unique_labels = set(labels)
            cluster_centroids = {}
            cluster_members = {}
            for feat, lbl in zip(segment_features, labels):
                lbl_int = int(lbl)
                if lbl_int not in cluster_centroids:
                    cluster_centroids[lbl_int] = []
                cluster_centroids[lbl_int].append(feat)
            for lbl_int, feats in cluster_centroids.items():
                cluster_centroids[lbl_int] = np.mean(feats, axis=0)
                cluster_members[lbl_int] = len(feats)

            # Match each cluster centroid against stored voice prints
            cluster_to_uuid = {}
            for lbl_int in sorted(unique_labels):
                centroid = cluster_centroids[int(lbl_int)]
                matched_uuid = _voice_print_store.match_centroid(session_id, centroid)
                if matched_uuid is None:
                    # New speaker — assign a fresh UUID
                    matched_uuid = uuid.uuid4().hex[:12]
                cluster_to_uuid[int(lbl_int)] = matched_uuid
                # Update the stored voice print with the new centroid data
                _voice_print_store.update_speaker(
                    session_id,
                    matched_uuid,
                    centroid,
                    count=cluster_members[int(lbl_int)],
                )

            # Assign persistent speaker IDs to segments
            for idx, label in zip(valid_indices, labels):
                spk_uuid = cluster_to_uuid[int(label)]
                segments[idx]["speaker"] = f"SPK_{spk_uuid}"

            for seg in segments:
                if "speaker" not in seg:
                    seg["speaker"] = f"SPK_{cluster_to_uuid.get(1, 'unknown')}"

            n_stored = len(_voice_print_store.get_session_prints(session_id))
            logging.info(
                f"[STT] Session {session_id[:8]}... has {n_stored} stored voice print(s)"
            )
        else:
            # No session — use plain SPEAKER_XX numbering
            for idx, label in zip(valid_indices, labels):
                segments[idx]["speaker"] = f"SPEAKER_{max(0, int(label) - 1):02d}"

            for seg in segments:
                if "speaker" not in seg:
                    seg["speaker"] = "SPEAKER_00"

        return segments

    async def transcribe_audio(
        self,
        base64_audio,
        audio_format="wav",
        language=None,
        prompt=None,
        temperature=0.0,
        translate=False,
        return_segments=False,
        beam_size=1,  # Default to 1 for speed; use 5 for higher accuracy if needed
        condition_on_previous_text=True,
        use_batched=False,  # Use batched inference for longer audio files
        batch_size=8,  # Batch size for batched inference
        enable_diarization=False,
        num_speakers=None,
        session_id=None,
    ):
        """
        Transcribe audio to text using faster-whisper.

        Args:
            base64_audio: Base64 encoded audio data
            audio_format: Audio format (wav, mp3, etc.)
            language: Language code (e.g., 'en', 'es') or None for auto-detection
            prompt: Initial prompt to guide transcription
            temperature: Sampling temperature (0.0 = greedy decoding)
            translate: If True, translate to English
            return_segments: If True, return detailed segment info
            beam_size: Beam size for decoding (1 = greedy, 5 = more accurate but slower)
            condition_on_previous_text: Use previous text as context
            use_batched: Use BatchedInferencePipeline for faster processing of longer audio
            batch_size: Batch size when using batched inference
            enable_diarization: If True, perform speaker diarization on segments
            num_speakers: Optional number of speakers for diarization (auto-detect if None)
            session_id: Optional session identifier for persistent speaker IDs across
                        multiple transcription chunks. When provided, speakers receive
                        consistent UUID-based IDs (e.g. SPK_a1b2c3d4e5f6) instead of
                        SPEAKER_00 numbering.

        Returns:
            Transcribed text or dict with segments if return_segments=True
        """
        # Handle None or empty audio_format - default to wav
        if not audio_format:
            audio_format = "wav"
        if "/" in audio_format:
            audio_format = audio_format.split("/")[1]
        filename = f"{uuid.uuid4().hex}.wav"
        file_path = os.path.join(os.getcwd(), "outputs", filename)
        audio_data = base64.b64decode(base64_audio)
        with open(file_path, "wb") as audio_file:
            audio_file.write(audio_data)
        if not os.path.exists(file_path):
            raise RuntimeError(f"Failed to load audio.")

        transcribe_info = None
        try:
            # Optimized VAD parameters for better speech detection
            vad_params = {
                "min_silence_duration_ms": 500,  # Shorter silence detection
                "speech_pad_ms": 200,  # Padding around speech
            }

            if use_batched:
                # Use batched inference for potentially faster processing
                # Batched inference uses VAD by default and processes in parallel
                segments, transcribe_info = self.batched_model.transcribe(
                    file_path,
                    task="transcribe" if not translate else "translate",
                    language=language,
                    initial_prompt=prompt,
                    beam_size=beam_size,
                    temperature=temperature if temperature > 0 else [0.0],
                    batch_size=batch_size,
                    vad_parameters=vad_params,
                )
            else:
                # Standard inference - good for shorter audio
                segments, transcribe_info = self.w.transcribe(
                    file_path,
                    task="transcribe" if not translate else "translate",
                    vad_filter=True,
                    vad_parameters=vad_params,
                    initial_prompt=prompt,
                    language=language,
                    temperature=temperature,
                    beam_size=beam_size,
                    condition_on_previous_text=condition_on_previous_text,
                )
            segments = list(segments)
        except (torch.cuda.OutOfMemoryError, RuntimeError, OSError) as e:
            error_str = str(e).lower()
            # Check for various GPU-related errors including cuDNN issues
            is_gpu_error = any(
                x in error_str
                for x in [
                    "out of memory",
                    "cuda",
                    "cudnn",
                    "libcudnn",
                    "invalid handle",
                ]
            )
            if is_gpu_error and self.device == "cuda":
                logging.warning(
                    f"[STT] GPU error during transcription, reloading on CPU: {e}"
                )
                # Free GPU memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Reload model on CPU with optimized settings
                del self.w
                if hasattr(self, "batched_model"):
                    del self.batched_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.device = "cpu"
                self.compute_type = "int8"
                cpu_threads = _get_optimal_cpu_threads()
                self.w = WhisperModel(
                    self.model_name,
                    download_root="models",
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=cpu_threads,
                )
                self.batched_model = (
                    BatchedInferencePipeline(model=self.w)
                    if BatchedInferencePipeline
                    else self.w
                )
                logging.debug(
                    "[STT] Model reloaded on CPU with int8, retrying transcription..."
                )

                # Optimized VAD parameters
                vad_params = {
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200,
                }

                # Retry on CPU
                segments, transcribe_info = self.w.transcribe(
                    file_path,
                    task="transcribe" if not translate else "translate",
                    vad_filter=True,
                    vad_parameters=vad_params,
                    initial_prompt=prompt,
                    language=language,
                    temperature=temperature,
                    beam_size=beam_size,
                    condition_on_previous_text=condition_on_previous_text,
                )
                segments = list(segments)
            else:
                os.remove(file_path)
                raise

        # Build full text and segment data
        user_input = ""
        segment_data = []
        for i, segment in enumerate(segments):
            user_input += segment.text
            segment_data.append(
                {
                    "id": i,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
            )

        # Perform speaker diarization if requested
        if enable_diarization and segment_data:
            logging.info(
                f"[STT] Performing speaker diarization"
                f"{' (session=' + session_id[:8] + '...)' if session_id else ''}..."
            )
            segment_data = self._diarize_segments(
                file_path,
                segment_data,
                num_speakers=num_speakers,
                session_id=session_id,
            )
            # Build speaker-attributed text
            user_input = ""
            current_speaker = None
            for seg in segment_data:
                speaker = seg.get("speaker", "SPEAKER_00")
                if speaker != current_speaker:
                    current_speaker = speaker
                    user_input += f"\n[{speaker}]: "
                user_input += seg["text"] + " "
            user_input = user_input.strip()

        logging.debug(f"[STT] Transcribed User Input: {user_input}")
        os.remove(file_path)

        if return_segments:
            detected_language = None
            if transcribe_info and hasattr(transcribe_info, "language"):
                detected_language = transcribe_info.language
            elif language:
                detected_language = language
            return {
                "text": user_input.strip(),
                "segments": segment_data,
                "language": detected_language,
            }
        return user_input

    def listen(self):
        print("Listening for wake word...")
        vad = webrtcvad.Vad(1)
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=320,
        )
        frames = []
        silence_frames = 0
        while True:
            data = stream.read(320)
            frames.append(data)
            is_speech = vad.is_speech(data, 16000)
            if not is_speech:
                silence_frames += 1
                if silence_frames > 1 * 16000 / 320:
                    audio_data = b"".join(frames)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    rms = np.sqrt(np.mean(audio_np**2))
                    if rms > 500:
                        buffer = BytesIO()
                        with wave.open(buffer, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                            wf.setframerate(16000)
                            wf.writeframes(b"".join(frames))
                        wav_buffer = buffer.getvalue()
                        base64_audio = base64.b64encode(wav_buffer).decode()
                        thread = threading.Thread(
                            target=self.transcribe_audio,
                            args=(base64_audio),
                        )
                        thread.start()
                    frames = []  # Clear frames after processing
                    silence_frames = 0
            else:
                silence_frames = 0
