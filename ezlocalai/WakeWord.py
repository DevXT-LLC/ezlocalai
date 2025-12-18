"""
Wake Word Detection Module for ezlocalai

Provides wake word model training, inference, and management functionality.
When VOICE_SERVER=true, these features are enabled alongside TTS/STT.

Based on the WakeWord project: https://github.com/Josh-XT/wakeword
"""

import os
import io
import json
import logging
import uuid
import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio processing constants
SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 512
HOP_LENGTH = 160  # 10ms at 16kHz
WIN_LENGTH = 400  # 25ms at 16kHz
MAX_AUDIO_LENGTH = 1.5  # seconds
MAX_FRAMES = int(MAX_AUDIO_LENGTH * SAMPLE_RATE / HOP_LENGTH)


# =============================================================================
# Data Classes and Enums
# =============================================================================


class JobStatus(Enum):
    PENDING = "pending"
    GENERATING_SAMPLES = "generating_samples"
    AUGMENTING = "augmenting"
    TRAINING = "training"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TTSEngine(Enum):
    GTTS = "gtts"
    EDGE = "edge"
    CHATTERBOX = "chatterbox"


@dataclass
class ModelConfig:
    """Configuration for wake word model."""

    word: str
    n_mfcc: int = N_MFCC
    sample_rate: int = SAMPLE_RATE
    max_length_sec: float = MAX_AUDIO_LENGTH
    model_type: str = "cnn"  # "cnn" or "gru"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelConfig":
        return cls(**data)


@dataclass
class TTSSample:
    """Represents a generated TTS sample."""

    audio_data: bytes
    sample_rate: int
    engine: TTSEngine
    voice: str
    word: str
    variation: str


@dataclass
class TrainingJob:
    """Represents a wake word training job."""

    job_id: str
    word: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress: float = 0.0
    current_stage: str = ""
    error_message: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    model_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "word": self.word,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "progress": self.progress,
            "current_stage": self.current_stage,
            "error_message": self.error_message,
            "estimated_completion": (
                self.estimated_completion.isoformat()
                if self.estimated_completion
                else None
            ),
            "model_path": self.model_path,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingJob":
        return cls(
            job_id=data["job_id"],
            word=data["word"],
            status=JobStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            progress=data.get("progress", 0.0),
            current_stage=data.get("current_stage", ""),
            error_message=data.get("error_message"),
            estimated_completion=(
                datetime.fromisoformat(data["estimated_completion"])
                if data.get("estimated_completion")
                else None
            ),
            model_path=data.get("model_path"),
            metrics=data.get("metrics", {}),
        )


# =============================================================================
# Audio Feature Extraction
# =============================================================================


class AudioFeatureExtractor:
    """Extract MFCC features from audio for wake word detection."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mfcc: int = N_MFCC,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        win_length: int = WIN_LENGTH,
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": win_length,
                "n_mels": 80,
            },
        )

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features from waveform."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mfcc = self.mfcc_transform(waveform)

        if mfcc.dim() == 3:
            mfcc = mfcc.squeeze(0)

        return mfcc

    def load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load and preprocess audio file."""
        waveform, sr = torchaudio.load(str(audio_path))

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        waveform = waveform / (waveform.abs().max() + 1e-8)
        return waveform

    def load_audio_bytes(self, audio_bytes: bytes) -> torch.Tensor:
        """Load audio from bytes."""
        buffer = io.BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(buffer)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        waveform = waveform / (waveform.abs().max() + 1e-8)
        return waveform


# =============================================================================
# Dataset
# =============================================================================


class WakeWordDataset(Dataset):
    """Dataset for wake word training."""

    def __init__(
        self,
        positive_samples: List[Tuple[bytes, Dict]],
        negative_samples: List[Tuple[bytes, Dict]],
        feature_extractor: AudioFeatureExtractor,
        max_frames: int = MAX_FRAMES,
    ):
        self.feature_extractor = feature_extractor
        self.max_frames = max_frames

        self.samples = []
        for audio_bytes, metadata in positive_samples:
            self.samples.append((audio_bytes, 1, metadata))
        for audio_bytes, metadata in negative_samples:
            self.samples.append((audio_bytes, 0, metadata))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_bytes, label, metadata = self.samples[idx]

        waveform = self.feature_extractor.load_audio_bytes(audio_bytes)
        features = self.feature_extractor.extract_features(waveform)

        if features.shape[1] < self.max_frames:
            padding = torch.zeros(
                features.shape[0], self.max_frames - features.shape[1]
            )
            features = torch.cat([features, padding], dim=1)
        else:
            features = features[:, : self.max_frames]

        return features, torch.tensor(label, dtype=torch.float32)


# =============================================================================
# Neural Network Models
# =============================================================================


class WakeWordCNN(nn.Module):
    """Compact CNN for wake word detection (~1.7MB)."""

    def __init__(
        self,
        n_mfcc: int = N_MFCC,
        max_frames: int = MAX_FRAMES,
        hidden_size: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.n_mfcc = n_mfcc
        self.max_frames = max_frames

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        h = n_mfcc // 8
        w = max_frames // 8
        flat_size = 64 * h * w

        self.fc1 = nn.Linear(flat_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x.squeeze(-1)


class WakeWordGRU(nn.Module):
    """Compact GRU for wake word detection (~460KB)."""

    def __init__(
        self,
        n_mfcc: int = N_MFCC,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.n_mfcc = n_mfcc
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=n_mfcc,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        output, _ = self.gru(x)
        x = output[:, -1, :]
        x = torch.sigmoid(self.fc(x))
        return x.squeeze(-1)


# =============================================================================
# TTS Sample Generator
# =============================================================================


class SampleGenerator:
    """Multi-engine TTS sample generator for wake word training."""

    EDGE_VOICES = [
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-US-AriaNeural",
        "en-US-DavisNeural",
        "en-US-AmberNeural",
        "en-US-AndrewNeural",
        "en-US-BrandonNeural",
        "en-US-ChristopherNeural",
        "en-US-CoraNeural",
        "en-US-ElizabethNeural",
        "en-US-EricNeural",
        "en-US-JacobNeural",
        "en-US-MichelleNeural",
        "en-US-MonicaNeural",
        "en-US-RogerNeural",
        "en-US-SaraNeural",
        "en-US-SteffanNeural",
        "en-GB-SoniaNeural",
        "en-GB-RyanNeural",
        "en-GB-LibbyNeural",
        "en-AU-NatashaNeural",
        "en-AU-WilliamNeural",
        "en-IN-NeerjaNeural",
        "en-IN-PrabhatNeural",
    ]

    GTTS_LANGUAGES = ["en", "en-au", "en-uk", "en-us", "en-ca", "en-in"]

    def __init__(
        self,
        output_dir: Path,
        enable_gtts: bool = True,
        enable_edge: bool = True,
        enable_chatterbox: bool = False,
        chatterbox_model=None,
        chatterbox_voices_dir: Optional[Path] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_gtts = enable_gtts
        self.enable_edge = enable_edge
        self.enable_chatterbox = enable_chatterbox
        self.chatterbox_model = chatterbox_model
        self.chatterbox_voices_dir = chatterbox_voices_dir

        self.executor = ThreadPoolExecutor(max_workers=8)
        self._init_engines()

    def _init_engines(self):
        """Initialize available TTS engines."""
        self.available_engines = []

        if self.enable_gtts:
            try:
                from gtts import gTTS

                self.available_engines.append(TTSEngine.GTTS)
                logger.info("gTTS engine initialized for wake word training")
            except ImportError:
                logger.warning("gTTS not available for wake word training")

        if self.enable_edge:
            try:
                import edge_tts

                self.available_engines.append(TTSEngine.EDGE)
                logger.info("Edge TTS engine initialized for wake word training")
            except ImportError:
                logger.warning("edge-tts not available for wake word training")

        if self.enable_chatterbox and self.chatterbox_model:
            self.available_engines.append(TTSEngine.CHATTERBOX)
            logger.info("Chatterbox TTS engine initialized for wake word training")

    async def generate_gtts_sample(
        self, word: str, lang: str = "en"
    ) -> Optional[TTSSample]:
        """Generate a sample using Google TTS."""
        try:
            from gtts import gTTS
            from pydub import AudioSegment

            tts = gTTS(text=word, lang=lang, slow=False)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)

            audio = AudioSegment.from_mp3(buffer)
            audio = audio.set_frame_rate(16000).set_channels(1)

            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            return TTSSample(
                audio_data=wav_buffer.read(),
                sample_rate=16000,
                engine=TTSEngine.GTTS,
                voice=lang,
                word=word,
                variation="normal",
            )
        except Exception as e:
            logger.error(f"gTTS generation failed: {e}")
            return None

    async def generate_edge_sample(self, word: str, voice: str) -> Optional[TTSSample]:
        """Generate a sample using Microsoft Edge TTS."""
        try:
            import edge_tts
            from pydub import AudioSegment

            communicate = edge_tts.Communicate(word, voice)
            buffer = io.BytesIO()

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buffer.write(chunk["data"])

            buffer.seek(0)

            audio = AudioSegment.from_mp3(buffer)
            audio = audio.set_frame_rate(16000).set_channels(1)

            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            return TTSSample(
                audio_data=wav_buffer.read(),
                sample_rate=16000,
                engine=TTSEngine.EDGE,
                voice=voice,
                word=word,
                variation="normal",
            )
        except Exception as e:
            logger.error(f"Edge TTS generation failed for voice {voice}: {e}")
            return None

    async def generate_chatterbox_sample(
        self, word: str, voice_file: Path
    ) -> Optional[TTSSample]:
        """Generate a sample using Chatterbox TTS with voice cloning."""
        try:
            if not self.chatterbox_model:
                return None

            from pydub import AudioSegment

            wav = self.chatterbox_model.generate(
                word, audio_prompt_path=str(voice_file)
            )

            temp_path = self.output_dir / f"temp_cb_{uuid.uuid4().hex}.wav"

            if isinstance(wav, torch.Tensor):
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                torchaudio.save(str(temp_path), wav.cpu(), self.chatterbox_model.sr)

            audio = AudioSegment.from_wav(str(temp_path))
            audio = audio.set_frame_rate(16000).set_channels(1)

            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            temp_path.unlink(missing_ok=True)

            return TTSSample(
                audio_data=wav_buffer.read(),
                sample_rate=16000,
                engine=TTSEngine.CHATTERBOX,
                voice=voice_file.stem,
                word=word,
                variation="cloned",
            )
        except Exception as e:
            logger.error(f"Chatterbox generation failed: {e}")
            return None

    async def generate_samples(
        self,
        word: str,
        target_count: int = 500,
        progress_callback: Optional[callable] = None,
    ) -> List[TTSSample]:
        """Generate diverse TTS samples for a wake word."""
        samples = []
        tasks = []

        enabled_count = len(self.available_engines)
        if enabled_count == 0:
            raise RuntimeError("No TTS engines available for wake word training")

        samples_per_engine = target_count // enabled_count

        if TTSEngine.GTTS in self.available_engines:
            for lang in self.GTTS_LANGUAGES[:samples_per_engine]:
                tasks.append(self.generate_gtts_sample(word, lang))

        if TTSEngine.EDGE in self.available_engines:
            for voice in self.EDGE_VOICES[:samples_per_engine]:
                tasks.append(self.generate_edge_sample(word, voice))

        if (
            TTSEngine.CHATTERBOX in self.available_engines
            and self.chatterbox_voices_dir
        ):
            voice_files = list(self.chatterbox_voices_dir.glob("*.wav"))
            for voice_file in voice_files[:samples_per_engine]:
                tasks.append(self.generate_chatterbox_sample(word, voice_file))

        logger.info(f"Generating {len(tasks)} base samples for '{word}'...")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, TTSSample):
                samples.append(result)
            elif isinstance(result, Exception):
                logger.debug(f"Sample generation failed: {result}")

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, len(tasks), "generating")

        logger.info(f"Generated {len(samples)} base samples")
        return samples


# =============================================================================
# Audio Augmentation
# =============================================================================


class AudioAugmenter:
    """Audio augmentation for expanding training dataset variety."""

    def augment_sample(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> List[Tuple[bytes, str]]:
        """Apply multiple augmentations to a single sample."""
        from pydub import AudioSegment

        augmented = []
        audio = AudioSegment.from_wav(io.BytesIO(audio_data))

        # Speed variations
        for speed in [0.85, 0.9, 1.1, 1.15]:
            aug = self._change_speed(audio, speed)
            augmented.append((self._to_bytes(aug), f"speed_{speed}"))

        # Pitch variations
        for semitones in [-2, -1, 1, 2]:
            aug = self._change_pitch(audio, semitones)
            augmented.append((self._to_bytes(aug), f"pitch_{semitones}"))

        # Volume variations
        for db in [-6, -3, 3, 6]:
            aug = audio + db
            augmented.append((self._to_bytes(aug), f"volume_{db}db"))

        # Add background noise
        for noise_level in [0.005, 0.01, 0.02]:
            aug = self._add_noise(audio, noise_level)
            augmented.append((self._to_bytes(aug), f"noise_{noise_level}"))

        # Reverb simulation
        aug = self._add_reverb(audio)
        augmented.append((self._to_bytes(aug), "reverb"))

        return augmented

    def _change_speed(self, audio, speed: float):
        """Change playback speed."""
        from pydub import AudioSegment

        new_sample_rate = int(audio.frame_rate * speed)
        return audio._spawn(
            audio.raw_data, overrides={"frame_rate": new_sample_rate}
        ).set_frame_rate(audio.frame_rate)

    def _change_pitch(self, audio, semitones: int):
        """Shift pitch by semitones."""
        new_sample_rate = int(audio.frame_rate * (2 ** (semitones / 12)))
        pitched = audio._spawn(
            audio.raw_data, overrides={"frame_rate": new_sample_rate}
        )
        return pitched.set_frame_rate(audio.frame_rate)

    def _add_noise(self, audio, noise_level: float):
        """Add white noise to audio."""
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        noise = np.random.normal(0, noise_level * 32767, len(samples))
        noisy = np.clip(samples + noise, -32768, 32767).astype(np.int16)
        return audio._spawn(noisy.tobytes())

    def _add_reverb(self, audio, decay: float = 0.3):
        """Add simple reverb effect."""
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        delay_samples = int(0.03 * audio.frame_rate)

        reverb = np.zeros(len(samples) + delay_samples)
        reverb[: len(samples)] = samples
        reverb[delay_samples:] += samples * decay

        reverb = np.clip(reverb[: len(samples)], -32768, 32767).astype(np.int16)
        return audio._spawn(reverb.tobytes())

    def _to_bytes(self, audio) -> bytes:
        """Convert AudioSegment to WAV bytes."""
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        return buffer.read()

    async def augment_dataset(
        self,
        samples: List[TTSSample],
        augmentations_per_sample: int = 5,
        progress_callback: Optional[callable] = None,
    ) -> List[Tuple[bytes, Dict]]:
        """Augment an entire dataset of samples."""
        augmented_dataset = []

        # Include original samples
        for sample in samples:
            augmented_dataset.append(
                (
                    sample.audio_data,
                    {
                        "engine": sample.engine.value,
                        "voice": sample.voice,
                        "word": sample.word,
                        "augmentation": "original",
                    },
                )
            )

        total = len(samples)
        for i, sample in enumerate(samples):
            augmentations = self.augment_sample(sample.audio_data, sample.sample_rate)
            selected = augmentations[:augmentations_per_sample]

            for aug_data, aug_name in selected:
                augmented_dataset.append(
                    (
                        aug_data,
                        {
                            "engine": sample.engine.value,
                            "voice": sample.voice,
                            "word": sample.word,
                            "augmentation": aug_name,
                        },
                    )
                )

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, total, "augmenting")

        logger.info(
            f"Created {len(augmented_dataset)} total samples after augmentation"
        )
        return augmented_dataset


# =============================================================================
# Negative Sample Generator
# =============================================================================


class NegativeSampleGenerator:
    """Generate negative samples for training."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    def generate_noise_samples(
        self, count: int, duration: float = 1.0
    ) -> List[Tuple[bytes, Dict]]:
        """Generate various types of noise samples."""
        import wave

        samples = []

        for i in range(count):
            noise_type = np.random.choice(["white", "pink", "brown", "silence"])
            num_samples = int(duration * self.sample_rate)

            if noise_type == "white":
                audio = np.random.randn(num_samples) * 0.1
            elif noise_type == "pink":
                white = np.random.randn(num_samples)
                b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
                a = [1, -2.494956002, 2.017265875, -0.522189400]
                from scipy.signal import lfilter

                audio = lfilter(b, a, white) * 0.3
            elif noise_type == "brown":
                audio = np.cumsum(np.random.randn(num_samples)) * 0.001
                audio = audio / (np.abs(audio).max() + 1e-8) * 0.2
            else:
                audio = np.random.randn(num_samples) * 0.001

            audio = (audio * 32767).astype(np.int16)

            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio.tobytes())

            buffer.seek(0)
            samples.append((buffer.read(), {"type": "noise", "noise_type": noise_type}))

        return samples

    async def generate_similar_words(
        self, target_word: str, tts_generator: SampleGenerator, count: int = 50
    ) -> List[Tuple[bytes, Dict]]:
        """Generate samples of similar-sounding words."""
        similar_patterns = [
            "hey",
            "hay",
            "hi",
            "he",
            "huh",
            "ok",
            "okay",
            "oh",
            "ow",
            "yeah",
            "yes",
            "yep",
            "no",
            "nope",
            "the",
            "a",
            "is",
            "it",
            "to",
            "and",
            "what",
            "that",
            "this",
            "there",
            "one",
            "two",
            "three",
            "four",
            "five",
        ]

        if len(target_word) > 2:
            similar_patterns.extend(
                [
                    target_word[:-1],
                    target_word[1:],
                    target_word + "s",
                    target_word + "ing",
                ]
            )

        samples = []
        for word in similar_patterns[:count]:
            try:
                tts_samples = await tts_generator.generate_samples(word, target_count=2)
                for sample in tts_samples:
                    samples.append(
                        (sample.audio_data, {"type": "similar_word", "word": word})
                    )
            except Exception as e:
                logger.debug(f"Could not generate sample for '{word}': {e}")

        return samples


# =============================================================================
# Wake Word Trainer
# =============================================================================


class WakeWordTrainer:
    """Trainer for wake word models."""

    def __init__(
        self,
        config: ModelConfig,
        device: Optional[str] = None,
    ):
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.feature_extractor = AudioFeatureExtractor(
            sample_rate=config.sample_rate,
            n_mfcc=config.n_mfcc,
        )

        if config.model_type == "gru":
            self.model = WakeWordGRU(
                n_mfcc=config.n_mfcc,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
            )
        else:
            self.model = WakeWordCNN(
                n_mfcc=config.n_mfcc,
                hidden_size=config.hidden_size,
                dropout=config.dropout,
            )

        self.model = self.model.to(self.device)

    def train(
        self,
        positive_samples: List[Tuple[bytes, Dict]],
        negative_samples: List[Tuple[bytes, Dict]],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """Train the wake word model."""
        dataset = WakeWordDataset(
            positive_samples,
            negative_samples,
            self.feature_extractor,
        )

        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
        criterion = nn.BCELoss()

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        best_model_state = None

        logger.info(f"Starting training on {self.device}")
        logger.info(f"Training samples: {train_size}, Validation samples: {val_size}")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predictions = (outputs > 0.5).float()
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(features)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if progress_callback:
                progress_callback(
                    epoch + 1,
                    epochs,
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                    },
                )

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        logger.info(f"Training complete. Best validation loss: {best_val_loss:.4f}")
        return history

    def predict(self, audio_bytes: bytes) -> Tuple[bool, float]:
        """Predict if audio contains the wake word."""
        self.model.eval()

        with torch.no_grad():
            waveform = self.feature_extractor.load_audio_bytes(audio_bytes)
            features = self.feature_extractor.extract_features(waveform)

            if features.shape[1] < MAX_FRAMES:
                padding = torch.zeros(features.shape[0], MAX_FRAMES - features.shape[1])
                features = torch.cat([features, padding], dim=1)
            else:
                features = features[:, :MAX_FRAMES]

            features = features.unsqueeze(0).to(self.device)
            confidence = self.model(features).item()

            return confidence > 0.5, confidence

    def save(self, output_dir: Path) -> Dict[str, Path]:
        """Save model in multiple formats."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save PyTorch model
        torch_path = output_dir / "model.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.to_dict(),
            },
            torch_path,
        )
        saved_files["pytorch"] = torch_path

        # Save config
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        saved_files["config"] = config_path

        # Export to ONNX
        try:
            onnx_path = output_dir / "model.onnx"
            dummy_input = torch.randn(1, self.config.n_mfcc, MAX_FRAMES).to(self.device)

            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )
            saved_files["onnx"] = onnx_path
            logger.info(f"Saved ONNX model to {onnx_path}")
        except Exception as e:
            logger.warning(f"Could not export ONNX: {e}")

        # Export to TFLite
        try:
            tflite_path = self._export_tflite(output_dir)
            if tflite_path:
                saved_files["tflite"] = tflite_path
        except Exception as e:
            logger.warning(f"Could not export TFLite: {e}")

        logger.info(f"Model saved to {output_dir}")
        return saved_files

    def _export_tflite(self, output_dir: Path) -> Optional[Path]:
        """Export model to TensorFlow Lite format."""
        try:
            import tensorflow as tf
            import onnx
            from onnx_tf.backend import prepare

            onnx_path = output_dir / "model.onnx"
            if not onnx_path.exists():
                return None

            onnx_model = onnx.load(str(onnx_path))
            tf_rep = prepare(onnx_model)

            saved_model_dir = output_dir / "saved_model"
            tf_rep.export_graph(str(saved_model_dir))

            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

            tflite_model = converter.convert()

            tflite_path = output_dir / "model.tflite"
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)

            logger.info(f"Saved TFLite model to {tflite_path}")
            return tflite_path

        except ImportError:
            logger.warning("TensorFlow not available for TFLite export")
            return None
        except Exception as e:
            logger.warning(f"TFLite export failed: {e}")
            return None

    @classmethod
    def load(cls, model_dir: Path, device: Optional[str] = None) -> "WakeWordTrainer":
        """Load a trained model."""
        model_dir = Path(model_dir)

        config_path = model_dir / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = ModelConfig.from_dict(config_dict)
        trainer = cls(config, device=device)

        torch_path = model_dir / "model.pt"
        checkpoint = torch.load(
            torch_path, map_location=trainer.device, weights_only=False
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])

        return trainer


# =============================================================================
# Wake Word Manager (Job Management)
# =============================================================================


class WakeWordManager:
    """
    Manages wake word model training, inference, and storage.

    This is the main interface for the wake word functionality in ezlocalai.
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        samples_dir: Optional[Path] = None,
        jobs_dir: Optional[Path] = None,
        chatterbox_model=None,
        voices_dir: Optional[Path] = None,
    ):
        base_dir = Path(os.getcwd())

        self.models_dir = (
            Path(models_dir) if models_dir else base_dir / "wakeword_models"
        )
        self.samples_dir = (
            Path(samples_dir) if samples_dir else base_dir / "wakeword_samples"
        )
        self.jobs_dir = Path(jobs_dir) if jobs_dir else base_dir / "wakeword_jobs"
        self.voices_dir = Path(voices_dir) if voices_dir else base_dir / "voices"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self.chatterbox_model = chatterbox_model

        # Job tracking
        self.jobs: Dict[str, TrainingJob] = {}
        self.word_to_job: Dict[str, str] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.lock = threading.Lock()

        # Loaded models cache
        self._loaded_models: Dict[str, WakeWordTrainer] = {}
        self._model_lock = threading.Lock()

        self._load_jobs()

    def _load_jobs(self):
        """Load existing jobs from disk."""
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                with open(job_file, "r") as f:
                    job_data = json.load(f)
                job = TrainingJob.from_dict(job_data)
                self.jobs[job.job_id] = job

                if job.status not in [
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                ]:
                    self.word_to_job[job.word.lower()] = job.job_id

                logger.info(
                    f"Loaded wake word job {job.job_id} for word '{job.word}' "
                    f"(status: {job.status.value})"
                )
            except Exception as e:
                logger.error(f"Failed to load job from {job_file}: {e}")

    def _save_job(self, job: TrainingJob):
        """Save job state to disk."""
        job_file = self.jobs_dir / f"{job.job_id}.json"
        with open(job_file, "w") as f:
            json.dump(job.to_dict(), f, indent=2)

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def get_job_for_word(self, word: str) -> Optional[TrainingJob]:
        """Get active job for a word."""
        word_lower = word.lower()
        job_id = self.word_to_job.get(word_lower)
        if job_id:
            return self.jobs.get(job_id)
        return None

    def get_model_for_word(self, word: str) -> Optional[Path]:
        """Get the model directory for a word if it exists."""
        word_lower = word.lower().replace(" ", "_")
        model_dir = self.models_dir / word_lower

        if model_dir.exists() and (model_dir / "model.pt").exists():
            return model_dir
        return None

    def list_jobs(
        self, status: Optional[JobStatus] = None, limit: int = 100
    ) -> List[TrainingJob]:
        """List jobs, optionally filtered by status."""
        jobs = list(self.jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available trained wake word models."""
        models = []

        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "model.pt").exists():
                config_path = model_dir / "config.json"
                config = {}
                if config_path.exists():
                    with open(config_path, "r") as f:
                        config = json.load(f)

                files = {}
                for ext in ["pt", "onnx", "tflite"]:
                    file_path = model_dir / f"model.{ext}"
                    if file_path.exists():
                        files[ext] = {
                            "path": str(file_path),
                            "size_bytes": file_path.stat().st_size,
                        }

                models.append(
                    {
                        "word": config.get("word", model_dir.name),
                        "directory": str(model_dir),
                        "config": config,
                        "files": files,
                        "created_at": datetime.fromtimestamp(
                            model_dir.stat().st_mtime
                        ).isoformat(),
                    }
                )

        return models

    async def create_job(
        self,
        word: str,
        sample_count: int = 500,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> TrainingJob:
        """Create a new training job for a word."""
        word_lower = word.lower()

        with self.lock:
            existing_job = self.get_job_for_word(word_lower)
            if existing_job and existing_job.status not in [
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            ]:
                raise ValueError(
                    f"Job already in progress for word '{word}' "
                    f"(job_id: {existing_job.job_id}, status: {existing_job.status.value})"
                )

            job = TrainingJob(
                job_id=str(uuid.uuid4()),
                word=word,
                status=JobStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                estimated_completion=datetime.now() + timedelta(minutes=15),
            )

            self.jobs[job.job_id] = job
            self.word_to_job[word_lower] = job.job_id
            self._save_job(job)

        task = asyncio.create_task(
            self._run_training(job, sample_count, epochs, batch_size)
        )
        self.running_tasks[job.job_id] = task

        logger.info(f"Created wake word training job {job.job_id} for word '{word}'")
        return job

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False

        task = self.running_tasks.get(job_id)
        if task:
            task.cancel()

        job.status = JobStatus.CANCELLED
        job.updated_at = datetime.now()
        self._save_job(job)

        word_lower = job.word.lower()
        if self.word_to_job.get(word_lower) == job_id:
            del self.word_to_job[word_lower]

        logger.info(f"Cancelled wake word job {job_id}")
        return True

    def _update_job(
        self,
        job: TrainingJob,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        current_stage: Optional[str] = None,
        error_message: Optional[str] = None,
        model_path: Optional[str] = None,
        metrics: Optional[Dict] = None,
    ):
        """Update job state."""
        if status:
            job.status = status
        if progress is not None:
            job.progress = progress
        if current_stage:
            job.current_stage = current_stage
        if error_message:
            job.error_message = error_message
        if model_path:
            job.model_path = model_path
        if metrics:
            job.metrics.update(metrics)

        job.updated_at = datetime.now()
        self._save_job(job)

    async def _run_training(
        self,
        job: TrainingJob,
        sample_count: int,
        epochs: int,
        batch_size: int,
    ):
        """Run the full training pipeline."""
        try:
            word = job.word
            word_dir = word.lower().replace(" ", "_")

            # Stage 1: Generate TTS samples
            self._update_job(
                job,
                status=JobStatus.GENERATING_SAMPLES,
                progress=0,
                current_stage="Generating TTS samples",
            )

            generator = SampleGenerator(
                output_dir=self.samples_dir / word_dir,
                enable_gtts=True,
                enable_edge=True,
                enable_chatterbox=self.chatterbox_model is not None,
                chatterbox_model=self.chatterbox_model,
                chatterbox_voices_dir=(
                    self.voices_dir if self.chatterbox_model else None
                ),
            )

            def sample_progress(current, total, stage):
                progress = (current / total) * 25
                self._update_job(job, progress=progress)

            samples = await generator.generate_samples(
                word,
                target_count=sample_count,
                progress_callback=sample_progress,
            )

            if len(samples) < 5:
                raise RuntimeError(
                    f"Only generated {len(samples)} samples, need at least 5"
                )

            # Stage 2: Augment samples
            self._update_job(
                job,
                status=JobStatus.AUGMENTING,
                progress=25,
                current_stage="Augmenting samples",
            )

            augmenter = AudioAugmenter()

            def augment_progress(current, total, stage):
                progress = 25 + (current / total) * 15
                self._update_job(job, progress=progress)

            positive_samples = await augmenter.augment_dataset(
                samples,
                augmentations_per_sample=5,
                progress_callback=augment_progress,
            )

            # Stage 3: Generate negative samples
            self._update_job(
                job,
                progress=40,
                current_stage="Generating negative samples",
            )

            neg_generator = NegativeSampleGenerator()
            noise_samples = neg_generator.generate_noise_samples(
                count=len(positive_samples) // 4
            )
            similar_samples = await neg_generator.generate_similar_words(
                word,
                generator,
                count=len(positive_samples) // 4,
            )

            negative_samples = noise_samples + similar_samples

            logger.info(
                f"Dataset prepared: {len(positive_samples)} positive, "
                f"{len(negative_samples)} negative samples"
            )

            # Stage 4: Train model
            self._update_job(
                job,
                status=JobStatus.TRAINING,
                progress=50,
                current_stage="Training model",
            )

            config = ModelConfig(word=word)
            trainer = WakeWordTrainer(config)

            def training_progress(epoch, total_epochs, metrics):
                progress = 50 + (epoch / total_epochs) * 40
                self._update_job(job, progress=progress, metrics=metrics)

            history = trainer.train(
                positive_samples=positive_samples,
                negative_samples=negative_samples,
                epochs=epochs,
                batch_size=batch_size,
                progress_callback=training_progress,
            )

            # Stage 5: Export model
            self._update_job(
                job,
                status=JobStatus.EXPORTING,
                progress=90,
                current_stage="Exporting model",
            )

            model_dir = self.models_dir / word_dir
            saved_files = trainer.save(model_dir)

            # Complete
            final_metrics = {
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
                "final_train_acc": history["train_acc"][-1],
                "final_val_acc": history["val_acc"][-1],
                "positive_samples": len(positive_samples),
                "negative_samples": len(negative_samples),
                "exported_formats": list(saved_files.keys()),
            }

            self._update_job(
                job,
                status=JobStatus.COMPLETED,
                progress=100,
                current_stage="Complete",
                model_path=str(model_dir),
                metrics=final_metrics,
            )

            word_lower = word.lower()
            if self.word_to_job.get(word_lower) == job.job_id:
                del self.word_to_job[word_lower]

            logger.info(f"Wake word training completed for job {job.job_id}")

        except asyncio.CancelledError:
            logger.info(f"Wake word job {job.job_id} was cancelled")
            raise
        except Exception as e:
            logger.error(f"Wake word training failed for job {job.job_id}: {e}")
            self._update_job(
                job,
                status=JobStatus.FAILED,
                error_message=str(e),
            )

            word_lower = job.word.lower()
            if self.word_to_job.get(word_lower) == job.job_id:
                del self.word_to_job[word_lower]

    def get_trainer(self, word: str) -> Optional[WakeWordTrainer]:
        """Get or load a trainer for inference."""
        model_dir = self.get_model_for_word(word)
        if not model_dir:
            return None

        word_lower = word.lower()

        with self._model_lock:
            if word_lower in self._loaded_models:
                return self._loaded_models[word_lower]

            trainer = WakeWordTrainer.load(model_dir)
            self._loaded_models[word_lower] = trainer
            return trainer

    def predict(self, word: str, audio_bytes: bytes) -> Tuple[bool, float]:
        """Predict if audio contains the wake word."""
        trainer = self.get_trainer(word)
        if not trainer:
            raise ValueError(f"No model found for word '{word}'")

        return trainer.predict(audio_bytes)

    def delete_model(self, word: str) -> bool:
        """Delete a trained model."""
        model_dir = self.get_model_for_word(word)
        if not model_dir:
            return False

        import shutil

        shutil.rmtree(model_dir)

        word_lower = word.lower()
        with self._model_lock:
            if word_lower in self._loaded_models:
                del self._loaded_models[word_lower]

        return True


# =============================================================================
# Global Instance
# =============================================================================

_wakeword_manager: Optional[WakeWordManager] = None
_wakeword_manager_lock = threading.Lock()


def get_wakeword_manager() -> WakeWordManager:
    """Get or create the global wake word manager."""
    global _wakeword_manager
    with _wakeword_manager_lock:
        if _wakeword_manager is None:
            _wakeword_manager = WakeWordManager()
        return _wakeword_manager


def set_wakeword_manager(manager: WakeWordManager):
    """Set a custom wake word manager (e.g., with Chatterbox model)."""
    global _wakeword_manager
    with _wakeword_manager_lock:
        _wakeword_manager = manager
