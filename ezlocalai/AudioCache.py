import os
import json
import hashlib
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
from datetime import datetime, timedelta
import numpy as np
from pydub import AudioSegment


class AudioCache:
    """
    Audio caching system for TTS outputs.
    Generates multiple samples and selects the best one based on duration.
    """

    def __init__(self, cache_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the audio cache with configuration.

        Args:
            cache_config: Configuration dictionary for cache settings
        """
        default_config = {
            "enabled": True,
            "cache_dir": os.path.join(os.getcwd(), "outputs", "cache"),
            "max_size_mb": 500,
            "ttl_days": 30,
            "generation_count": 2,
            "selection_method": "shortest",
            "warm_cache_on_init": False,
            "common_phrases": ["Hello", "Goodbye", "Thank you", "Yes", "No"],
            "enable_stats": True,
        }

        self.config = {**default_config, **(cache_config or {})}
        self.cache_dir = Path(self.config["cache_dir"])
        self.audio_dir = self.cache_dir / "audio"
        self.metadata_dir = self.cache_dir / "metadata"
        self.stats_file = self.cache_dir / "stats.json"

        # Create cache directories
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Initialize statistics
        self.stats = self._load_stats()

        # Clean up expired cache entries on initialization
        if self.config["enabled"]:
            self._cleanup_expired_cache()
            self._check_cache_size()

        logging.debug(f"[AudioCache] Initialized with cache dir: {self.cache_dir}")

    def generate_cache_key(
        self,
        text: str,
        voice: str,
        language: str = "en",
        extra_params: Optional[Dict] = None,
    ) -> str:
        """
        Generate a unique cache key for the given parameters.

        Args:
            text: The text to be spoken
            voice: Voice identifier
            language: Language code
            extra_params: Additional parameters to include in the hash

        Returns:
            SHA256 hash as cache key
        """
        # Normalize text for consistent hashing
        normalized_text = self._normalize_text(text)

        # Create hash input
        hash_input = {"text": normalized_text, "voice": voice, "language": language}

        if extra_params:
            hash_input.update(extra_params)

        # Generate hash
        hash_string = json.dumps(hash_input, sort_keys=True)
        cache_key = hashlib.sha256(hash_string.encode()).hexdigest()

        return cache_key

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent caching.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        # Convert to lowercase, strip whitespace, normalize spaces
        normalized = text.lower().strip()
        normalized = " ".join(normalized.split())
        return normalized

    def get_cached_audio(self, cache_key: str) -> Optional[bytes]:
        """
        Retrieve cached audio if it exists and is valid.

        Args:
            cache_key: The cache key to look up

        Returns:
            Audio data as bytes if found, None otherwise
        """
        if not self.config["enabled"]:
            return None

        audio_path = self.audio_dir / f"{cache_key}.wav"
        metadata_path = self.metadata_dir / f"{cache_key}.json"

        # Check if both files exist
        if not (audio_path.exists() and metadata_path.exists()):
            self._record_stats("miss")
            return None

        # Check if cache entry is expired
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            created_time = datetime.fromisoformat(metadata["created_at"])
            ttl_days = self.config["ttl_days"]

            if datetime.now() - created_time > timedelta(days=ttl_days):
                # Cache expired, remove it
                self._remove_cache_entry(cache_key)
                self._record_stats("expired")
                return None

            # Update last accessed time
            metadata["last_accessed"] = datetime.now().isoformat()
            metadata["access_count"] = metadata.get("access_count", 0) + 1

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Read and return audio data
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            self._record_stats("hit")
            logging.debug(f"[AudioCache] Cache hit for key: {cache_key[:8]}...")
            return audio_data

        except Exception as e:
            logging.error(f"[AudioCache] Error reading cache: {e}")
            self._record_stats("error")
            return None

    def store_cached_audio(
        self, cache_key: str, audio_data: bytes, metadata: Dict[str, Any]
    ) -> bool:
        """
        Store audio data in cache with metadata.

        Args:
            cache_key: The cache key
            audio_data: Audio data as bytes
            metadata: Metadata about the audio

        Returns:
            True if successfully stored, False otherwise
        """
        if not self.config["enabled"]:
            return False

        try:
            audio_path = self.audio_dir / f"{cache_key}.wav"
            metadata_path = self.metadata_dir / f"{cache_key}.json"

            # Add timestamp and initial access count
            metadata["created_at"] = datetime.now().isoformat()
            metadata["last_accessed"] = datetime.now().isoformat()
            metadata["access_count"] = 0
            metadata["cache_key"] = cache_key

            # Write audio file
            with open(audio_path, "wb") as f:
                f.write(audio_data)

            # Write metadata
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self._record_stats("store")
            logging.debug(f"[AudioCache] Stored cache entry: {cache_key[:8]}...")

            # Check cache size after storing
            self._check_cache_size()

            return True

        except Exception as e:
            logging.error(f"[AudioCache] Error storing cache: {e}")
            return False

    def select_best_audio(
        self, audio_samples: list, generation_func, generation_params: Dict
    ) -> Tuple[bytes, Dict]:
        """
        Generate multiple audio samples and select the best one.

        Args:
            audio_samples: List to store generated samples (for external generation)
            generation_func: Function to generate audio
            generation_params: Parameters for generation function

        Returns:
            Tuple of (best_audio_data, metadata)
        """
        generation_count = self.config["generation_count"]
        selection_method = self.config["selection_method"]

        samples = []
        durations = []

        logging.debug(
            f"[AudioCache] Generating {generation_count} samples for selection..."
        )

        # Generate multiple samples
        for i in range(generation_count):
            try:
                # Generate audio using the provided function
                audio_data = generation_func(**generation_params)

                # Save temporarily to analyze duration
                temp_path = self.cache_dir / f"temp_{i}.wav"
                with open(temp_path, "wb") as f:
                    f.write(audio_data)

                # Get duration using pydub
                audio_segment = AudioSegment.from_file(temp_path)
                duration_ms = len(audio_segment)

                samples.append(
                    {"data": audio_data, "duration_ms": duration_ms, "index": i}
                )
                durations.append(duration_ms)

                # Clean up temp file
                temp_path.unlink()

                logging.debug(f"[AudioCache] Sample {i+1}: duration = {duration_ms}ms")

            except Exception as e:
                logging.error(f"[AudioCache] Error generating sample {i+1}: {e}")
                continue

        if not samples:
            raise ValueError("Failed to generate any audio samples")

        # Select best sample based on method
        if selection_method == "shortest":
            best_sample = min(samples, key=lambda x: x["duration_ms"])
        elif selection_method == "statistical":
            # Select the sample closest to median duration
            median_duration = np.median(durations)
            best_sample = min(
                samples, key=lambda x: abs(x["duration_ms"] - median_duration)
            )
        else:
            # Default to shortest
            best_sample = min(samples, key=lambda x: x["duration_ms"])

        # Calculate statistics
        metadata = {
            "selection_method": selection_method,
            "generation_count": len(samples),
            "selected_index": best_sample["index"],
            "selected_duration_ms": best_sample["duration_ms"],
            "all_durations_ms": durations,
            "mean_duration_ms": np.mean(durations),
            "std_duration_ms": np.std(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
        }

        logging.debug(
            f"[AudioCache] Selected sample {best_sample['index']+1} "
            f"(duration: {best_sample['duration_ms']}ms)"
        )

        return best_sample["data"], metadata

    def _cleanup_expired_cache(self):
        """Remove expired cache entries."""
        if not self.config["enabled"]:
            return

        ttl_days = self.config["ttl_days"]
        current_time = datetime.now()
        removed_count = 0

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                created_time = datetime.fromisoformat(metadata["created_at"])

                if current_time - created_time > timedelta(days=ttl_days):
                    cache_key = metadata_file.stem
                    self._remove_cache_entry(cache_key)
                    removed_count += 1

            except Exception as e:
                logging.error(f"[AudioCache] Error checking expiry: {e}")

        if removed_count > 0:
            logging.debug(f"[AudioCache] Removed {removed_count} expired entries")

    def _check_cache_size(self):
        """Check and enforce cache size limit."""
        if not self.config["enabled"]:
            return

        max_size_mb = self.config["max_size_mb"]
        max_size_bytes = max_size_mb * 1024 * 1024

        # Calculate current cache size
        total_size = 0
        cache_entries = []

        for audio_file in self.audio_dir.glob("*.wav"):
            size = audio_file.stat().st_size
            total_size += size

            # Get metadata for sorting
            metadata_file = self.metadata_dir / f"{audio_file.stem}.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    last_accessed = metadata.get(
                        "last_accessed", metadata["created_at"]
                    )
                    cache_entries.append(
                        {
                            "key": audio_file.stem,
                            "size": size,
                            "last_accessed": last_accessed,
                        }
                    )
                except:
                    pass

        # Remove oldest entries if size exceeded
        if total_size > max_size_bytes:
            logging.debug(
                f"[AudioCache] Cache size ({total_size/1024/1024:.2f}MB) "
                f"exceeds limit ({max_size_mb}MB)"
            )

            # Sort by last accessed time (LRU)
            cache_entries.sort(key=lambda x: x["last_accessed"])

            # Remove entries until under limit
            while total_size > max_size_bytes and cache_entries:
                entry = cache_entries.pop(0)
                self._remove_cache_entry(entry["key"])
                total_size -= entry["size"]

            logging.debug(
                f"[AudioCache] Cache size after cleanup: {total_size/1024/1024:.2f}MB"
            )

    def _remove_cache_entry(self, cache_key: str):
        """Remove a cache entry."""
        audio_path = self.audio_dir / f"{cache_key}.wav"
        metadata_path = self.metadata_dir / f"{cache_key}.json"

        if audio_path.exists():
            audio_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()

    def _load_stats(self) -> Dict:
        """Load cache statistics."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, "r") as f:
                    return json.load(f)
            except:
                pass

        return {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "errors": 0,
            "expired": 0,
            "last_reset": datetime.now().isoformat(),
        }

    def _record_stats(self, event_type: str):
        """Record a cache event for statistics."""
        if not self.config["enable_stats"]:
            return

        if event_type == "hit":
            self.stats["hits"] += 1
        elif event_type == "miss":
            self.stats["misses"] += 1
        elif event_type == "store":
            self.stats["stores"] += 1
        elif event_type == "error":
            self.stats["errors"] += 1
        elif event_type == "expired":
            self.stats["expired"] += 1

        # Save stats periodically (every 10 events)
        total_events = sum(
            [
                self.stats["hits"],
                self.stats["misses"],
                self.stats["stores"],
                self.stats["errors"],
            ]
        )
        if total_events % 10 == 0:
            self._save_stats()

    def _save_stats(self):
        """Save cache statistics to file."""
        try:
            with open(self.stats_file, "w") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logging.error(f"[AudioCache] Error saving stats: {e}")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        )

        return {
            **self.stats,
            "hit_rate": f"{hit_rate:.2f}%",
            "total_requests": total_requests,
        }

    def clear_cache(self, voice: Optional[str] = None):
        """
        Clear cache entries.

        Args:
            voice: If specified, only clear entries for this voice
        """
        if voice:
            # Clear only specific voice entries
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                    if metadata.get("voice") == voice:
                        self._remove_cache_entry(metadata_file.stem)

                except Exception as e:
                    logging.error(f"[AudioCache] Error clearing cache: {e}")
        else:
            # Clear all cache
            shutil.rmtree(self.audio_dir, ignore_errors=True)
            shutil.rmtree(self.metadata_dir, ignore_errors=True)
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_dir.mkdir(parents=True, exist_ok=True)

            # Reset stats
            self.stats = {
                "hits": 0,
                "misses": 0,
                "stores": 0,
                "errors": 0,
                "expired": 0,
                "last_reset": datetime.now().isoformat(),
            }
            self._save_stats()

        logging.debug("[AudioCache] Cache cleared")

    def warm_cache(
        self,
        phrases: list,
        voice: str,
        language: str,
        generation_func,
        base_params: Dict,
    ):
        """
        Pre-generate cache for common phrases.

        Args:
            phrases: List of phrases to cache
            voice: Voice to use
            language: Language code
            generation_func: Function to generate audio
            base_params: Base parameters for generation
        """
        if not self.config["enabled"] or not self.config["warm_cache_on_init"]:
            return

        logging.debug(f"[AudioCache] Warming cache with {len(phrases)} phrases...")

        for phrase in phrases:
            cache_key = self.generate_cache_key(phrase, voice, language)

            # Skip if already cached
            if (self.audio_dir / f"{cache_key}.wav").exists():
                continue

            try:
                # Generate and cache
                generation_params = {
                    **base_params,
                    "text": phrase,
                    "voice": voice,
                    "language": language,
                }

                audio_data, metadata = self.select_best_audio(
                    [], generation_func, generation_params
                )

                # Add phrase info to metadata
                metadata["text"] = phrase
                metadata["voice"] = voice
                metadata["language"] = language
                metadata["warmed"] = True

                self.store_cached_audio(cache_key, audio_data, metadata)

            except Exception as e:
                logging.error(f"[AudioCache] Error warming cache for '{phrase}': {e}")

        logging.debug("[AudioCache] Cache warming complete")
