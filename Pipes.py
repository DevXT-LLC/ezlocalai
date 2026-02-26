import os
import logging
import time
import math
import threading
import queue
import tempfile
from dotenv import load_dotenv
from ezlocalai.LLM import (
    LLM,
    get_free_vram_per_gpu,
    get_total_vram_per_gpu,
    calculate_tensor_split_from_free_vram,
)
from ezlocalai.CTTS import CTTS
from precache import has_voice_server_url
from pyngrok import ngrok
import requests
import base64
import pdfplumber
import json
from Globals import getenv
import gc
import torch
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

try:
    from ezlocalai.IMG import IMG

    img_import_success = True
except ImportError:
    img_import_success = False


# =============================================================================
# Video Processing Helpers
# =============================================================================


def extract_frames_from_video(
    video_source: str,
    fps: float = 1.0,
    max_frames: int = 16,
) -> List[str]:
    """Extract frames from a video and return as base64-encoded images.

    Args:
        video_source: Either a URL, file path, or base64 data URL of the video
        fps: Frames per second to extract (default 1.0 = 1 frame per second)
        max_frames: Maximum number of frames to extract (default 16)

    Returns:
        List of base64 data URLs in format "data:image/jpeg;base64,..."
    """
    try:
        import cv2
        from PIL import Image as PILImage
        from io import BytesIO
    except ImportError:
        logging.error(
            "[Video] OpenCV (cv2) is required for video processing. Install with: pip install opencv-python"
        )
        return []

    temp_file = None
    frames_base64 = []

    try:
        # Handle different video source types
        if video_source.startswith("data:"):
            # Base64 data URL - extract and save to temp file
            try:
                header, encoded = video_source.split(",", 1)
                video_bytes = base64.b64decode(encoded)
                temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                temp_file.write(video_bytes)
                temp_file.close()
                video_path = temp_file.name
            except Exception as e:
                logging.error(f"[Video] Failed to decode base64 video: {e}")
                return []
        elif video_source.startswith("http://") or video_source.startswith("https://"):
            # URL - download to temp file
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(
                    video_source, timeout=60, headers=headers, stream=True
                )
                response.raise_for_status()
                temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file.close()
                video_path = temp_file.name
            except Exception as e:
                logging.error(f"[Video] Failed to download video from URL: {e}")
                return []
        else:
            # Local file path
            video_path = video_source
            if not os.path.exists(video_path):
                logging.error(f"[Video] Video file not found: {video_path}")
                return []

        # Open video with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"[Video] Failed to open video: {video_path}")
            return []

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        logging.debug(
            f"[Video] Video properties: {video_fps:.1f} fps, {total_frames} frames, {duration:.1f}s duration"
        )

        # Calculate frame interval based on requested fps
        frame_interval = int(video_fps / fps) if fps > 0 else int(video_fps)
        frame_interval = max(1, frame_interval)  # At least 1 frame interval

        # Calculate total frames to extract
        estimated_frames = total_frames // frame_interval
        if estimated_frames > max_frames:
            # Adjust interval to get approximately max_frames
            frame_interval = total_frames // max_frames

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract frame at specified intervals
            if frame_count % frame_interval == 0 and extracted_count < max_frames:
                try:
                    # Convert BGR (OpenCV) to RGB (PIL)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = PILImage.fromarray(frame_rgb)

                    # Resize if too large (max 1024px on longest side) to save tokens
                    max_dim = max(pil_img.size)
                    if max_dim > 1024:
                        scale = 1024 / max_dim
                        new_size = (
                            int(pil_img.size[0] * scale),
                            int(pil_img.size[1] * scale),
                        )
                        pil_img = pil_img.resize(new_size, PILImage.Resampling.LANCZOS)

                    # DEBUG: Save first frame for verification
                    if extracted_count == 0:
                        debug_path = "/app/outputs/debug_frame_0.png"
                        try:
                            pil_img.save(debug_path)
                            logging.info(
                                f"[Video DEBUG] Saved first frame to {debug_path}, size={pil_img.size}"
                            )
                        except Exception as save_err:
                            logging.warning(
                                f"[Video DEBUG] Could not save debug frame: {save_err}"
                            )

                    # Convert to base64 JPEG
                    buffer = BytesIO()
                    pil_img.save(buffer, format="JPEG", quality=85)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    frames_base64.append(f"data:image/jpeg;base64,{img_base64}")
                    extracted_count += 1

                    # Calculate timestamp for this frame
                    timestamp = frame_count / video_fps if video_fps > 0 else 0
                    logging.debug(
                        f"[Video] Extracted frame {extracted_count} at {timestamp:.1f}s"
                    )

                except Exception as e:
                    logging.error(f"[Video] Failed to process frame {frame_count}: {e}")

            frame_count += 1

        cap.release()
        logging.info(
            f"[Video] Extracted {extracted_count} frames from video ({duration:.1f}s)"
        )

    except Exception as e:
        logging.error(f"[Video] Error processing video: {e}")
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass

    return frames_base64


# =============================================================================
# Resource Management System
# =============================================================================


class ModelType(Enum):
    """Types of models that can be loaded."""

    LLM = "llm"
    VISION_LLM = "vision_llm"
    TTS = "tts"
    STT = "stt"
    IMG = "img"
    EMBEDDING = "embedding"


@dataclass
class ModelResource:
    """Tracks resource usage for a loaded model."""

    model_type: ModelType
    name: str
    vram_gb: float  # Estimated VRAM usage in GB
    device: str  # "cuda", "cuda:0", "cuda:1", "cpu"
    in_use: bool = False  # Is the model currently processing a request?
    last_used: float = 0.0  # Timestamp of last use


# Approximate VRAM requirements for different model types (in GB)
# These are estimates used for planning, actual usage may vary
MODEL_VRAM_ESTIMATES = {
    ModelType.LLM: 8.0,  # Varies greatly by model size and context
    ModelType.VISION_LLM: 6.0,  # Vision models with projector
    ModelType.TTS: 4.0,  # Chatterbox TTS
    ModelType.STT: 2.0,  # Whisper (varies by size)
    ModelType.IMG: 16.0,  # Z-Image-Turbo (can use CPU offload)
    ModelType.EMBEDDING: 1.5,  # BGE-M3
}


class ResourceManager:
    """Manages GPU/CPU resources across all models.

    Key principles:
    1. LLM is the primary workload - keep it loaded when possible
    2. Auxiliary models (TTS, STT, IMG) can run on CPU if GPU is busy
    3. Only unload models when we actually need the VRAM
    4. Fall back to external server when truly resource-exhausted
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._loaded_models: Dict[ModelType, ModelResource] = {}
        self._model_locks: Dict[ModelType, threading.Lock] = {
            mt: threading.Lock() for mt in ModelType
        }

        # Get system resources
        self.gpu_count = get_gpu_count()
        self.per_gpu_total_vram = get_per_gpu_vram_gb()
        self.total_vram = sum(self.per_gpu_total_vram) if self.per_gpu_total_vram else 0
        self.system_ram_gb = self._get_system_ram_gb()

        # Reserve some VRAM for system overhead
        self.vram_safety_margin = 1.0  # GB

        logging.info(
            f"[ResourceManager] Initialized: {self.gpu_count} GPU(s), "
            f"{self.total_vram:.1f}GB total VRAM, {self.system_ram_gb:.1f}GB system RAM"
        )

    def _get_system_ram_gb(self) -> float:
        """Get total system RAM in GB."""
        try:
            import psutil

            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 128.0  # Assume 128GB if psutil not available

    def get_free_vram(self, gpu_index: int = None) -> float:
        """Get current free VRAM in GB."""
        return get_free_vram_gb(gpu_index)

    def get_total_free_vram(self) -> float:
        """Get total free VRAM across all GPUs."""
        if self.gpu_count == 0:
            return 0.0
        return sum(get_per_gpu_free_vram_gb())

    def is_model_loaded(self, model_type: ModelType) -> bool:
        """Check if a model type is currently loaded."""
        with self._lock:
            return model_type in self._loaded_models

    def is_model_in_use(self, model_type: ModelType) -> bool:
        """Check if a model is currently processing a request."""
        with self._lock:
            resource = self._loaded_models.get(model_type)
            return resource.in_use if resource else False

    def mark_model_in_use(self, model_type: ModelType, in_use: bool = True):
        """Mark a model as in-use or idle."""
        with self._lock:
            if model_type in self._loaded_models:
                self._loaded_models[model_type].in_use = in_use
                if in_use:
                    self._loaded_models[model_type].last_used = time.time()

    def register_model(
        self, model_type: ModelType, name: str, device: str, vram_gb: float = None
    ):
        """Register a newly loaded model."""
        with self._lock:
            if vram_gb is None:
                vram_gb = MODEL_VRAM_ESTIMATES.get(model_type, 2.0)

            self._loaded_models[model_type] = ModelResource(
                model_type=model_type,
                name=name,
                vram_gb=vram_gb if "cuda" in device else 0.0,
                device=device,
                last_used=time.time(),
            )
            logging.debug(
                f"[ResourceManager] Registered {model_type.value}: {name} on {device} "
                f"({vram_gb:.1f}GB VRAM)"
            )

    def unregister_model(self, model_type: ModelType):
        """Unregister a model that's being unloaded."""
        with self._lock:
            if model_type in self._loaded_models:
                resource = self._loaded_models.pop(model_type)
                logging.debug(
                    f"[ResourceManager] Unregistered {model_type.value}: {resource.name}"
                )

    def can_load_model(
        self, model_type: ModelType, required_vram: float = None
    ) -> Tuple[bool, str, str]:
        """Check if we can load a model and determine optimal device.

        Returns:
            Tuple of (can_load, device, reason)
            - can_load: True if we can load the model
            - device: Recommended device ("cuda", "cuda:N", "cpu")
            - reason: Human-readable explanation
        """
        with self._lock:
            if required_vram is None:
                required_vram = MODEL_VRAM_ESTIMATES.get(model_type, 2.0)

            free_vram = self.get_total_free_vram()

            # Check if we have enough free VRAM
            if free_vram >= required_vram + self.vram_safety_margin:
                # Find the best GPU
                if self.gpu_count == 1:
                    return True, "cuda", f"Sufficient VRAM ({free_vram:.1f}GB free)"
                else:
                    # Multi-GPU: for non-LLM models, prefer the secondary (less
                    # powerful) GPU so the primary stays free for the LLM.
                    free_per_gpu = get_per_gpu_free_vram_gb()
                    primary = get_primary_gpu()
                    secondary = get_secondary_gpu()

                    if model_type != ModelType.LLM and secondary is not None:
                        # Try secondary GPU first for non-LLM models
                        if free_per_gpu[secondary] >= required_vram:
                            return (
                                True,
                                f"cuda:{secondary}",
                                f"Secondary GPU {secondary} has {free_per_gpu[secondary]:.1f}GB free (primary reserved for LLM)",
                            )

                    # Fall back to GPU with most free VRAM
                    best_gpu = max(
                        range(len(free_per_gpu)), key=lambda i: free_per_gpu[i]
                    )
                    if free_per_gpu[best_gpu] >= required_vram:
                        return (
                            True,
                            f"cuda:{best_gpu}",
                            f"GPU {best_gpu} has {free_per_gpu[best_gpu]:.1f}GB free",
                        )
                    # Use tensor split across GPUs
                    return (
                        True,
                        "cuda",
                        f"Using tensor split ({free_vram:.1f}GB total free)",
                    )

            # Check if we can free VRAM by unloading idle models
            idle_vram = 0.0
            idle_models = []
            for mt, resource in self._loaded_models.items():
                if (
                    not resource.in_use and mt != ModelType.LLM
                ):  # Never suggest unloading active LLM
                    idle_vram += resource.vram_gb
                    idle_models.append(mt)

            if free_vram + idle_vram >= required_vram + self.vram_safety_margin:
                return (
                    True,
                    "cuda",
                    f"Can free {idle_vram:.1f}GB by unloading {[m.value for m in idle_models]}",
                )

            # Check if model can run on CPU
            if model_type in [ModelType.TTS, ModelType.STT, ModelType.EMBEDDING]:
                return True, "cpu", f"Insufficient VRAM, using CPU (model supports CPU)"

            # IMG can use CPU offload
            if model_type == ModelType.IMG:
                return True, "cuda", "Using sequential CPU offload for image generation"

            # LLM - last resort, check if we have fallback
            fallback_server = getenv("FALLBACK_SERVER")
            if fallback_server:
                return (
                    False,
                    "fallback",
                    f"Insufficient VRAM ({free_vram:.1f}GB), will use fallback server",
                )

            return (
                False,
                "none",
                f"Insufficient VRAM ({free_vram:.1f}GB free, need {required_vram}GB)",
            )

    def get_models_to_unload(
        self, required_vram: float, exclude: List[ModelType] = None
    ) -> List[ModelType]:
        """Get list of models that should be unloaded to free VRAM.

        Prioritizes unloading:
        1. Models not currently in use
        2. Least recently used models
        3. Models that aren't the primary LLM

        Args:
            required_vram: Amount of VRAM we need to free
            exclude: Model types to never unload

        Returns:
            List of ModelType to unload, in order
        """
        with self._lock:
            if exclude is None:
                exclude = []

            # Always exclude LLM unless specifically requested
            exclude = list(exclude) + [ModelType.LLM]

            # Get idle models sorted by last used (oldest first)
            candidates = []
            for mt, resource in self._loaded_models.items():
                if mt not in exclude and not resource.in_use and resource.vram_gb > 0:
                    candidates.append((mt, resource))

            candidates.sort(key=lambda x: x[1].last_used)

            # Select models to unload until we have enough VRAM
            to_unload = []
            freed_vram = 0.0
            for mt, resource in candidates:
                if freed_vram >= required_vram:
                    break
                to_unload.append(mt)
                freed_vram += resource.vram_gb

            return to_unload

    def get_status(self) -> Dict[str, Any]:
        """Get current resource status for monitoring."""
        with self._lock:
            return {
                "gpu_count": self.gpu_count,
                "total_vram_gb": self.total_vram,
                "free_vram_gb": self.get_total_free_vram(),
                "free_per_gpu_gb": get_per_gpu_free_vram_gb(),
                "system_ram_gb": self.system_ram_gb,
                "loaded_models": {
                    mt.value: {
                        "name": r.name,
                        "device": r.device,
                        "vram_gb": r.vram_gb,
                        "in_use": r.in_use,
                        "last_used": r.last_used,
                    }
                    for mt, r in self._loaded_models.items()
                },
            }


# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None
_resource_manager_lock = threading.Lock()


def get_resource_manager() -> ResourceManager:
    """Get or create the global resource manager."""
    global _resource_manager
    with _resource_manager_lock:
        if _resource_manager is None:
            _resource_manager = ResourceManager()
        return _resource_manager


# =============================================================================
# Fallback ezlocalai Client
# =============================================================================


class EzlocalaiClient:
    """Client for forwarding requests to another ezlocalai instance when local resources are exhausted.

    This enables a distributed setup where multiple ezlocalai servers can fall back to each other,
    providing redundancy and load balancing based on resource availability (VRAM/RAM).

    The client auto-detects if FALLBACK_SERVER is an ezlocalai instance by checking for
    the /v1/resources endpoint.
    """

    def __init__(self, base_url: str = None, api_key: str = None):
        """Initialize the ezlocalai fallback client.

        Args:
            base_url: URL of the fallback server (e.g., "http://192.168.1.100:8091")
            api_key: API key for the fallback server (uses local key if not provided)
        """
        self.base_url = (
            (base_url or getenv("FALLBACK_SERVER")).rstrip("/")
            if (base_url or getenv("FALLBACK_SERVER"))
            else ""
        )
        self.api_key = (
            api_key or getenv("FALLBACK_API_KEY") or getenv("EZLOCALAI_API_KEY")
        )
        self._session = None
        self._available = None  # Cache availability check
        self._last_check = 0
        self._check_interval = 30  # Re-check availability every 30 seconds

    @property
    def is_configured(self) -> bool:
        """Check if a fallback server is configured."""
        return bool(self.base_url)

    def _get_headers(self) -> dict:
        """Get request headers with authentication."""
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "none":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def check_availability(self) -> Tuple[bool, str]:
        """Check if the fallback server is available and has resources.

        Returns:
            Tuple of (is_available, reason)
        """
        if not self.is_configured:
            return False, "No fallback server configured"

        # Use cached result if recent
        current_time = time.time()
        if (
            self._available is not None
            and (current_time - self._last_check) < self._check_interval
        ):
            return self._available, "cached"

        try:
            async with self._get_session() as session:
                # Check health endpoint
                async with session.get(
                    f"{self.base_url}/v1/resources",
                    headers=self._get_headers(),
                    timeout=5,
                ) as resp:
                    if resp.status != 200:
                        self._available = False
                        self._last_check = current_time
                        return False, f"Fallback server returned status {resp.status}"

                    data = await resp.json()
                    free_vram = data.get("free_vram_gb", 0)

                    # Check if fallback has enough VRAM
                    min_vram = float(getenv("FALLBACK_VRAM_THRESHOLD", "2.0"))
                    if free_vram < min_vram:
                        self._available = False
                        self._last_check = current_time
                        return (
                            False,
                            f"Fallback server low on VRAM ({free_vram:.1f}GB < {min_vram}GB)",
                        )

                    self._available = True
                    self._last_check = current_time
                    return (
                        True,
                        f"Fallback server available ({free_vram:.1f}GB VRAM free)",
                    )

        except Exception as e:
            self._available = False
            self._last_check = current_time
            return False, f"Fallback server unreachable: {str(e)}"

    def _get_session(self):
        """Get or create an aiohttp session."""
        import aiohttp

        return aiohttp.ClientSession()

    async def forward_chat_completion(self, data: dict, stream: bool = False):
        """Forward a chat completion request to the fallback server.

        Args:
            data: The request data dict
            stream: Whether to stream the response

        Returns:
            The response from the fallback server
        """
        if not self.is_configured:
            raise RuntimeError("No fallback server configured")

        import aiohttp

        logging.info(f"[Fallback] Forwarding chat completion to {self.base_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=data,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(
                        total=float(getenv("REQUEST_TIMEOUT", "300"))
                    ),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"Fallback server error {resp.status}: {error_text}"
                        )

                    if stream:
                        # Return async generator for streaming
                        async def stream_response():
                            async for line in resp.content:
                                line_str = line.decode("utf-8").strip()
                                if line_str.startswith("data: "):
                                    chunk_data = line_str[6:]
                                    if chunk_data == "[DONE]":
                                        break
                                    try:
                                        yield json.loads(chunk_data)
                                    except json.JSONDecodeError:
                                        continue

                        return stream_response()
                    else:
                        return await resp.json()

        except Exception as e:
            logging.error(f"[Fallback] Chat completion forward failed: {e}")
            raise

    async def forward_completion(self, data: dict, stream: bool = False):
        """Forward a completion request to the fallback server."""
        if not self.is_configured:
            raise RuntimeError("No fallback server configured")

        import aiohttp

        logging.info(f"[Fallback] Forwarding completion to {self.base_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/completions",
                    json=data,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(
                        total=float(getenv("REQUEST_TIMEOUT", "300"))
                    ),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"Fallback server error {resp.status}: {error_text}"
                        )

                    if stream:

                        async def stream_response():
                            async for line in resp.content:
                                line_str = line.decode("utf-8").strip()
                                if line_str.startswith("data: "):
                                    chunk_data = line_str[6:]
                                    if chunk_data == "[DONE]":
                                        break
                                    try:
                                        yield json.loads(chunk_data)
                                    except json.JSONDecodeError:
                                        continue

                        return stream_response()
                    else:
                        return await resp.json()

        except Exception as e:
            logging.error(f"[Fallback] Completion forward failed: {e}")
            raise

    async def forward_embeddings(self, data: dict):
        """Forward an embeddings request to the fallback server."""
        if not self.is_configured:
            raise RuntimeError("No fallback server configured")

        import aiohttp

        logging.info(f"[Fallback] Forwarding embeddings to {self.base_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/embeddings",
                    json=data,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"Fallback server error {resp.status}: {error_text}"
                        )
                    return await resp.json()

        except Exception as e:
            logging.error(f"[Fallback] Embeddings forward failed: {e}")
            raise

    async def forward_transcription(
        self, file_content: bytes, content_type: str, **params
    ):
        """Forward a transcription request to the fallback server."""
        if not self.is_configured:
            raise RuntimeError("No fallback server configured")

        import aiohttp

        logging.info(f"[Fallback] Forwarding transcription to {self.base_url}")

        try:
            headers = {}
            if self.api_key and self.api_key != "none":
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Build multipart form data
            data = aiohttp.FormData()
            data.add_field(
                "file", file_content, content_type=content_type, filename="audio.wav"
            )
            for key, value in params.items():
                if value is not None:
                    data.add_field(key, str(value))

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/audio/transcriptions",
                    data=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"Fallback server error {resp.status}: {error_text}"
                        )
                    return await resp.json()

        except Exception as e:
            logging.error(f"[Fallback] Transcription forward failed: {e}")
            raise

    async def forward_tts(
        self, text: str, voice: str = "default", language: str = "en"
    ):
        """Forward a TTS request to the fallback server."""
        if not self.is_configured:
            raise RuntimeError("No fallback server configured")

        import aiohttp

        logging.info(f"[Fallback] Forwarding TTS to {self.base_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/audio/speech",
                    json={"input": text, "voice": voice, "language": language},
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"Fallback server error {resp.status}: {error_text}"
                        )
                    # Return raw audio bytes
                    return await resp.read()

        except Exception as e:
            logging.error(f"[Fallback] TTS forward failed: {e}")
            raise

    async def forward_image_generation(
        self, prompt: str, response_format: str = "url", size: str = "512x512"
    ):
        """Forward an image generation request to the fallback server."""
        if not self.is_configured:
            raise RuntimeError("No fallback server configured")

        import aiohttp

        logging.info(f"[Fallback] Forwarding image generation to {self.base_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/images/generations",
                    json={
                        "prompt": prompt,
                        "response_format": response_format,
                        "size": size,
                    },
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"Fallback server error {resp.status}: {error_text}"
                        )
                    return await resp.json()

        except Exception as e:
            logging.error(f"[Fallback] Image generation forward failed: {e}")
            raise

    async def get_models(self):
        """Get available models from the fallback server."""
        if not self.is_configured:
            return {"data": []}

        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/v1/models",
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return {"data": []}
                    return await resp.json()

        except Exception as e:
            logging.debug(f"[Fallback] Failed to get models: {e}")
            return {"data": []}


# Global fallback client instance
_fallback_client: Optional[EzlocalaiClient] = None
_fallback_client_lock = threading.Lock()


def get_fallback_client() -> EzlocalaiClient:
    """Get or create the global fallback client."""
    global _fallback_client
    with _fallback_client_lock:
        if _fallback_client is None:
            _fallback_client = EzlocalaiClient()
        return _fallback_client


# =============================================================================
# Voice Server Client - For offloading TTS/STT to dedicated voice server
# =============================================================================


class VoiceServerClient:
    """Client for forwarding TTS/STT requests to a dedicated voice server.

    This enables separating voice processing (TTS/STT) from LLM inference,
    allowing a dedicated GPU (e.g., RTX 3090) to handle voice workloads while
    the main server focuses on LLM inference.

    Configuration via VOICE_SERVER env var:
    - Empty (default): Load voice models locally on demand (lazy loading)
    - URL (e.g., "http://192.168.1.100:8091"): Forward voice requests to voice server
    - "true": This server IS a voice server - keep TTS/STT loaded, lazy-load LLMs
    """

    def __init__(self, base_url: str = None, api_key: str = None):
        """Initialize the voice server client.

        Args:
            base_url: URL of the voice server (e.g., "http://192.168.1.100:8091")
            api_key: API key for the voice server (uses local key if not provided)
        """
        voice_server = base_url or getenv("VOICE_SERVER")
        # Handle "true" case - this server IS the voice server
        if voice_server and voice_server.lower() == "true":
            self.base_url = ""
            self.is_voice_server_mode = True
        else:
            self.base_url = voice_server.rstrip("/") if voice_server else ""
            self.is_voice_server_mode = False

        self.api_key = (
            api_key or getenv("VOICE_SERVER_API_KEY") or getenv("EZLOCALAI_API_KEY")
        )
        self._available = None
        self._last_check = 0
        self._check_interval = 30  # Re-check availability every 30 seconds

    @property
    def is_configured(self) -> bool:
        """Check if a voice server URL is configured (not "true" mode)."""
        return bool(self.base_url) and not self.is_voice_server_mode

    @property
    def should_keep_voice_loaded(self) -> bool:
        """Check if this server should keep voice models loaded (voice server mode)."""
        return self.is_voice_server_mode

    def _get_headers(self) -> dict:
        """Get request headers with authentication."""
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "none":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def check_availability(self) -> Tuple[bool, str]:
        """Check if the voice server is available.

        Returns:
            Tuple of (is_available, reason)
        """
        if not self.is_configured:
            return False, "No voice server configured"

        # Use cached result if recent
        current_time = time.time()
        if (
            self._available is not None
            and (current_time - self._last_check) < self._check_interval
        ):
            return self._available, "cached"

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                # Check if voice server is reachable via /v1/audio/voices endpoint
                async with session.get(
                    f"{self.base_url}/v1/audio/voices",
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        self._available = True
                        self._last_check = current_time
                        return True, "Voice server available"
                    else:
                        self._available = False
                        self._last_check = current_time
                        return False, f"Voice server returned status {resp.status}"

        except Exception as e:
            self._available = False
            self._last_check = current_time
            return False, f"Voice server unreachable: {str(e)}"

    async def forward_tts(
        self, text: str, voice: str = "default", language: str = "en"
    ) -> Optional[bytes]:
        """Forward a TTS request to the voice server.

        Args:
            text: Text to convert to speech
            voice: Voice to use
            language: Language code

        Returns:
            Audio bytes if successful, None if failed
        """
        if not self.is_configured:
            return None

        import aiohttp

        logging.info(f"[VoiceServer] Forwarding TTS to {self.base_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/audio/speech",
                    json={"input": text, "voice": voice, "language": language},
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        audio_bytes = await resp.read()
                        logging.debug(
                            f"[VoiceServer] TTS successful, received {len(audio_bytes)} bytes"
                        )
                        return audio_bytes
                    else:
                        error_text = await resp.text()
                        logging.warning(
                            f"[VoiceServer] TTS failed with status {resp.status}: {error_text}"
                        )
                        return None

        except Exception as e:
            logging.warning(f"[VoiceServer] TTS request failed: {e}")
            return None

    async def forward_tts_stream(
        self, text: str, voice: str = "default", language: str = "en"
    ):
        """Forward a streaming TTS request to the voice server.

        Args:
            text: Text to convert to speech
            voice: Voice to use
            language: Language code

        Yields:
            Audio chunks as they arrive from the voice server
        """
        if not self.is_configured:
            return

        import aiohttp

        logging.info(f"[VoiceServer] Forwarding streaming TTS to {self.base_url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/audio/speech/stream",
                    json={"input": text, "voice": voice, "language": language},
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status == 200:
                        async for chunk in resp.content.iter_any():
                            if chunk:
                                yield chunk
                        logging.debug("[VoiceServer] TTS stream completed")
                    else:
                        error_text = await resp.text()
                        logging.warning(
                            f"[VoiceServer] TTS stream failed with status {resp.status}: {error_text}"
                        )

        except Exception as e:
            logging.warning(f"[VoiceServer] TTS stream request failed: {e}")

    async def forward_transcription(
        self,
        file_content: bytes,
        content_type: str,
        model: str = None,
        language: str = None,
        prompt: str = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> Optional[dict]:
        """Forward a transcription request to the voice server.

        Args:
            file_content: Audio file content
            content_type: MIME type of the audio
            model: STT model to use
            language: Language code
            prompt: Optional prompt for transcription
            response_format: Response format
            temperature: Sampling temperature

        Returns:
            Transcription result dict if successful, None if failed
        """
        if not self.is_configured:
            return None

        import aiohttp

        logging.info(f"[VoiceServer] Forwarding transcription to {self.base_url}")

        try:
            headers = {}
            if self.api_key and self.api_key != "none":
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Build multipart form data
            data = aiohttp.FormData()
            data.add_field(
                "file", file_content, content_type=content_type, filename="audio.wav"
            )
            if model:
                data.add_field("model", model)
            if language:
                data.add_field("language", language)
            if prompt:
                data.add_field("prompt", prompt)
            data.add_field("response_format", response_format)
            data.add_field("temperature", str(temperature))

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/audio/transcriptions",
                    data=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        if response_format == "text":
                            text = await resp.text()
                            return {"text": text}
                        return await resp.json()
                    else:
                        error_text = await resp.text()
                        logging.warning(
                            f"[VoiceServer] Transcription failed with status {resp.status}: {error_text}"
                        )
                        return None

        except Exception as e:
            logging.warning(f"[VoiceServer] Transcription request failed: {e}")
            return None

    async def get_voices(self) -> Optional[dict]:
        """Get available voices from the voice server.

        Returns:
            Dict with voices list if successful, None if failed
        """
        if not self.is_configured:
            return None

        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/v1/audio/voices",
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return None

        except Exception as e:
            logging.debug(f"[VoiceServer] Failed to get voices: {e}")
            return None


# Global voice server client instance
_voice_server_client: Optional[VoiceServerClient] = None
_voice_server_client_lock = threading.Lock()


def get_voice_server_client() -> VoiceServerClient:
    """Get or create the global voice server client."""
    global _voice_server_client
    with _voice_server_client_lock:
        if _voice_server_client is None:
            _voice_server_client = VoiceServerClient()
        return _voice_server_client


def is_voice_server_mode() -> bool:
    """Check if this server is running in voice server mode (VOICE_SERVER=true).

    In voice server mode:
    - TTS and STT models are kept loaded (not lazy loaded/unloaded)
    - LLM models are lazy loaded instead

    Returns:
        True if VOICE_SERVER env var is set to "true"
    """
    return get_voice_server_client().should_keep_voice_loaded


def has_voice_server_url() -> bool:
    """Check if a voice server URL is configured (not 'true' mode, but actual URL).

    When a voice server URL is configured:
    - Voice requests (TTS/STT) are forwarded to the voice server
    - Local voice models should NOT be loaded at all

    Returns:
        True if VOICE_SERVER is set to a URL (not empty, not 'true')
    """
    return get_voice_server_client().is_configured


def should_preload_voice() -> bool:
    """Check if voice models should be preloaded at startup.

    Voice models (TTS/STT) are preloaded when:
    - LAZY_LOAD_VOICE=false (explicitly requested preload), OR
    - VOICE_SERVER=true (voice server mode always preloads)

    Returns:
        True if voice models should be preloaded at startup
    """
    lazy_load = getenv("LAZY_LOAD_VOICE").lower()
    return lazy_load == "false" or is_voice_server_mode()


def should_use_ezlocalai_fallback() -> Tuple[bool, str]:
    """Check if we should use the fallback server based on local resource state.

    Uses combined VRAM + RAM as the threshold since models can offload layers to system RAM.

    Returns:
        Tuple of (should_fallback, reason)
    """
    fallback_client = get_fallback_client()

    if not fallback_client.is_configured:
        return False, "No fallback server configured"

    resource_mgr = get_resource_manager()

    # Get free VRAM
    free_vram = resource_mgr.get_total_free_vram()

    # Get free RAM
    try:
        import psutil

        free_ram = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        free_ram = 0.0  # If psutil not available, only count VRAM

    # Combined free memory (VRAM + RAM)
    free_memory = free_vram + free_ram
    memory_threshold = float(getenv("FALLBACK_MEMORY_THRESHOLD", "8.0"))

    if free_memory < memory_threshold:
        return (
            True,
            f"Low combined memory ({free_vram:.1f}GB VRAM + {free_ram:.1f}GB RAM = {free_memory:.1f}GB < {memory_threshold}GB threshold)",
        )

    return False, "Local resources sufficient"


# Background cleanup queue for async model unloading
_cleanup_queue = queue.Queue()
_cleanup_thread = None
_cleanup_thread_lock = threading.Lock()


def _cleanup_worker():
    """Background worker thread that handles model cleanup asynchronously.

    This allows responses to be returned to the user immediately while
    model unloading and VRAM cleanup happens in the background.
    """
    while True:
        try:
            cleanup_task = _cleanup_queue.get(timeout=1.0)
            if cleanup_task is None:
                # Shutdown signal
                break

            cleanup_func, args, kwargs = cleanup_task
            try:
                cleanup_func(*args, **kwargs)
            except Exception as e:
                logging.error(f"[Cleanup] Background cleanup error: {e}")
        except queue.Empty:
            continue


def _ensure_cleanup_thread():
    """Ensure the background cleanup thread is running."""
    global _cleanup_thread
    with _cleanup_thread_lock:
        if _cleanup_thread is None or not _cleanup_thread.is_alive():
            _cleanup_thread = threading.Thread(target=_cleanup_worker, daemon=True)
            _cleanup_thread.start()
            logging.debug("[Cleanup] Background cleanup thread started")


def _schedule_cleanup(cleanup_func, *args, **kwargs):
    """Schedule a cleanup function to run in the background."""
    _ensure_cleanup_thread()
    _cleanup_queue.put((cleanup_func, args, kwargs))


# xllamacpp memory estimation
try:
    import xllamacpp
    from xllamacpp import estimate_gpu_layers, get_device_info
    from huggingface_hub import hf_hub_download

    xllamacpp_available = True
except ImportError:
    xllamacpp_available = False


def get_gpu_count() -> int:
    """Get the number of available CUDA GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_available_vram_gb(gpu_index: int = None) -> float:
    """Get available VRAM in GB, rounded down to nearest 1GB for safety margin.

    Args:
        gpu_index: Specific GPU index, or None to sum all GPUs for multi-GPU setups.
    """
    if torch.cuda.is_available():
        if gpu_index is not None:
            # Single GPU
            if gpu_index < torch.cuda.device_count():
                total = torch.cuda.get_device_properties(gpu_index).total_memory
                return math.floor(total / (1024**3))
            return 0.0
        else:
            # Sum all GPUs for total available VRAM budget
            total_vram = 0.0
            for i in range(torch.cuda.device_count()):
                total_vram += torch.cuda.get_device_properties(i).total_memory
            return math.floor(total_vram / (1024**3))
    return 0.0


def get_free_vram_gb(gpu_index: int = None) -> float:
    """Get FREE (available) VRAM in GB, accounting for other processes.

    Args:
        gpu_index: Specific GPU index, or None to sum all GPUs.
    """
    if not torch.cuda.is_available():
        return 0.0

    if gpu_index is not None:
        if gpu_index < torch.cuda.device_count():
            free, _ = torch.cuda.mem_get_info(gpu_index)
            return free / (1024**3)
        return 0.0
    else:
        total_free = 0.0
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            total_free += free
        return total_free / (1024**3)


def get_per_gpu_vram_gb() -> list:
    """Get VRAM for each GPU as a list of GB values."""
    if torch.cuda.is_available():
        vram_list = []
        for i in range(torch.cuda.device_count()):
            vram_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            vram_list.append(math.floor(vram_gb))
        return vram_list
    return []


def get_per_gpu_free_vram_gb() -> list:
    """Get FREE VRAM for each GPU as a list of GB values.

    Uses torch.cuda.mem_get_info() which accounts for other processes.
    """
    if torch.cuda.is_available():
        free_list = []
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            free_list.append(free / (1024**3))
        return free_list
    return []


def calculate_context_size(estimated_prompt_tokens: int) -> int:
    """Calculate context size with fixed 8k headspace for generation.

    Simply adds 8k to the estimated prompt tokens to provide headspace
    for response generation without over-allocating.
    """
    return estimated_prompt_tokens + 8192


def get_vram_usage_gb(gpu_index: int = 0) -> float:
    """Get current VRAM usage in GB for a specific GPU."""
    if torch.cuda.is_available() and gpu_index < torch.cuda.device_count():
        return torch.cuda.memory_allocated(gpu_index) / (1024**3)
    return 0.0


def get_total_vram_gb(gpu_index: int = None) -> float:
    """Get total VRAM in GB.

    Args:
        gpu_index: Specific GPU index, or None to sum all GPUs.
    """
    if torch.cuda.is_available():
        if gpu_index is not None:
            if gpu_index < torch.cuda.device_count():
                return torch.cuda.get_device_properties(gpu_index).total_memory / (
                    1024**3
                )
            return 0.0
        else:
            # Sum all GPUs
            total = 0.0
            for i in range(torch.cuda.device_count()):
                total += torch.cuda.get_device_properties(i).total_memory
            return total / (1024**3)
    return 0.0


def calculate_tensor_split() -> list:
    """Calculate tensor split ratios based on available VRAM per GPU.

    DEPRECATED: Use calculate_tensor_split_from_free_vram() for accurate splits.

    Returns a list of 128 floats (xllamacpp expects exactly 128).
    Non-zero values indicate relative VRAM proportions for each GPU.
    """
    if not torch.cuda.is_available():
        return [0.0] * 128

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        return [0.0] * 128

    # Get VRAM for each GPU
    vram_per_gpu = []
    for i in range(gpu_count):
        vram = torch.cuda.get_device_properties(i).total_memory
        vram_per_gpu.append(vram)

    total_vram = sum(vram_per_gpu)

    # Calculate proportional splits
    tensor_split = [0.0] * 128
    for i, vram in enumerate(vram_per_gpu):
        tensor_split[i] = vram / total_vram if total_vram > 0 else 0.0

    return tensor_split


def extract_gpu_generation_score(gpu_name: str) -> Tuple[int, int, int]:
    """Extract a capability score from NVIDIA GPU name.

    Parses GPU names like "NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 3090",
    "NVIDIA A100", etc. and returns a tuple for comparison.

    Returns:
        Tuple of (generation, tier, variant) where higher is better.
        - generation: 50 for 5000 series, 40 for 4000 series, etc.
        - tier: 90 for x090, 80 for x080, 70 for x070, etc.
        - variant: Ti/Super variants get bonus points
    """
    import re

    gpu_name_upper = gpu_name.upper()

    # Default low score for unknown GPUs
    generation = 0
    tier = 0
    variant = 0

    # Check for datacenter/professional cards first (A100, H100, etc.)
    datacenter_match = re.search(r"\b([AH])(\d{2,3})\b", gpu_name_upper)
    if datacenter_match:
        prefix = datacenter_match.group(1)
        number = int(datacenter_match.group(2))
        # H100 > A100 > A40, etc.
        if prefix == "H":
            generation = 100
        elif prefix == "A":
            generation = 90
        tier = number
        return (generation, tier, variant)

    # RTX/GTX consumer cards - extract the model number
    # Matches: RTX 5090, RTX 4090, RTX 3090, GTX 1080, etc.
    rtx_match = re.search(r"\b(?:RTX|GTX)\s*(\d)(\d{2,3})\b", gpu_name_upper)
    if rtx_match:
        gen_digit = int(rtx_match.group(1))  # 5, 4, 3, 2, 1
        tier_digits = rtx_match.group(2)  # 090, 080, 070, 80, 70

        # Normalize generation (5xxx = 50, 4xxx = 40, etc.)
        generation = gen_digit * 10

        # Normalize tier (90, 80, 70, 60, 50)
        if len(tier_digits) == 3:
            tier = int(tier_digits[0:2])  # 090 -> 90
        else:
            tier = int(tier_digits)  # 80 -> 80

        # Check for Ti/Super variants
        if "TI" in gpu_name_upper:
            variant = 5
        elif "SUPER" in gpu_name_upper:
            variant = 3

    # Quadro cards
    quadro_match = re.search(r"QUADRO\s*(?:RTX\s*)?(\d+)", gpu_name_upper)
    if quadro_match:
        number = int(quadro_match.group(1))
        generation = 35  # Place between GTX and newer RTX
        tier = number // 100 if number >= 1000 else number // 10

    return (generation, tier, variant)


def get_gpu_capability_ranking() -> List[Tuple[int, float, str]]:
    """Get GPUs ranked by capability (most powerful first).

    Returns:
        List of tuples: (gpu_index, capability_score, gpu_name)
        Sorted by capability_score descending (most powerful first).
    """
    if not torch.cuda.is_available():
        return []

    gpu_info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpu_name = props.name

        # Get generation/tier score from name
        gen_score = extract_gpu_generation_score(gpu_name)

        # Also factor in compute capability and VRAM as tiebreakers
        compute_capability = props.major * 10 + props.minor
        total_vram_gb = props.total_memory / (1024**3)

        # Combined score: prioritize generation, then tier, then variant,
        # then compute capability, then VRAM
        # Score = gen*10000 + tier*100 + variant*10 + compute_cap + vram/100
        capability_score = (
            gen_score[0] * 10000
            + gen_score[1] * 100
            + gen_score[2] * 10
            + compute_capability
            + total_vram_gb / 100
        )

        gpu_info.append((i, capability_score, gpu_name))

    # Sort by capability score descending
    gpu_info.sort(key=lambda x: x[1], reverse=True)

    return gpu_info


def get_gpus_by_priority() -> List[int]:
    """Get GPU indices ordered by capability (most powerful first).

    Returns:
        List of GPU indices, ordered from most to least capable.
    """
    ranking = get_gpu_capability_ranking()
    return [gpu_idx for gpu_idx, _, _ in ranking]


def get_primary_gpu() -> int:
    """Get the index of the most powerful GPU (for LLM loading)."""
    gpus = get_gpus_by_priority()
    return gpus[0] if gpus else 0


def get_secondary_gpu() -> Optional[int]:
    """Get the index of the second most powerful GPU (for image gen / non-LLM models).

    Returns None if only one GPU is available.
    """
    gpus = get_gpus_by_priority()
    return gpus[1] if len(gpus) > 1 else None


def estimate_model_vram_requirement(
    model_path: str, context_size: int, projectors: list = None
) -> float:
    """Estimate VRAM requirement for a model in GB.

    Uses xllamacpp's estimate_gpu_layers to get memory estimates.
    Returns estimated VRAM in GB, or a conservative estimate if xllamacpp is unavailable.
    """

    # Better fallback estimation formula based on actual llama.cpp memory usage:
    # - KV cache (F16): context_size * 2 bytes * 2 (K+V) * n_layer * n_embd / n_head
    #   For typical models: ~0.14GB per 1K context for 4B models, ~0.35GB for larger
    # - Model weights: varies by quantization
    # - Compute buffers: ~0.5GB
    # Simplified: model_base + (context/1000) * kv_factor
    def fallback_estimate(ctx_size: int) -> float:
        # Get model file size to estimate base requirement
        model_size_gb = 4.0  # Default assumption for 4B Q4 models
        try:
            if model_path and os.path.exists(model_path):
                model_size_gb = (
                    os.path.getsize(model_path) / (1024**3) * 1.2
                )  # 20% overhead
        except Exception:
            pass

        # KV cache estimate: scales with context size but NOT linearly with model size.
        # Modern models use GQA/MQA (fewer KV heads than Q heads), so KV cache is much
        # smaller than the model weight size would suggest. Cap the factor to prevent
        # wild overestimates (e.g., 80GB estimate for a model that actually needs 21GB).
        # 0.1 GB/1K is conservative enough for most models up to ~70B.
        kv_factor = min(0.14 * (model_size_gb / 2.5), 0.1)
        kv_estimate = (ctx_size / 1000) * kv_factor

        # Compute buffers and overhead: ~1GB
        overhead = 1.0

        total = model_size_gb + kv_estimate + overhead
        logging.debug(
            f"[GPU Selection] Fallback VRAM estimate: {total:.1f}GB "
            f"(model={model_size_gb:.1f}GB, KV={kv_estimate:.1f}GB, overhead={overhead:.1f}GB)"
        )
        return total

    if not xllamacpp_available:
        return fallback_estimate(context_size)

    try:
        # Create a fake GPU with unlimited memory to get total requirement
        fake_gpu = {
            "name": "Virtual GPU",
            "memory_free": 1024 * 1024 * 1024 * 1024,  # 1TB
            "memory_total": 1024 * 1024 * 1024 * 1024,
        }

        result = estimate_gpu_layers(
            gpus=[fake_gpu],
            model_path=model_path,
            projectors=projectors or [],
            context_length=context_size,
            batch_size=2048,
            num_parallel=1,
            kv_cache_type="q4_0",
        )

        # Extract memory requirement
        estimated_gb = None
        if hasattr(result, "memory"):
            estimated_gb = result.memory / (1024**3)
        elif isinstance(result, dict) and "memory" in result:
            estimated_gb = result["memory"] / (1024**3)

        if estimated_gb is not None:
            # Sanity check: xllamacpp sometimes overestimates significantly
            # Compare with our fallback and use the lower of the two if xllamacpp
            # returns more than 2x our estimate
            fallback = fallback_estimate(context_size)
            if estimated_gb > fallback * 2.5:
                logging.warning(
                    f"[GPU Selection] xllamacpp estimate ({estimated_gb:.1f}GB) seems too high, "
                    f"using adjusted estimate ({fallback * 1.3:.1f}GB)"
                )
                return fallback * 1.3  # Add 30% safety margin to fallback
            logging.debug(
                f"[GPU Selection] xllamacpp VRAM estimate: {estimated_gb:.1f}GB "
                f"(fallback would be {fallback:.1f}GB)"
            )
            return estimated_gb

        # Fallback estimation
        return fallback_estimate(context_size)

    except Exception as e:
        logging.warning(f"[GPU Selection] Failed to estimate VRAM: {e}")
        # When xllamacpp estimation fails (e.g., unknown quantization type like Q4_K_XL),
        # use a file-size-based estimate plus context-dependent KV/overhead costs.
        try:
            if model_path and os.path.exists(model_path):
                file_size_gb = os.path.getsize(model_path) / (1024**3)
                # Model weights ~ file size. Add context-dependent costs:
                # - KV cache (q4_0): ~5.5 KB/token for hybrid models with few attn layers
                #   For standard models it would be higher. Use 6KB as conservative default.
                # - mmproj/CLIP: ~0.86 GB if projector exists
                # - Compute buffers: ~0.3-0.5 GB
                # - RS (recurrent state for SSM): ~0.065 GB
                kv_gb = (context_size * 6000) / (1024**3)
                mmproj_gb = 0.0
                if projectors:
                    for proj in projectors:
                        try:
                            if proj and os.path.exists(proj):
                                mmproj_gb += os.path.getsize(proj) / (1024**3)
                        except Exception:
                            mmproj_gb += 0.86  # Conservative default
                overhead_gb = 0.5  # compute buffers, RS, misc
                estimate = file_size_gb + kv_gb + mmproj_gb + overhead_gb
                logging.info(
                    f"[GPU Selection] Using file-size estimate: {estimate:.1f}GB "
                    f"(model={file_size_gb:.1f}GB, KV={kv_gb:.2f}GB, "
                    f"mmproj={mmproj_gb:.1f}GB, overhead={overhead_gb:.1f}GB) "
                    f"for {context_size//1024}k context"
                )
                return estimate
        except Exception:
            pass
        return fallback_estimate(context_size)


def _estimate_optimal_layers(
    model_path: str,
    context_size: int,
    projectors: list = None,
    available_vram_per_gpu: list = None,
) -> int:
    """Estimate optimal number of GPU layers for partial offloading.

    Uses xllamacpp's estimate_gpu_layers to determine how many layers
    can fit in the available VRAM.

    Args:
        model_path: Path to the model file
        context_size: Context window size
        projectors: List of projector paths (for vision models)
        available_vram_per_gpu: List of available VRAM per GPU in GB

    Returns:
        Number of layers that can fit on GPU, or 0 if estimation fails
    """
    if not xllamacpp_available or not available_vram_per_gpu:
        return 0

    try:
        # Build GPU info list for xllamacpp
        gpus = []
        for i, avail_gb in enumerate(available_vram_per_gpu):
            avail_bytes = int(avail_gb * 1024 * 1024 * 1024)
            gpus.append(
                {
                    "name": f"GPU {i}",
                    "memory_free": avail_bytes,
                    "memory_total": avail_bytes,  # Use available as total for estimation
                }
            )

        result = estimate_gpu_layers(
            gpus=gpus,
            model_path=model_path,
            projectors=projectors or [],
            context_length=context_size,
            batch_size=2048,
            num_parallel=1,
            kv_cache_type="q4_0",
        )

        # Extract layer count from result
        if hasattr(result, "layers"):
            return result.layers
        elif isinstance(result, dict):
            return result.get(
                "layers", result.get("gpu_layers", result.get("n_gpu_layers", 0))
            )
        elif isinstance(result, int):
            return result

        return 0

    except Exception as e:
        logging.warning(f"[GPU Selection] Failed to estimate optimal layers: {e}")
        return 0


def determine_gpu_strategy(
    model_path: str,
    context_size: int,
    projectors: list = None,
    reserved_vram: float = 5.0,
) -> Dict[str, Any]:
    """Determine optimal GPU loading strategy based on available VRAM and GPU capability.

    Smart GPU selection logic (GPUs ordered by capability, not nvidia-smi index):
    1. If the most powerful GPU has enough free VRAM → load on it only
    2. If combined GPUs have enough → tensor split across them (weighted by free VRAM)
    3. If most powerful is full but another GPU has enough → load on that GPU only
    4. Otherwise → fall back to CPU

    Args:
        model_path: Path to the model file
        context_size: Context window size
        projectors: List of projector paths (for vision models)
        reserved_vram: VRAM to reserve for TTS/STT (default 5GB)

    Returns:
        Dict with keys:
        - main_gpu: GPU index to use as primary (0, 1, etc.)
        - tensor_split: List of 128 floats for tensor splitting, or None for single GPU
        - gpu_layers: Number of GPU layers, or 0 for CPU-only
        - strategy: Description of the strategy ("gpu0", "gpu1", "tensor_split", "cpu")
    """
    gpu_count = get_gpu_count()

    if gpu_count == 0 or not torch.cuda.is_available():
        return {"main_gpu": 0, "tensor_split": None, "gpu_layers": 0, "strategy": "cpu"}

    # Get GPU capability ranking (most powerful first)
    gpu_ranking = get_gpu_capability_ranking()
    gpus_by_priority = [gpu_idx for gpu_idx, _, _ in gpu_ranking]

    # Log GPU ranking (debug-level to reduce noise)
    logging.debug(f"[GPU Selection] GPU capability ranking (most powerful first):")
    for gpu_idx, score, name in gpu_ranking:
        logging.debug(f"[GPU Selection]   GPU {gpu_idx}: {name} (score: {score:.1f})")

    # Get free VRAM for each GPU
    free_vram = get_per_gpu_free_vram_gb()
    total_vram = get_per_gpu_vram_gb()

    # Estimate model VRAM requirement
    estimated_vram = estimate_model_vram_requirement(
        model_path, context_size, projectors
    )

    # Account for reserved VRAM
    # For single-GPU: reserve on the only GPU
    # For multi-GPU: initial estimate uses even distribution; the multi-GPU section
    # below recalculates to reserve only on secondary GPUs
    if gpu_count == 1:
        reserved_per_gpu = reserved_vram
    else:
        reserved_per_gpu = 0  # Multi-GPU path recalculates below
    available_vram = [max(0, free - reserved_per_gpu) for free in free_vram]

    logging.info(
        f"[GPU Selection] Model VRAM estimate: {estimated_vram:.1f}GB for {context_size//1000}k context"
    )
    for i in range(gpu_count):
        if reserved_per_gpu > 0:
            logging.info(
                f"[GPU Selection]   GPU {i}: {free_vram[i]:.1f}GB free, "
                f"{available_vram[i]:.1f}GB available after {reserved_per_gpu:.1f}GB reservation"
            )
        else:
            logging.info(f"[GPU Selection]   GPU {i}: {free_vram[i]:.1f}GB free")

    # Single GPU case
    if gpu_count == 1:
        if available_vram[0] >= estimated_vram:
            return {
                "main_gpu": 0,
                "tensor_split": None,
                "gpu_layers": -1,  # Auto-detect (all layers on GPU)
                "strategy": "gpu0",
            }
        else:
            # Not enough VRAM for full model - try partial offloading
            # Use xllamacpp to estimate how many layers can fit
            optimal_layers = _estimate_optimal_layers(
                model_path, context_size, projectors, [available_vram[0]]
            )
            if optimal_layers and optimal_layers > 0:
                logging.info(
                    f"[GPU Selection] GPU 0 has {available_vram[0]:.1f}GB available, "
                    f"need {estimated_vram:.1f}GB - using partial offload ({optimal_layers} layers on GPU)"
                )
                return {
                    "main_gpu": 0,
                    "tensor_split": None,
                    "gpu_layers": optimal_layers,
                    "strategy": "gpu0_partial",
                }
            else:
                # Check if this might be a bad estimate: if the model file itself fits
                # in VRAM, try full GPU offload anyway. The VRAM estimate can be wildly
                # wrong for newer quant types or GQA/MQA models. The resilient loader
                # will fall back to partial offload if GPU loading actually fails with OOM.
                try:
                    if model_path and os.path.exists(model_path):
                        file_size_gb = os.path.getsize(model_path) / (1024**3)
                        vram_after_model = available_vram[0] - file_size_gb
                        # Estimate non-model costs (KV, mmproj, compute, RS)
                        kv_gb = (context_size * 6000) / (1024**3)
                        mmproj_gb = (
                            sum(
                                os.path.getsize(p) / (1024**3)
                                for p in (projectors or [])
                                if p and os.path.exists(p)
                            )
                            if projectors
                            else 0.0
                        )
                        non_model_gb = kv_gb + mmproj_gb + 0.5  # +0.5 for compute/RS
                        if vram_after_model >= non_model_gb:
                            logging.info(
                                f"[GPU Selection] VRAM estimate ({estimated_vram:.1f}GB) exceeds "
                                f"available ({available_vram[0]:.1f}GB), but detailed check shows "
                                f"model({file_size_gb:.1f}GB) + KV({kv_gb:.2f}GB) + "
                                f"mmproj({mmproj_gb:.1f}GB) fits - trying full GPU offload"
                            )
                            return {
                                "main_gpu": 0,
                                "tensor_split": None,
                                "gpu_layers": -1,
                                "strategy": "gpu0_optimistic",
                            }
                        elif available_vram[0] > file_size_gb:
                            # Model fits but not everything else — partial offload
                            # Estimate how many layers to offload to CPU to free VRAM
                            shortfall_gb = (
                                file_size_gb + non_model_gb
                            ) - available_vram[0]
                            # Each layer is roughly model_size / n_layers
                            n_layers = 41  # Default
                            try:
                                import gguf

                                reader = gguf.GGUFReader(model_path)
                                for kv in reader.fields.values():
                                    if kv.name and "block_count" in kv.name:
                                        n_layers = int(kv.parts[-1][0]) + 1
                                        break
                            except Exception:
                                pass
                            gb_per_layer = file_size_gb / n_layers
                            layers_to_offload = (
                                int(shortfall_gb / gb_per_layer) + 2
                            )  # +2 safety margin
                            gpu_layers = max(1, n_layers - layers_to_offload)
                            logging.info(
                                f"[GPU Selection] Need {shortfall_gb:.1f}GB more than available. "
                                f"Offloading {layers_to_offload} layers to CPU, keeping "
                                f"{gpu_layers}/{n_layers} on GPU"
                            )
                            return {
                                "main_gpu": 0,
                                "tensor_split": None,
                                "gpu_layers": gpu_layers,
                                "strategy": "gpu0_partial_optimistic",
                            }
                except Exception:
                    pass

                # Fall back to CPU
                logging.warning(
                    f"[GPU Selection] GPU 0 has {available_vram[0]:.1f}GB available, "
                    f"need {estimated_vram:.1f}GB, falling back to CPU"
                )
                return {
                    "main_gpu": 0,
                    "tensor_split": None,
                    "gpu_layers": 0,
                    "strategy": "cpu",
                }

    # Helper: create tensor_split that routes 100% to a single GPU.
    # llama.cpp treats all-zeros tensor_split as "auto-distribute by VRAM",
    # so we MUST set an explicit split when we want single-GPU loading on multi-GPU systems.
    def make_single_gpu_split(target_gpu: int) -> list:
        ts = [0.0] * 128
        ts[target_gpu] = 1.0
        return ts

    # Multi-GPU case - use capability ranking
    # The primary GPU (most powerful) is dedicated to the LLM.
    # Non-LLM models (image gen, TTS, STT) go on secondary GPUs.
    # Only reserve VRAM on the secondary GPU(s), not on the primary.
    primary_gpu = gpus_by_priority[0]
    primary_name = gpu_ranking[0][2]

    # Recalculate available VRAM: reserve only on non-primary GPUs
    available_vram = list(free_vram)  # Start with raw free VRAM
    for i in range(gpu_count):
        if i != primary_gpu:
            available_vram[i] = max(0, free_vram[i] - reserved_vram)
        # Primary GPU gets full free VRAM — non-LLM models go elsewhere

    logging.info(
        f"[GPU Selection] Multi-GPU: primary=GPU {primary_gpu} ({primary_name}), "
        f"reserve {reserved_vram:.1f}GB on secondary GPUs only"
    )
    for i in range(gpu_count):
        logging.info(
            f"[GPU Selection]   GPU {i}: {free_vram[i]:.1f}GB free, "
            f"{available_vram[i]:.1f}GB available"
            + (
                " [PRIMARY - no reservation]"
                if i == primary_gpu
                else f" [reserved {reserved_vram:.1f}GB]"
            )
        )

    # Strategy 1: Try most powerful GPU alone (most common for high-end GPUs)
    if available_vram[primary_gpu] >= estimated_vram:
        logging.info(
            f"[GPU Selection] Primary GPU {primary_gpu} ({primary_name}) has {available_vram[primary_gpu]:.1f}GB available, "
            f"sufficient for {estimated_vram:.1f}GB model - loading on GPU {primary_gpu} only"
        )
        return {
            "main_gpu": primary_gpu,
            "tensor_split": make_single_gpu_split(primary_gpu),
            "gpu_layers": -1,
            "strategy": f"gpu{primary_gpu}",
        }

    # Strategy 1b: Optimistic file-size check on primary GPU
    # The VRAM estimate can be wrong for newer quant types. If model weights + KV + overhead
    # fit on the primary GPU, try loading there before falling back to tensor split.
    try:
        if model_path and os.path.exists(model_path):
            file_size_gb = os.path.getsize(model_path) / (1024**3)
            kv_gb = (context_size * 6000) / (1024**3)
            mmproj_gb = (
                sum(
                    os.path.getsize(p) / (1024**3)
                    for p in (projectors or [])
                    if p and os.path.exists(p)
                )
                if projectors
                else 0.0
            )
            actual_need = file_size_gb + kv_gb + mmproj_gb + 0.5  # +0.5 for compute/RS
            if available_vram[primary_gpu] >= actual_need:
                logging.info(
                    f"[GPU Selection] VRAM estimate ({estimated_vram:.1f}GB) exceeds primary GPU VRAM "
                    f"({available_vram[primary_gpu]:.1f}GB), but detailed check shows "
                    f"model({file_size_gb:.1f}GB) + KV({kv_gb:.2f}GB) + mmproj({mmproj_gb:.1f}GB) = "
                    f"{actual_need:.1f}GB fits - loading on GPU {primary_gpu} only"
                )
                return {
                    "main_gpu": primary_gpu,
                    "tensor_split": make_single_gpu_split(primary_gpu),
                    "gpu_layers": -1,
                    "strategy": f"gpu{primary_gpu}_optimistic",
                }
            elif available_vram[primary_gpu] > file_size_gb:
                # Model fits but not all overhead — partial offload on primary GPU
                shortfall_gb = actual_need - available_vram[primary_gpu]
                n_layers = 41  # Default
                try:
                    import gguf

                    reader = gguf.GGUFReader(model_path)
                    for kv in reader.fields.values():
                        if kv.name and "block_count" in kv.name:
                            n_layers = int(kv.parts[-1][0]) + 1
                            break
                except Exception:
                    pass
                gb_per_layer = file_size_gb / n_layers
                layers_to_offload = int(shortfall_gb / gb_per_layer) + 2  # +2 safety
                gpu_layers = max(1, n_layers - layers_to_offload)
                logging.info(
                    f"[GPU Selection] Primary GPU needs {shortfall_gb:.1f}GB more. "
                    f"Partial offload: {gpu_layers}/{n_layers} layers on GPU {primary_gpu}"
                )
                return {
                    "main_gpu": primary_gpu,
                    "tensor_split": make_single_gpu_split(primary_gpu),
                    "gpu_layers": gpu_layers,
                    "strategy": f"gpu{primary_gpu}_partial_optimistic",
                }
    except Exception as e:
        logging.debug(f"[GPU Selection] Optimistic check failed: {e}")

    # Strategy 2: Try tensor split across all GPUs (primary + secondary)
    total_available = sum(available_vram)
    if total_available >= estimated_vram:
        # Calculate tensor split based on FREE VRAM proportions
        tensor_split = [0.0] * 128
        for i, avail in enumerate(available_vram):
            tensor_split[i] = avail / total_available if total_available > 0 else 0.0

        logging.info(
            f"[GPU Selection] Tensor splitting across {gpu_count} GPUs "
            f"(total: {total_available:.1f}GB available, need: {estimated_vram:.1f}GB)"
        )
        logging.debug(f"[GPU Selection] Split ratios: {tensor_split[:gpu_count]}")

        # Use the most powerful GPU as main_gpu for tensor split
        return {
            "main_gpu": primary_gpu,
            "tensor_split": tensor_split,
            "gpu_layers": -1,
            "strategy": "tensor_split",
        }

    # Strategy 3: Try other GPUs in order of capability
    for gpu_idx in gpus_by_priority[1:]:  # Skip the primary GPU we already tried
        if available_vram[gpu_idx] >= estimated_vram:
            gpu_name = next(
                (name for idx, _, name in gpu_ranking if idx == gpu_idx),
                f"GPU {gpu_idx}",
            )
            logging.info(
                f"[GPU Selection] Primary GPU {primary_gpu} full ({available_vram[primary_gpu]:.1f}GB), "
                f"but GPU {gpu_idx} ({gpu_name}) has {available_vram[gpu_idx]:.1f}GB - loading on GPU {gpu_idx} only"
            )
            return {
                "main_gpu": gpu_idx,
                "tensor_split": make_single_gpu_split(gpu_idx),
                "gpu_layers": -1,
                "strategy": f"gpu{gpu_idx}",
            }

    # Strategy 4: Try partial offloading with tensor split
    # Model doesn't fit entirely but we can offload some layers to GPUs
    total_available = sum(available_vram)
    if total_available > 2:  # At least 2GB available across all GPUs
        optimal_layers = _estimate_optimal_layers(
            model_path, context_size, projectors, available_vram
        )
        if optimal_layers and optimal_layers > 0:
            # Calculate tensor split for partial offloading
            tensor_split = [0.0] * 128
            for i, avail in enumerate(available_vram):
                tensor_split[i] = (
                    avail / total_available if total_available > 0 else 0.0
                )

            logging.info(
                f"[GPU Selection] Partial offload: {optimal_layers} layers across {gpu_count} GPUs "
                f"(total: {total_available:.1f}GB available, need: {estimated_vram:.1f}GB for full model)"
            )
            return {
                "main_gpu": primary_gpu,
                "tensor_split": tensor_split,
                "gpu_layers": optimal_layers,
                "strategy": "tensor_split_partial",
            }

    # Strategy 5: File-size safety check before CPU fallback
    # If the model file fits in total available VRAM, try full offload.
    # VRAM estimates can be wrong for newer quant types; let llama.cpp decide.
    try:
        if model_path and os.path.exists(model_path):
            file_size_gb = os.path.getsize(model_path) / (1024**3)
            if total_available > file_size_gb:
                logging.info(
                    f"[GPU Selection] VRAM estimate ({estimated_vram:.1f}GB) exceeds "
                    f"available ({total_available:.1f}GB), but model file "
                    f"({file_size_gb:.1f}GB) fits - trying full GPU offload"
                )
                return {
                    "main_gpu": primary_gpu,
                    "tensor_split": make_single_gpu_split(primary_gpu),
                    "gpu_layers": -1,
                    "strategy": f"gpu{primary_gpu}_optimistic",
                }
    except Exception:
        pass

    # Strategy 6: Fall back to CPU
    logging.warning(
        f"[GPU Selection] Insufficient VRAM across all GPUs "
        f"(need {estimated_vram:.1f}GB, have {total_available:.1f}GB), falling back to CPU"
    )
    return {"main_gpu": 0, "tensor_split": None, "gpu_layers": 0, "strategy": "cpu"}


# Model-specific config overrides for optimal inference settings
# These override user-provided values for known models
MODEL_CONFIG_OVERRIDES = {
    "unsloth/Qwen3-VL-4B-Instruct-GGUF": {
        "top_p": 0.8,
        "top_k": 20,
        "temperature": 0.7,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
    },
    "unsloth/Qwen3-4B-Instruct-2507-GGUF": {
        "top_p": 1.0,
        "top_k": 40,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 2.0,
    },
    "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF": {
        "top_p": 0.8,
        "top_k": 20,
        "temperature": 0.7,
        "repetition_penalty": 1.05,
    },
    # Qwen3.5 models use hybrid Gated DeltaNet + sparse MoE architecture.
    # Only 10/40 layers use standard attention (2 KV heads), so KV cache is tiny.
    # Recommended coding-mode params from HuggingFace model card.
    "unsloth/Qwen3.5-35B-A3B-GGUF": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        # Non-thinking mode by default; send chat_template_kwargs={"enable_thinking": True}
        # in the request to enable thinking (uses <think> tags).
        "chat_template_kwargs": {"enable_thinking": False},
    },
}


class Pipes:
    def __init__(self):
        load_dotenv()
        global img_import_success

        # Lock for model access - prevents race conditions when multiple
        # requests try to load/switch models simultaneously
        self._model_lock = threading.Lock()
        # Track how many inferences are currently in progress (thread-safe counter)
        # Using a counter instead of boolean allows multiple concurrent requests
        self._inference_count = 0
        self._inference_count_lock = threading.Lock()

        # Track context reset state for auto-optimization
        # When we increase context for a large request, we schedule a reset
        # back to optimal context after a cooldown period
        self._context_reset_timer = None
        self._context_reset_lock = threading.Lock()
        self._optimal_context = int(
            getenv("LLM_MAX_TOKENS", "40000")
        )  # Default optimal context
        self._context_reset_cooldown = 60  # Seconds to wait before resetting context

        # Initialize resource manager for intelligent VRAM management
        # Use the global singleton so all components share the same state
        self.resource_manager = get_resource_manager()

        # Check if precache already ran (models already downloaded/warmed)
        from pathlib import Path

        precache_done = Path("/tmp/ezlocalai_precache.done").exists()
        if precache_done:
            logging.debug(
                "[Init] Precache completed, skipping redundant warmup operations"
            )

        # Auto-detect multi-GPU configuration
        self.gpu_count = get_gpu_count()
        self.per_gpu_vram = get_per_gpu_vram_gb()

        # Auto-detect total VRAM budget across all GPUs (rounded down to nearest 1GB for safety margin)
        self.vram_budget_gb = get_available_vram_gb()

        # Calculate tensor split for multi-GPU
        self.tensor_split = calculate_tensor_split()

        if self.vram_budget_gb > 0:
            if self.gpu_count > 1:
                logging.debug(
                    f"[VRAM] Multi-GPU detected: {self.gpu_count} GPUs, "
                    f"{self.vram_budget_gb}GB total VRAM budget "
                    f"(per GPU: {self.per_gpu_vram})"
                )
            else:
                logging.debug(
                    f"[VRAM] Auto-detected {self.vram_budget_gb}GB VRAM budget"
                )

        # Parse model list: "model1,model2" (simple comma-separated)
        model_config = getenv("DEFAULT_MODEL")
        self.available_models = []  # List of model names
        self.calibrated_gpu_layers = {}  # {model_name: {context: gpu_layers}}

        # Persistent LLM instances (kept loaded to avoid reload overhead)
        self.primary_llm = (
            None  # First non-vision model (or first model if all are vision)
        )
        self.primary_llm_name = None
        self.primary_llm_context = None

        self.vision_llm = None  # Vision model (if different from primary)
        self.vision_llm_name = None
        self.vision_llm_context = None

        # Active LLM pointer (points to one of the above, or a temp large model)
        self.llm = None
        self.current_llm_name = None
        self.current_context = None  # Track current context size

        # Track if we're using a "large" model that should be unloaded after use
        self._using_large_model = False

        if model_config.lower() != "none":
            for model_entry in model_config.split(","):
                model_name = model_entry.strip()
                # Strip any legacy @tokens suffix for backward compat
                if "@" in model_name:
                    model_name = model_name.rsplit("@", 1)[0]
                if model_name and model_name not in self.available_models:
                    self.available_models.append(model_name)

            # Pre-load persistent LLMs (primary + vision if different)
            # Skip LLM preloading in voice server mode - only load voice models
            if is_voice_server_mode():
                logging.info(
                    "[LLM] Voice server mode - skipping LLM preload. "
                    "LLMs will be lazy-loaded on first request."
                )
            elif self.available_models:
                # Pre-load with optimal context (default 40k) to maximize GPU layers
                # while providing reasonable context for most requests
                default_context = self._optimal_context

                # Find primary (first non-vision) and vision models
                primary_model = None
                vision_model = None

                for model_name in self.available_models:
                    is_vision = self._is_vision_model(model_name)
                    if is_vision and vision_model is None:
                        vision_model = model_name
                    if not is_vision and primary_model is None:
                        primary_model = model_name

                # If no non-vision model, use first model as primary
                if primary_model is None:
                    primary_model = self.available_models[0]
                    # If primary is vision, don't load it twice
                    if primary_model == vision_model:
                        vision_model = None

                # Load primary model
                logging.info(f"[LLM] Pre-loading primary model: {primary_model}...")
                start_time = time.time()
                try:
                    self.primary_llm = self._load_llm_resilient(
                        model_name=primary_model,
                        max_tokens=default_context,
                    )
                    self.primary_llm_name = primary_model
                    self.primary_llm_context = default_context
                    self.llm = self.primary_llm
                    self.current_llm_name = primary_model
                    self.current_context = default_context
                    load_time = time.time() - start_time
                    logging.info(
                        f"[LLM] Primary model {primary_model} loaded in {load_time:.1f}s"
                    )
                    # Register with resource manager
                    is_vision = self._is_vision_model(primary_model)
                    model_type = ModelType.VISION_LLM if is_vision else ModelType.LLM
                    self.resource_manager.register_model(
                        model_type,
                        primary_model,
                        "cuda",
                        vram_gb=MODEL_VRAM_ESTIMATES.get(model_type, 8.0),
                    )
                except Exception as e:
                    logging.warning(
                        f"[LLM] Failed to pre-load primary model {primary_model}: {e}"
                    )

                # Load vision model if different from primary
                # Vision models don't need large context - they process images and generate
                # relatively short descriptions. Use smaller context to maximize GPU layer offload.
                vision_context = int(getenv("VLM_MAX_TOKENS", "8192"))
                if vision_model and vision_model != primary_model:
                    logging.info(
                        f"[LLM] Pre-loading vision model: {vision_model} (context: {vision_context})..."
                    )
                    start_time = time.time()
                    try:
                        self.vision_llm = self._load_llm_resilient(
                            model_name=vision_model,
                            max_tokens=vision_context,
                        )
                        self.vision_llm_name = vision_model
                        self.vision_llm_context = vision_context
                        load_time = time.time() - start_time
                        logging.info(
                            f"[LLM] Vision model {vision_model} loaded in {load_time:.1f}s"
                        )
                        # Register with resource manager
                        self.resource_manager.register_model(
                            ModelType.VISION_LLM,
                            vision_model,
                            "cuda",
                            vram_gb=MODEL_VRAM_ESTIMATES.get(ModelType.VISION_LLM, 6.0),
                        )
                    except Exception as e:
                        logging.warning(
                            f"[LLM] Failed to pre-load vision model {vision_model}: {e}"
                        )

        # TTS initialization - Chatterbox TTS
        self.ctts = None
        if getenv("TTS_ENABLED").lower() == "true":
            tts_name = self._get_tts_name()
            tts_vram = 4.0  # Chatterbox uses about 4GB VRAM

            # Skip local TTS loading if voice server URL is configured (passthrough mode)
            if has_voice_server_url():
                voice_url = getenv("VOICE_SERVER")
                logging.info(
                    f"[TTS] Voice server configured ({voice_url}) - skipping local model loading"
                )
            # Check if we should preload TTS (voice server mode OR LAZY_LOAD_VOICE=false)
            elif should_preload_voice():
                mode_str = (
                    "voice server mode"
                    if is_voice_server_mode()
                    else "LAZY_LOAD_VOICE=false"
                )
                logging.info(f"[TTS] {mode_str} - loading {tts_name} to keep resident")
                start_time = time.time()
                self.ctts = self._create_tts_model()
                load_time = time.time() - start_time
                logging.info(
                    f"[TTS] {tts_name} loaded in {load_time:.2f}s ({mode_str} - staying loaded)"
                )
                self.resource_manager.register_model(
                    ModelType.TTS,
                    tts_name,
                    "cuda" if torch.cuda.is_available() else "cpu",
                    vram_gb=tts_vram if torch.cuda.is_available() else 0.0,
                )
            elif precache_done:
                # Precache already warmed the TTS cache, skip loading/unloading
                logging.debug(
                    f"[TTS] {tts_name} cache already warmed by precache, will lazy-load on first request"
                )
            else:
                # No precache - warm the cache now (first run or single-worker mode)
                logging.debug(f"[TTS] Preloading {tts_name} to warm cache...")
                start_time = time.time()
                self.ctts = self._create_tts_model()
                load_time = time.time() - start_time
                logging.debug(
                    f"[TTS] {tts_name} preloaded in {load_time:.2f}s, unloading to free VRAM..."
                )
                self._destroy_tts()
                logging.debug(
                    "[TTS] TTS unloaded, will lazy-load on first TTS request."
                )

        # Lazy-loaded models (loaded on first use, destroyed after)
        self.stt = None
        self.embedder = None
        self.img = None
        self.current_stt = getenv("WHISPER_MODEL")

        # Pre-load STT if preloading is enabled (voice server mode OR LAZY_LOAD_VOICE=false)
        # Skip if voice server URL is configured (passthrough mode)
        if (
            should_preload_voice()
            and getenv("STT_ENABLED").lower() == "true"
            and not has_voice_server_url()
        ):
            mode_str = (
                "voice server mode"
                if is_voice_server_mode()
                else "LAZY_LOAD_VOICE=false"
            )
            logging.info(f"[STT] {mode_str} - loading STT to keep resident")
            from ezlocalai.STT import STT

            start_time = time.time()
            self.stt = STT(model=self.current_stt)
            load_time = time.time() - start_time
            actual_device = getattr(self.stt, "device", "cpu")
            logging.info(
                f"[STT] {self.current_stt} loaded on {actual_device} in {load_time:.2f}s ({mode_str} - staying loaded)"
            )
            self.resource_manager.register_model(
                ModelType.STT,
                self.current_stt,
                actual_device,
                vram_gb=2.0 if "cuda" in actual_device else 0.0,
            )
        elif has_voice_server_url() and getenv("STT_ENABLED").lower() == "true":
            voice_url = getenv("VOICE_SERVER")
            logging.info(
                f"[STT] Voice server configured ({voice_url}) - skipping local model loading"
            )

        NGROK_TOKEN = getenv("NGROK_TOKEN")
        if NGROK_TOKEN:
            ngrok.set_auth_token(NGROK_TOKEN)
            public_url = ngrok.connect(8091)
            logging.info(f"[ngrok] Public Tunnel: {public_url.public_url}")
            self.local_uri = public_url.public_url
        else:
            self.local_uri = getenv("EZLOCALAI_URL")

        logging.info(f"[Server] Ready!")

    def _load_llm_resilient(
        self,
        model_name: str,
        max_tokens: int,
        gpu_layers: int = None,
        main_gpu: int = None,
        tensor_split: list = None,
    ) -> "LLM":
        """Load an LLM with smart GPU selection and resilient fallback.

        This method uses smart GPU selection to determine the optimal loading strategy:
        1. If GPU 0 has enough VRAM → load on GPU 0 only
        2. If GPU 0 + GPU 1 together have enough → tensor split across both
        3. If GPU 0 is full but GPU 1 has enough → load on GPU 1 only
        4. Otherwise → fall back to CPU

        Args:
            model_name: Name of the model to load
            max_tokens: Context size for the model
            gpu_layers: Number of GPU layers (None for smart auto-detect)
            main_gpu: Primary GPU index (None for smart auto-detect)
            tensor_split: Tensor split ratios (None for smart auto-detect)

        Returns:
            LLM instance

        Raises:
            Exception: Only if all loading strategies fail
        """
        # Get model path for VRAM estimation
        from ezlocalai.LLM import download_model

        try:
            model_path, mmproj_path = download_model(
                model_name=model_name, models_dir="./models"
            )
            projectors = [mmproj_path] if mmproj_path else []
        except Exception as e:
            logging.warning(f"[LLM] Could not get model path for estimation: {e}")
            model_path = None
            projectors = []

        # Use smart GPU selection if no explicit configuration provided
        if (
            gpu_layers is None
            and main_gpu is None
            and tensor_split is None
            and model_path
        ):
            # Only reserve VRAM for TTS/STT if running locally (no voice server URL)
            # When a voice server URL is configured, voice models are NOT loaded locally
            reserved_vram = 0.0 if has_voice_server_url() else 5.0
            strategy = determine_gpu_strategy(
                model_path=model_path,
                context_size=max_tokens,
                projectors=projectors,
                reserved_vram=reserved_vram,
            )

            gpu_layers = strategy["gpu_layers"]
            main_gpu = strategy["main_gpu"]
            tensor_split = strategy["tensor_split"]

            logging.info(
                f"[LLM] Smart GPU selection: strategy='{strategy['strategy']}', "
                f"main_gpu={main_gpu}, gpu_layers={gpu_layers}, "
                f"tensor_split={tensor_split[:get_gpu_count()] if tensor_split else 'none'}"
            )

        # First attempt: Try with determined/specified configuration
        try:
            logging.debug(
                f"[LLM] Attempting to load {model_name} (main_gpu={main_gpu}, "
                f"gpu_layers={gpu_layers or 'auto'}, tensor_split={'yes' if tensor_split else 'no'})..."
            )
            llm = LLM(
                model=model_name,
                max_tokens=max_tokens,
                gpu_layers=gpu_layers,
                main_gpu=main_gpu,
                tensor_split=tensor_split,
            )
            return llm
        except Exception as gpu_error:
            error_str = str(gpu_error).lower()
            # Check if this is a resource exhaustion error
            is_resource_error = any(
                x in error_str
                for x in [
                    "out of memory",
                    "cuda",
                    "vram",
                    "gpu",
                    "allocat",
                    "memory",
                    "resource",
                ]
            )

            if is_resource_error and gpu_layers != 0:
                logging.warning(
                    f"[LLM] GPU loading failed for {model_name}: {gpu_error}"
                )
                # Progressive fallback: try reducing GPU layers before going full CPU
                # This keeps most layers on GPU for speed while offloading some to
                # CPU/RAM to free VRAM for the larger KV cache at higher context sizes.
                total_layers = 41  # Default for most models
                try:
                    # Try to get actual layer count from model metadata
                    import gguf

                    if model_path:
                        reader = gguf.GGUFReader(model_path)
                        for kv in reader.fields.values():
                            if kv.name and "block_count" in kv.name:
                                total_layers = (
                                    int(kv.parts[-1][0]) + 1
                                )  # +1 for output layer
                                break
                except Exception:
                    pass

                # Try reducing layers: 75% -> 50% -> 25% -> CPU
                for fraction in [0.75, 0.5, 0.25, 0.0]:
                    try_layers = max(0, int(total_layers * fraction))
                    label = (
                        f"{try_layers}/{total_layers} layers on GPU"
                        if try_layers > 0
                        else "CPU-only"
                    )
                    logging.info(
                        f"[LLM] Trying {label} for {max_tokens//1024}k context..."
                    )
                    try:
                        llm = LLM(
                            model=model_name,
                            max_tokens=max_tokens,
                            gpu_layers=try_layers,
                            main_gpu=main_gpu,
                        )
                        logging.info(f"[LLM] {model_name} loaded with {label}")
                        return llm
                    except Exception as partial_error:
                        partial_str = str(partial_error).lower()
                        is_mem_error = any(
                            x in partial_str
                            for x in ["out of memory", "cuda", "alloc", "memory"]
                        )
                        if is_mem_error and try_layers > 0:
                            logging.warning(f"[LLM] {label} failed: {partial_error}")
                            continue
                        elif try_layers == 0:
                            logging.error(
                                f"[LLM] CPU fallback also failed for {model_name}: {partial_error}"
                            )
                            raise partial_error
                        else:
                            raise partial_error
            else:
                # Not a resource error, or already at gpu_layers=0
                raise gpu_error

    def _calibrate_model(self, model_name: str, max_tokens: int) -> int:
        """Calibrate a single model to find optimal GPU layers.

        First tries xllamacpp's native estimate_gpu_layers for a fast estimation,
        then falls back to binary search if needed.
        """
        # Try native estimation first if available
        if xllamacpp_available:
            try:
                estimated = self._estimate_layers_native(model_name, max_tokens)
                if estimated is not None:
                    return estimated
            except Exception as e:
                logging.warning(
                    f"[Calibration] Native estimation failed: {e}, falling back to binary search"
                )

        # Fallback: Binary search calibration
        return self._calibrate_binary_search(model_name, max_tokens)

    def _estimate_layers_native(self, model_name: str, max_tokens: int) -> int:
        """Use xllamacpp's native GPU layer estimation.

        Returns optimal GPU layers or None if estimation fails.
        """
        logging.debug(
            f"[Calibration] Using native estimation for {model_name} (budget: {self.vram_budget_gb}GB)"
        )

        # Get GPU info
        devices = get_device_info()
        gpus = []
        gpu_idx = 0
        for dev in devices:
            # Check if it's a GPU device
            dev_type = str(dev.get("type", ""))
            if "GPU" in dev_type:
                # Use per-GPU VRAM budget, not total budget
                # self.per_gpu_vram has the individual VRAM for each GPU
                if gpu_idx < len(self.per_gpu_vram):
                    per_gpu_budget = self.per_gpu_vram[gpu_idx]
                else:
                    per_gpu_budget = self.vram_budget_gb / max(
                        1, len(self.per_gpu_vram)
                    )
                budget_bytes = per_gpu_budget * 1024 * 1024 * 1024
                gpus.append(
                    {
                        "name": dev["name"],
                        "memory_free": budget_bytes,  # Use per-GPU budget
                        "memory_total": dev["memory_total"],
                    }
                )
                gpu_idx += 1

        if not gpus:
            logging.warning("[Calibration] No GPU devices found for estimation")
            return None

        logging.debug(
            f"[Calibration] GPUs: {[g['name'] + ' (' + str(round(g['memory_free']/1e9, 1)) + 'GB budget)' for g in gpus]}"
        )

        # Download model file to get path
        try:
            model_path = self._get_model_path(model_name)
            if not model_path:
                return None

            logging.debug(f"[Calibration] Model path: {model_path}")

            # Get projector if this might be a vision model
            projectors = []
            vision_proj = self._get_vision_projector_path(model_name)
            if vision_proj:
                projectors.append(vision_proj)

            # Estimate GPU layers
            result = estimate_gpu_layers(
                gpus=gpus,
                model_path=model_path,
                projectors=projectors,
                context_length=max_tokens,
                batch_size=2048,
                num_parallel=1,
                kv_cache_type="q4_0",
            )

            logging.debug(f"[Calibration] Native estimation result: {result}")

            # Extract layer count from result
            # xllamacpp returns MemoryEstimate object with .layers attribute
            if hasattr(result, "layers"):
                gpu_layers = result.layers
            elif isinstance(result, dict):
                gpu_layers = result.get(
                    "layers", result.get("gpu_layers", result.get("n_gpu_layers", 0))
                )
            elif isinstance(result, int):
                gpu_layers = result
            else:
                logging.warning(
                    f"[Calibration] Unexpected result format: {type(result)}"
                )
                return None

            logging.debug(
                f"[Calibration] {model_name} estimated to {gpu_layers} GPU layers (native)"
            )
            return gpu_layers

        except Exception as e:
            logging.error(f"[Calibration] Native estimation error: {e}")
            return None

    def _get_model_path(self, model_name: str) -> str:
        """Get the local path to a model file."""
        try:
            # Parse model name: "org/repo" -> download the GGUF file
            if "/" in model_name:
                # Try common GGUF patterns
                parts = model_name.split("/")
                repo_id = model_name

                # Try to find the GGUF file
                from huggingface_hub import list_repo_files

                files = list_repo_files(repo_id)
                gguf_files = [f for f in files if f.endswith(".gguf")]

                if not gguf_files:
                    return None

                # Prefer QUANT_TYPE from env, then fall back to common quantizations
                quant_type = getenv("QUANT_TYPE")
                best_file = None
                patterns = [
                    quant_type,
                    "Q4_K_M",
                    "Q4_K_XL",
                    "Q4_K",
                    "Q5_K",
                    "Q6_K",
                    "Q8",
                ]
                for pattern in patterns:
                    for f in gguf_files:
                        if pattern in f:
                            best_file = f
                            break
                    if best_file:
                        break

                if not best_file:
                    best_file = gguf_files[0]

                return hf_hub_download(repo_id, best_file)

            return None
        except Exception as e:
            logging.error(f"[Calibration] Failed to get model path: {e}")
            return None

    def _get_vision_projector_path(self, model_name: str) -> str:
        """Get vision projector path if this is a vision model."""
        try:
            if "/" in model_name:
                from huggingface_hub import list_repo_files

                files = list_repo_files(model_name)

                # Look for mmproj files
                for f in files:
                    if "mmproj" in f.lower() and f.endswith(".gguf"):
                        return hf_hub_download(model_name, f)
            return None
        except:
            return None

    def _is_vision_model(self, model_name: str) -> bool:
        """Check if a model has vision capability (has mmproj file)."""
        # Cache results to avoid repeated HuggingFace API calls
        if not hasattr(self, "_vision_model_cache"):
            self._vision_model_cache = {}

        if model_name in self._vision_model_cache:
            return self._vision_model_cache[model_name]

        has_vision = self._get_vision_projector_path(model_name) is not None
        self._vision_model_cache[model_name] = has_vision
        return has_vision

    def _find_vision_model(self) -> str:
        """Find a vision-capable model from available models."""
        for model_name in self.available_models:
            if self._is_vision_model(model_name):
                return model_name
        return None

    def _find_non_vision_model(self) -> str:
        """Find the first non-vision model from available models.

        Returns None if all models are vision models.
        """
        for model_name in self.available_models:
            if not self._is_vision_model(model_name):
                return model_name
        return None

    async def _describe_images_with_vision_model(
        self, images: list, user_text: str
    ) -> str:
        """Use a vision model to describe images, then return descriptions for use with non-vision model.

        This enables non-vision models to respond about images by using a vision model
        to first describe what's in the images.
        """
        vision_model = self._find_vision_model()
        if not vision_model:
            logging.warning(
                "[Vision Fallback] No vision model available to describe images"
            )
            return None

        try:
            # Load vision model (will destroy any current model first)
            self._get_llm(vision_model, 16384)  # Use 16k context for image description

            if not self.llm.is_vision:
                logging.error(
                    f"[Vision Fallback] {vision_model} failed to load with vision capability"
                )
                return None

            # Process images same way as in get_response
            from PIL import Image as PILImage
            from io import BytesIO

            processed_images = []
            for img in images:
                if "image_url" in img:
                    img_url = (
                        img["image_url"].get("url", "")
                        if isinstance(img["image_url"], dict)
                        else img["image_url"]
                    )
                    if img_url and not img_url.startswith("data:"):
                        try:
                            headers = {"User-Agent": "Mozilla/5.0"}
                            img_response = requests.get(
                                img_url, timeout=30, headers=headers
                            )
                            img_response.raise_for_status()
                            content_type = img_response.headers.get(
                                "Content-Type", "image/jpeg"
                            )

                            # Convert unsupported formats to PNG
                            if content_type in [
                                "image/webp",
                                "image/gif",
                                "image/bmp",
                                "image/tiff",
                                "image/avif",
                            ]:
                                pil_img = PILImage.open(BytesIO(img_response.content))
                                if pil_img.mode in ("RGBA", "P", "LA"):
                                    pil_img = pil_img.convert("RGB")
                                buffer = BytesIO()
                                pil_img.save(buffer, format="PNG")
                                img_bytes = buffer.getvalue()
                                content_type = "image/png"
                            else:
                                img_bytes = img_response.content

                            if not content_type.startswith("image/"):
                                content_type = "image/jpeg"
                            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                            processed_images.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{content_type};base64,{img_base64}"
                                    },
                                }
                            )
                        except Exception as e:
                            logging.error(
                                f"[Vision Fallback] Failed to fetch image: {e}"
                            )
                            continue
                    else:
                        # Already a data URL - check if it needs conversion
                        if img_url.startswith("data:image/webp") or img_url.startswith(
                            "data:image/gif"
                        ):
                            try:
                                # Extract base64 data and convert
                                header, encoded = img_url.split(",", 1)
                                img_bytes = base64.b64decode(encoded)
                                pil_img = PILImage.open(BytesIO(img_bytes))
                                if pil_img.mode in ("RGBA", "P", "LA"):
                                    pil_img = pil_img.convert("RGB")
                                buffer = BytesIO()
                                pil_img.save(buffer, format="PNG")
                                img_base64 = base64.b64encode(buffer.getvalue()).decode(
                                    "utf-8"
                                )
                                processed_images.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{img_base64}"
                                        },
                                    }
                                )
                                logging.debug(
                                    f"[Vision Fallback] Converted data URL WebP/GIF to PNG"
                                )
                            except Exception as conv_err:
                                logging.error(
                                    f"[Vision Fallback] Failed to convert data URL: {conv_err}"
                                )
                                processed_images.append(img)
                        else:
                            processed_images.append(img)

            if not processed_images:
                return None

            # Build multimodal message asking to describe the images
            describe_prompt = f"Describe the contents of the image(s) in detail. The user's question about the image(s) is: {user_text}"
            multimodal_content = [{"type": "text", "text": describe_prompt}]
            multimodal_content.extend(processed_images)

            # Get description from vision model
            response = self.llm.chat(
                messages=[{"role": "user", "content": multimodal_content}],
                local_uri=self.local_uri,
                max_tokens=2048,
                temperature=0.3,
            )

            # Extract the text response
            if hasattr(response, "choices") and response.choices:
                description = response.choices[0].message.content
            elif isinstance(response, dict) and "choices" in response:
                description = response["choices"][0]["message"]["content"]
            else:
                description = str(response)

            return description

        except Exception as e:
            logging.error(f"[Vision Fallback] Failed to describe images: {e}")
            return None
        # Note: LLM will be destroyed at the end of get_response() - no need to swap back

    def _calibrate_binary_search(self, model_name: str, max_tokens: int) -> int:
        """Calibrate using binary search (fallback method).

        Uses binary search to efficiently find the highest GPU layer count that
        fits within VRAM budget. Returns the optimal number of GPU layers.
        """
        # Binary search for optimal layers - start at 70 as most models have 40-80 layers
        low = 0
        high = 70
        best_layers = 0  # Default to CPU if nothing works

        logging.debug(
            f"[Calibration] Binary search for {model_name} (budget: {self.vram_budget_gb}GB)"
        )

        while low <= high:
            mid = (low + high) // 2

            try:
                # Clear VRAM before test
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                logging.debug(
                    f"[Calibration] Testing {model_name} with {mid} GPU layers"
                )

                # Try to load the model
                test_llm = LLM(model=model_name, max_tokens=max_tokens, gpu_layers=mid)

                # Check VRAM usage
                vram_used = get_vram_usage_gb()

                # Unload test model
                del test_llm
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if vram_used <= self.vram_budget_gb:
                    # Fits! Try higher
                    best_layers = mid
                    logging.debug(
                        f"[Calibration] {mid} layers OK ({vram_used:.1f}GB), trying higher"
                    )
                    low = mid + 1
                else:
                    # Too much VRAM, try lower
                    logging.debug(
                        f"[Calibration] {mid} layers too high ({vram_used:.1f}GB), trying lower"
                    )
                    high = mid - 1

            except Exception as e:
                error_msg = str(e).lower()
                if (
                    "out of memory" in error_msg
                    or "oom" in error_msg
                    or "cuda" in error_msg
                ):
                    logging.warning(
                        f"[Calibration] OOM at {mid} layers, trying lower..."
                    )
                else:
                    logging.error(f"[Calibration] Error at {mid} layers: {e}")

                # Cleanup after error
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # OOM means too many layers
                high = mid - 1

        logging.debug(
            f"[Calibration] {model_name} calibrated to {best_layers} GPU layers"
        )
        return best_layers

    def _get_gpu_layers_for_model(self, model_name: str, context_size: int) -> int:
        """Get GPU layers for a model at a specific context size.

        If not pre-calibrated for this context, calibrate now and cache.
        """
        if self.vram_budget_gb <= 0:
            return None  # No GPU available

        # Check cache for this model+context combo
        if model_name in self.calibrated_gpu_layers:
            if context_size in self.calibrated_gpu_layers[model_name]:
                return self.calibrated_gpu_layers[model_name][context_size]

        # Need to calibrate for this context size
        logging.debug(
            f"[Calibration] Calibrating {model_name} for {context_size//1000}k context"
        )
        calibrated = self._calibrate_model(model_name, context_size)

        # Cache it
        if model_name not in self.calibrated_gpu_layers:
            self.calibrated_gpu_layers[model_name] = {}
        self.calibrated_gpu_layers[model_name][context_size] = calibrated

        return calibrated

    def get_models(self):
        """Return list of available models without loading the LLM.

        This allows /v1/models endpoint to work even when LLM is not loaded.
        """
        from ezlocalai.LLM import get_models

        return get_models()

    def _ensure_context_size(self, required_context: int):
        """Reload LLM with larger context if needed using smart GPU selection.

        Thread-safe: Uses _model_lock to prevent race conditions.

        IMPORTANT: When increasing context size, we must unload ALL models (including
        vision model) to free GPU VRAM before reloading. Otherwise the GPU will be
        full and the model will fall back to CPU, which is very slow.

        If the requested context exceeds what fits fully on GPU, layers will be
        partially offloaded to CPU/RAM rather than capping the context.
        """
        with self._model_lock:
            if self.current_context and self.current_context >= required_context:
                # Current context is sufficient
                return

            # Need to reload with larger context
            logging.info(
                f"[LLM] Context {self.current_context//1000 if self.current_context else 0}k insufficient for {required_context:,} tokens, reloading at {required_context//1000}k"
            )

            model_name = self.current_llm_name

            # Store which models were loaded so we know what to potentially reload later
            had_vision_model = self.vision_llm is not None
            vision_model_name = self.vision_llm_name

            # Unload ALL models to free GPU VRAM for the larger context
            logging.info(
                "[LLM] Unloading all models to free GPU VRAM for larger context..."
            )

            # Unload vision model first
            if self.vision_llm:
                logging.debug(f"[LLM] Unloading vision model {self.vision_llm_name}")
                del self.vision_llm
                self.vision_llm = None
                self.vision_llm_name = None
                self.vision_llm_context = None

            # Unload primary model
            if self.primary_llm:
                logging.debug(f"[LLM] Unloading primary model {self.primary_llm_name}")
                del self.primary_llm
                self.primary_llm = None
                self.primary_llm_name = None
                self.primary_llm_context = None

            # Unload current active model (might be same as primary)
            if self.llm:
                del self.llm
                self.llm = None

            # Force garbage collection and clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for CUDA operations to complete
                # Log available VRAM after cleanup
                free_vram = get_free_vram_gb()
                logging.info(f"[LLM] After cleanup: {free_vram:.1f}GB VRAM free")

            # Load with new context - let smart GPU selection determine configuration
            start_time = time.time()
            self.llm = self._load_llm_resilient(
                model_name=model_name,
                max_tokens=required_context,
                # gpu_layers, main_gpu, tensor_split determined by smart selection
            )
            self.current_llm_name = model_name
            self.current_context = required_context

            # Also update primary model reference if this was the primary
            if (
                model_name == self.available_models[0]
                if self.available_models
                else None
            ):
                self.primary_llm = self.llm
                self.primary_llm_name = model_name
                self.primary_llm_context = required_context

            load_time = time.time() - start_time
            logging.info(
                f"[LLM] {model_name} reloaded at {required_context//1000}k context in {load_time:.2f}s"
            )

            # Note: We intentionally do NOT reload the vision model here.
            # It will be lazy-loaded on demand if needed, keeping VRAM available
            # for the large context model.

            # If we increased beyond optimal context, schedule a reset
            if required_context > self._optimal_context:
                self._schedule_context_reset()

    def _schedule_context_reset(self):
        """Schedule a reset back to optimal context after a cooldown period.

        This ensures that after handling a large context request, we eventually
        reload the model at the optimal context size to maximize GPU layers.
        The cooldown prevents constant reloading if multiple large requests come in.
        """
        with self._context_reset_lock:
            # Cancel any existing timer
            if self._context_reset_timer is not None:
                self._context_reset_timer.cancel()
                self._context_reset_timer = None

            logging.info(
                f"[LLM] Scheduling context reset to {self._optimal_context//1000}k in {self._context_reset_cooldown}s"
            )

            # Schedule new reset
            self._context_reset_timer = threading.Timer(
                self._context_reset_cooldown, self._perform_context_reset
            )
            self._context_reset_timer.daemon = True
            self._context_reset_timer.start()

    def _cancel_context_reset(self):
        """Cancel any pending context reset."""
        with self._context_reset_lock:
            if self._context_reset_timer is not None:
                self._context_reset_timer.cancel()
                self._context_reset_timer = None
                logging.debug("[LLM] Context reset cancelled")

    def _perform_context_reset(self):
        """Reset the model back to optimal context size.

        This runs in a background thread after the cooldown period.
        It reloads the model at the optimal context size to maximize GPU layers.
        """
        with self._context_reset_lock:
            self._context_reset_timer = None

        # Check if reset is still needed
        if (
            self.current_context is None
            or self.current_context <= self._optimal_context
        ):
            logging.debug("[LLM] Context reset not needed, already at optimal or lower")
            return

        # Check if inference is in progress - if so, reschedule
        if self._is_inference_in_progress():
            logging.debug("[LLM] Inference in progress, rescheduling context reset")
            self._schedule_context_reset()
            return

        logging.info(
            f"[LLM] Performing context reset: {self.current_context//1000}k -> {self._optimal_context//1000}k"
        )

        try:
            with self._model_lock:
                # Re-check inference status after acquiring lock (another request may have started)
                if self._is_inference_in_progress():
                    logging.debug(
                        "[LLM] Inference started while waiting for lock, aborting context reset"
                    )
                    self._schedule_context_reset()
                    return

                model_name = self.current_llm_name

                if model_name is None:
                    logging.debug("[LLM] No model loaded, skipping context reset")
                    return

                # Unload current model
                if self.llm:
                    logging.debug(f"[LLM] Unloading {model_name} for context reset")
                    del self.llm
                    self.llm = None

                if self.primary_llm:
                    del self.primary_llm
                    self.primary_llm = None

                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    free_vram = get_free_vram_gb()
                    logging.info(f"[LLM] After cleanup: {free_vram:.1f}GB VRAM free")

                # Reload at optimal context
                start_time = time.time()
                self.llm = self._load_llm_resilient(
                    model_name=model_name,
                    max_tokens=self._optimal_context,
                )
                self.current_llm_name = model_name
                self.current_context = self._optimal_context

                # Update primary model reference
                if self.available_models and model_name == self.available_models[0]:
                    self.primary_llm = self.llm
                    self.primary_llm_name = model_name
                    self.primary_llm_context = self._optimal_context

                load_time = time.time() - start_time
                logging.info(
                    f"[LLM] Context reset complete: {model_name} at {self._optimal_context//1000}k context in {load_time:.2f}s"
                )

        except Exception as e:
            logging.error(f"[LLM] Context reset failed: {e}")
            # Don't retry on failure - the model will be reloaded on next request

    def _swap_llm(self, requested_model: str, required_context: int = None):
        """Hot-swap to a different LLM if needed with smart GPU selection.

        Thread-safe: Uses _model_lock to prevent race conditions.

        Args:
            requested_model: Model name to swap to
            required_context: Minimum context size needed
        """
        with self._model_lock:
            # Check if this is a known model
            target_model = None

            for model_name in self.available_models:
                # Match by exact name or by the short name (last part after /)
                short_name = model_name.split("/")[-1].lower()
                requested_short = requested_model.split("/")[-1].lower()
                if (
                    model_name.lower() == requested_model.lower()
                    or short_name == requested_short
                ):
                    target_model = model_name
                    break

            if target_model is None:
                # Model not in available list, use current
                return

            # Determine context size
            target_context = required_context if required_context else 16384

            # Check if we already have this model loaded with sufficient context
            if (
                self.current_llm_name == target_model
                and self.current_context
                and self.current_context >= target_context
            ):
                return

            # Swap models - must unload old model first to free VRAM
            logging.debug(
                f"[LLM] Swapping to {target_model} at {target_context//1000}k context"
            )
            start_time = time.time()

            # Store old model info in case we need to rollback
            old_model_name = self.current_llm_name
            old_context = self.current_context or 16384

            # Destroy current LLM first to free VRAM
            if self.llm:
                logging.debug(f"[LLM] Unloading {old_model_name} to free VRAM")
                del self.llm
                self.llm = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Try to load new LLM with smart GPU selection
            try:
                self.llm = self._load_llm_resilient(
                    model_name=target_model,
                    max_tokens=target_context,
                    # Let smart GPU selection handle gpu_layers, main_gpu, tensor_split
                )
                self.current_llm_name = target_model
                self.current_context = target_context
                load_time = time.time() - start_time
                logging.debug(f"[LLM] {target_model} loaded in {load_time:.2f}s")
                if self.llm.is_vision:
                    logging.debug(f"[LLM] Vision capability enabled for {target_model}")
            except Exception as e:
                logging.error(f"[LLM] Failed to load {target_model}: {e}")
                # Rollback to old model with smart GPU selection
                logging.debug(f"[LLM] Rolling back to {old_model_name}")
                try:
                    self.llm = self._load_llm_resilient(
                        model_name=old_model_name,
                        max_tokens=old_context,
                        # Let smart GPU selection handle configuration
                    )
                    self.current_llm_name = old_model_name
                    self.current_context = old_context
                    logging.debug(f"[LLM] Rolled back to {old_model_name}")
                except Exception as rollback_error:
                    logging.error(
                        f"[LLM] CRITICAL: Failed to rollback to {old_model_name}: {rollback_error}"
                    )
                    # Last resort - try to load first available model at 16k with CPU fallback
                    for model_name in self.available_models:
                        try:
                            self.llm = self._load_llm_resilient(
                                model_name=model_name,
                                max_tokens=16384,
                                gpu_layers=0,  # Force CPU to maximize chance of success
                            )
                            self.current_llm_name = model_name
                            self.current_context = 16384
                            logging.debug(
                                f"[LLM] Recovered with {model_name} (CPU mode)"
                            )
                            break
                        except:
                            continue

    def _get_embedder(self):
        """Lazy load embedding model on demand."""
        if self.embedder is None:
            from ezlocalai.Embedding import Embedding

            logging.debug("[Embedding] Loading BGE-M3 on demand")
            start_time = time.time()
            self.embedder = Embedding()
            logging.debug(
                f"[Embedding] BGE-M3 loaded in {time.time() - start_time:.2f}s"
            )
        return self.embedder

    def _destroy_embedder_sync(self, embedder_ref):
        """Synchronous embedder destruction."""
        try:
            start_time = time.time()
            del embedder_ref
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cleanup_time = time.time() - start_time
            logging.debug(f"[Embedding] BGE-M3 unloaded in {cleanup_time:.2f}s")
        except Exception as e:
            logging.error(f"[Embedding] Error during cleanup: {e}")

    def _destroy_embedder(self, async_cleanup: bool = True):
        """Destroy embedding model to free resources."""
        if self.embedder is not None:
            embedder_ref = self.embedder
            self.embedder = None

            if async_cleanup:
                logging.debug("[Embedding] Scheduling async unload...")
                _schedule_cleanup(self._destroy_embedder_sync, embedder_ref)
            else:
                self._destroy_embedder_sync(embedder_ref)

    def _get_stt(self, force_cpu: bool = False):
        """Lazy load STT model on demand with smart resource management.

        Args:
            force_cpu: If True, force CPU mode even if GPU is available
        """
        resource_mgr = get_resource_manager()

        if self.stt is None:
            from ezlocalai.STT import STT

            # Check resource availability
            can_load, device, reason = resource_mgr.can_load_model(ModelType.STT)

            if force_cpu or device == "cpu":
                # STT class handles device selection internally, but we can hint via VRAM check
                logging.debug(
                    f"[STT] Loading {self.current_stt} on CPU (reason: {reason})"
                )
            else:
                logging.debug(f"[STT] Loading {self.current_stt} ({reason})")

            start_time = time.time()
            self.stt = STT(model=self.current_stt)
            load_time = time.time() - start_time

            # Register with resource manager
            actual_device = getattr(self.stt, "device", "cpu")
            resource_mgr.register_model(
                ModelType.STT,
                self.current_stt,
                actual_device,
                vram_gb=2.0 if "cuda" in actual_device else 0.0,
            )

            logging.debug(
                f"[STT] {self.current_stt} loaded on {actual_device} in {load_time:.2f}s"
            )

        resource_mgr.mark_model_in_use(ModelType.STT, True)
        return self.stt

    def _destroy_stt_sync(self, stt_ref):
        """Synchronous STT destruction."""
        try:
            start_time = time.time()
            del stt_ref
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cleanup_time = time.time() - start_time
            logging.debug(f"[STT] Whisper unloaded in {cleanup_time:.2f}s")
        except Exception as e:
            logging.error(f"[STT] Error during cleanup: {e}")

    def _destroy_stt(self, async_cleanup: bool = True, force: bool = False):
        """Destroy STT model to free resources.

        Args:
            async_cleanup: If True, run cleanup in background thread
            force: If True, destroy even if other requests might need it soon
        """
        resource_mgr = get_resource_manager()
        resource_mgr.mark_model_in_use(ModelType.STT, False)

        # When preloading is enabled (voice server mode OR LAZY_LOAD_VOICE=false), never unload unless forced
        if should_preload_voice() and not force:
            logging.debug("[STT] Preload mode - keeping STT loaded")
            return

        # Always unload STT after use - loads quickly (~3s) and frees ~3GB VRAM
        if self.stt is not None:
            stt_ref = self.stt
            self.stt = None
            resource_mgr.unregister_model(ModelType.STT)

            if async_cleanup:
                logging.debug("[STT] Scheduling async unload...")
                _schedule_cleanup(self._destroy_stt_sync, stt_ref)
            else:
                self._destroy_stt_sync(stt_ref)

    def _get_img(self, force_cpu: bool = False):
        """Lazy load IMG model on demand with smart resource management.

        Args:
            force_cpu: If True, force CPU mode (slower but frees GPU for LLM)
        """
        global img_import_success
        resource_mgr = get_resource_manager()

        if self.img is None and img_import_success:
            IMG_MODEL = getenv("IMG_MODEL")
            if IMG_MODEL:
                # Determine target GPU: prefer secondary GPU so LLM stays on primary
                secondary = get_secondary_gpu()
                if secondary is not None:
                    # Multi-GPU: load image model on the less powerful GPU
                    target_device = f"cuda:{secondary}"
                    logging.info(
                        f"[IMG] Multi-GPU: routing image model to secondary GPU {secondary} "
                        f"(primary GPU reserved for LLM)"
                    )
                else:
                    target_device = None  # Single GPU - use resource manager

                # Check resource availability
                can_load, device, reason = resource_mgr.can_load_model(
                    ModelType.IMG, required_vram=16.0
                )

                if force_cpu:
                    img_device = "cpu"
                elif target_device is not None:
                    # Multi-GPU: always use the secondary GPU
                    img_device = target_device
                elif not can_load and device == "fallback":
                    logging.warning(f"[IMG] {reason} - image generation may be slow")
                    img_device = "cuda"  # Will use CPU offload
                elif device == "cpu":
                    img_device = "cpu"
                else:
                    img_device = "cuda"

                logging.debug(f"[IMG] Loading {IMG_MODEL} on {img_device} ({reason})")
                start_time = time.time()

                try:
                    self.img = IMG(
                        model=IMG_MODEL, local_uri=self.local_uri, device=img_device
                    )
                    load_time = time.time() - start_time

                    # Register with resource manager (IMG uses CPU offload so may use less VRAM)
                    actual_vram = (
                        8.0 if img_device == "cuda" else 0.0
                    )  # Conservative estimate with offload
                    resource_mgr.register_model(
                        ModelType.IMG, IMG_MODEL, img_device, actual_vram
                    )

                    logging.debug(
                        f"[IMG] {IMG_MODEL} loaded on {img_device} in {load_time:.2f}s"
                    )
                except Exception as e:
                    logging.error(f"[IMG] Failed to load the model: {e}")
                    self.img = None

        if self.img:
            resource_mgr.mark_model_in_use(ModelType.IMG, True)
        return self.img

    def _destroy_img_sync(self, img_ref):
        """Synchronous IMG destruction."""
        try:
            start_time = time.time()
            del img_ref
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cleanup_time = time.time() - start_time
            logging.debug(f"[IMG] Image model unloaded in {cleanup_time:.2f}s")
        except Exception as e:
            logging.error(f"[IMG] Error during cleanup: {e}")

    def _destroy_img(self, async_cleanup: bool = True, force: bool = False):
        """Destroy IMG model to free resources.

        Args:
            async_cleanup: If True, run cleanup in background thread
            force: If True, destroy even if other requests might need it soon
        """
        resource_mgr = get_resource_manager()
        resource_mgr.mark_model_in_use(ModelType.IMG, False)

        # Always unload IMG after use - uses ~16GB VRAM and loads in ~2-3s
        if self.img is not None:
            img_ref = self.img
            self.img = None
            resource_mgr.unregister_model(ModelType.IMG)

            if async_cleanup:
                logging.debug("[IMG] Scheduling async unload...")
                _schedule_cleanup(self._destroy_img_sync, img_ref)
            else:
                self._destroy_img_sync(img_ref)

    def _free_vram_for_llm(self, required_vram: float):
        """Free VRAM by unloading idle auxiliary models.

        Called before loading an LLM when VRAM is tight.
        Unloads TTS, STT, IMG, Embedding models if they're not in use.

        Args:
            required_vram: Amount of VRAM we need to free in GB
        """
        resource_mgr = get_resource_manager()

        # Get models to unload (never unloads LLM)
        to_unload = resource_mgr.get_models_to_unload(required_vram)

        if not to_unload:
            logging.debug("[Resource] No idle models to unload")
            return

        logging.info(
            f"[Resource] Freeing VRAM by unloading: {[m.value for m in to_unload]}"
        )

        for model_type in to_unload:
            if model_type == ModelType.TTS:
                self._destroy_tts(async_cleanup=False, force=True)
            elif model_type == ModelType.STT:
                self._destroy_stt(async_cleanup=False, force=True)
            elif model_type == ModelType.IMG:
                self._destroy_img(async_cleanup=False, force=True)
            elif model_type == ModelType.EMBEDDING:
                self._destroy_embedder(async_cleanup=False)

        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        free_vram = resource_mgr.get_total_free_vram()
        logging.info(f"[Resource] After cleanup: {free_vram:.1f}GB VRAM free")

    def _get_llm(self, model_name: str = None, context_size: int = 16384):
        """Get LLM instance, using persistent models when possible.

        Persistent models (primary_llm, vision_llm) are kept loaded to avoid
        reload overhead for frequent requests. Large models are loaded on demand
        and unloaded after use.

        Uses smart GPU selection to determine optimal loading strategy:
        - GPU 0 only if it has enough VRAM
        - Tensor split if GPU 0 needs help from other GPUs
        - GPU 1 alone if GPU 0 is fully occupied
        - CPU fallback if no GPU has sufficient VRAM

        Thread-safe: Uses _model_lock to prevent race conditions when multiple
        requests try to access/load models simultaneously.
        """
        resource_mgr = get_resource_manager()

        if model_name is None:
            model_name = self.available_models[0] if self.available_models else None

        if model_name is None:
            logging.warning("[LLM] No model available to load")
            return None

        # Acquire lock to prevent race conditions during model check/load
        with self._model_lock:
            logging.debug(
                f"[LLM _get_llm] Requested model='{model_name}', context={context_size}"
            )
            logging.debug(
                f"[LLM _get_llm] Current: llm={self.llm is not None}, name='{self.current_llm_name}', ctx={self.current_context}"
            )
            logging.debug(
                f"[LLM _get_llm] Primary: llm={self.primary_llm is not None}, name='{self.primary_llm_name}', ctx={self.primary_llm_context}"
            )
            logging.debug(
                f"[LLM _get_llm] Vision: llm={self.vision_llm is not None}, name='{self.vision_llm_name}', ctx={self.vision_llm_context}"
            )

            # Check if we already have the right model loaded with sufficient context
            if (
                self.llm is not None
                and self.current_llm_name == model_name
                and self.current_context
                and self.current_context >= context_size
            ):
                logging.debug(f"[LLM _get_llm] Using current self.llm (already loaded)")
                return self.llm

            # Check if this is one of our persistent models
            is_primary = model_name == self.primary_llm_name
            is_vision = model_name == self.vision_llm_name
            logging.debug(
                f"[LLM _get_llm] is_primary={is_primary}, is_vision={is_vision}"
            )

            # If we were using a large model, unload it first
            if self._using_large_model and self.llm is not None:
                logging.debug(
                    f"[LLM] Unloading large model to switch back to persistent model"
                )
                self._destroy_llm_temp()
                self._using_large_model = False

            # Try to use persistent models
            if is_primary and self.primary_llm is not None:
                if (
                    self.primary_llm_context
                    and self.primary_llm_context >= context_size
                ):
                    logging.debug(f"[LLM] Using persistent primary model: {model_name}")
                    self.llm = self.primary_llm
                    self.current_llm_name = model_name
                    self.current_context = self.primary_llm_context
                    return self.llm
                else:
                    # Need to reload primary with larger context
                    logging.info(
                        f"[LLM] Reloading primary model with larger context: {context_size//1000}k (had {self.primary_llm_context})"
                    )
                    del self.primary_llm
                    self.primary_llm = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if is_vision and self.vision_llm is not None:
                if self.vision_llm_context and self.vision_llm_context >= context_size:
                    logging.debug(f"[LLM] Using persistent vision model: {model_name}")
                    self.llm = self.vision_llm
                    self.current_llm_name = model_name
                    self.current_context = self.vision_llm_context
                    return self.llm
                else:
                    # Need to reload vision with larger context
                    logging.info(
                        f"[LLM] Reloading vision model with larger context: {context_size//1000}k (had {self.vision_llm_context})"
                    )
                    del self.vision_llm
                    self.vision_llm = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Loading a new model - check if it's a "large" model (not primary or vision)
            is_large_model = not is_primary and not is_vision
            logging.debug(f"[LLM] Loading new model - is_large_model={is_large_model}")

            if is_large_model:
                # Unload persistent models temporarily to make room for large model
                logging.debug(
                    f"[LLM] Loading large model {model_name}, temporarily unloading persistent models"
                )
                if self.primary_llm is not None:
                    del self.primary_llm
                    self.primary_llm = None
                if self.vision_llm is not None:
                    del self.vision_llm
                    self.vision_llm = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._using_large_model = True

            # Check if we need to free VRAM before loading
            estimated_vram = MODEL_VRAM_ESTIMATES.get(ModelType.LLM, 8.0)
            free_vram = resource_mgr.get_total_free_vram()
            if free_vram < estimated_vram:
                logging.debug(
                    f"[LLM] Low VRAM ({free_vram:.1f}GB), freeing auxiliary models"
                )
                self._free_vram_for_llm(
                    estimated_vram - free_vram + 2.0
                )  # Extra 2GB buffer

            logging.debug(
                f"[LLM] Loading {model_name} (context: {context_size//1000}k)"
            )
            start_time = time.time()

            # Let _load_llm_resilient handle smart GPU selection
            new_llm = self._load_llm_resilient(
                model_name=model_name,
                max_tokens=context_size,
            )

            # Store in appropriate slot
            if is_primary:
                self.primary_llm = new_llm
                self.primary_llm_name = model_name
                self.primary_llm_context = context_size
                # Register primary LLM with resource manager
                resource_mgr.register_model(
                    ModelType.LLM, model_name, "cuda", vram_gb=estimated_vram
                )
            elif is_vision:
                self.vision_llm = new_llm
                self.vision_llm_name = model_name
                self.vision_llm_context = context_size
                # Register vision LLM with resource manager
                resource_mgr.register_model(
                    ModelType.VISION_LLM,
                    model_name,
                    "cuda",
                    vram_gb=MODEL_VRAM_ESTIMATES.get(ModelType.VISION_LLM, 6.0),
                )

            self.llm = new_llm
            self.current_llm_name = model_name
            self.current_context = context_size

            # Mark LLM as in use
            resource_mgr.mark_model_in_use(ModelType.LLM, True)

            load_time = time.time() - start_time
            logging.debug(f"[LLM] {model_name} loaded in {load_time:.2f}s")
            if self.llm.is_vision:
                logging.debug(f"[LLM] Vision capability enabled for {model_name}")

            return self.llm

    def _destroy_llm_temp(self):
        """Destroy temporary (large) LLM without affecting persistent models.

        Note: This method should only be called while holding _model_lock.
        """
        if self.llm is not None and self._using_large_model:
            logging.debug(
                f"[LLM] Unloading temporary large model: {self.current_llm_name}"
            )
            del self.llm
            self.llm = None
            self.current_llm_name = None
            self.current_context = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._using_large_model = False

    def _reload_persistent_models(self):
        """Reload persistent models after using a large model.

        Thread-safe: Uses _model_lock to prevent race conditions.
        """
        with self._model_lock:
            default_context = self._optimal_context

            # Reload primary if it was set
            if self.primary_llm_name and self.primary_llm is None:
                logging.debug(
                    f"[LLM] Reloading persistent primary model: {self.primary_llm_name}"
                )
                try:
                    self.primary_llm = self._load_llm_resilient(
                        model_name=self.primary_llm_name,
                        max_tokens=self.primary_llm_context or default_context,
                    )
                    self.llm = self.primary_llm
                    self.current_llm_name = self.primary_llm_name
                    self.current_context = self.primary_llm_context or default_context
                except Exception as e:
                    logging.warning(f"[LLM] Failed to reload primary model: {e}")

            # Reload vision if it was set and different from primary
            if (
                self.vision_llm_name
                and self.vision_llm is None
                and self.vision_llm_name != self.primary_llm_name
            ):
                logging.debug(
                    f"[LLM] Reloading persistent vision model: {self.vision_llm_name}"
                )
                try:
                    self.vision_llm = self._load_llm_resilient(
                        model_name=self.vision_llm_name,
                        max_tokens=self.vision_llm_context or default_context,
                    )
                except Exception as e:
                    logging.warning(f"[LLM] Failed to reload vision model: {e}")

    def _destroy_llm_sync(self, llm_ref, model_name: str):
        """Synchronous LLM destruction - runs the actual cleanup.

        This is called either directly or from a background thread.
        """
        try:
            start_time = time.time()
            del llm_ref
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            cleanup_time = time.time() - start_time
            logging.debug(
                f"[LLM] {model_name} unloaded, VRAM freed in {cleanup_time:.2f}s"
            )
        except Exception as e:
            logging.error(f"[LLM] Error during cleanup of {model_name}: {e}")

    def _destroy_llm(self, async_cleanup: bool = True):
        """Destroy LLM to free VRAM.

        This is called after each inference to ensure VRAM is freed.
        The model will be reloaded on the next request via _get_llm().

        Thread-safe: Uses _model_lock to prevent race conditions.

        Args:
            async_cleanup: If True (default), run cleanup in background thread
                          for faster response times. If False, run synchronously.
        """
        with self._model_lock:
            if self.llm is not None:
                model_name = self.current_llm_name or "LLM"
                llm_ref = self.llm

                # Clear references immediately so new requests can load fresh
                self.llm = None
                self.current_llm_name = None
                self.current_context = None

                if async_cleanup:
                    logging.debug(f"[LLM] Scheduling async unload of {model_name}")
                    _schedule_cleanup(self._destroy_llm_sync, llm_ref, model_name)
                else:
                    logging.debug(f"[LLM] Unloading {model_name} synchronously")
                    self._destroy_llm_sync(llm_ref, model_name)

    def _create_tts_model(self):
        """Create TTS model (Chatterbox TTS)."""
        return CTTS()

    def _get_tts_name(self):
        """Get the human-readable name for the TTS provider."""
        return "Chatterbox TTS"

    def _get_tts(self, force_cpu: bool = False):
        """Lazy load TTS model on demand with smart resource management.

        Args:
            force_cpu: If True, force CPU mode (not directly supported but affects loading)
        """
        resource_mgr = get_resource_manager()
        tts_name = self._get_tts_name()

        if self.ctts is None:
            # Check resource availability
            can_load, device, reason = resource_mgr.can_load_model(
                ModelType.TTS, required_vram=4.0
            )

            logging.debug(f"[TTS] Loading {tts_name} ({reason})")
            start_time = time.time()
            self.ctts = self._create_tts_model()
            load_time = time.time() - start_time

            # TTS uses CUDA if available (handled internally by model)
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            resource_mgr.register_model(
                ModelType.TTS,
                tts_name,
                actual_device,
                vram_gb=4.0 if actual_device == "cuda" else 0.0,
            )

            logging.debug(
                f"[TTS] {tts_name} loaded on {actual_device} in {load_time:.2f}s"
            )

        resource_mgr.mark_model_in_use(ModelType.TTS, True)
        return self.ctts

    def _destroy_tts_sync(self, tts_ref):
        """Synchronous TTS destruction."""
        try:
            start_time = time.time()
            del tts_ref
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cleanup_time = time.time() - start_time
            logging.debug(f"[TTS] TTS model unloaded in {cleanup_time:.2f}s")
        except Exception as e:
            logging.error(f"[TTS] Error during TTS cleanup: {e}")

    def _destroy_tts(self, async_cleanup: bool = True, force: bool = False):
        """Destroy TTS model to free resources.

        Args:
            async_cleanup: If True, run cleanup in background thread
            force: If True, destroy even if other requests might need it soon
        """
        resource_mgr = get_resource_manager()
        resource_mgr.mark_model_in_use(ModelType.TTS, False)

        # When preloading is enabled (voice server mode OR LAZY_LOAD_VOICE=false), never unload unless forced
        if should_preload_voice() and not force:
            logging.debug("[TTS] Preload mode - keeping TTS loaded")
            return

        # Always unload TTS after use - loads quickly (~1-2s) and frees ~4GB VRAM
        if self.ctts is not None:
            tts_ref = self.ctts
            self.ctts = None
            resource_mgr.unregister_model(ModelType.TTS)

            if async_cleanup:
                logging.debug("[TTS] Scheduling async unload...")
                _schedule_cleanup(self._destroy_tts_sync, tts_ref)
            else:
                self._destroy_tts_sync(tts_ref)

    async def fallback_inference(self, messages, data: dict = None):
        """Use fallback server for inference when local resources are exhausted.

        If FALLBACK_SERVER is an ezlocalai instance (detected by /v1/resources endpoint),
        uses full ezlocalai forwarding. Otherwise, uses OpenAI-compatible client with
        the originally requested model (pass-through).

        Args:
            messages: The messages list for chat completion
            data: Optional full request data dict (used for ezlocalai forwarding)
        """
        fallback_client = get_fallback_client()

        if not fallback_client.is_configured:
            logging.warning("[Fallback] No fallback server configured")
            return "Unable to process request. Local resources exhausted and no fallback server configured."

        # Get the requested model from data, or use FALLBACK_MODEL as override, or DEFAULT_MODEL as last resort
        requested_model = None
        if data and "model" in data:
            requested_model = data["model"]

        # FALLBACK_MODEL env var can be used to override the model for non-ezlocalai servers
        fallback_model_override = getenv("FALLBACK_MODEL")

        # Check if fallback is an ezlocalai instance (has resources endpoint)
        available, reason = await fallback_client.check_availability()

        if available:
            # It's an ezlocalai instance - use full forwarding with original request
            logging.info(
                f"[Fallback] Using ezlocalai server: {fallback_client.base_url}"
            )
            try:
                request_data = (
                    data
                    if data
                    else {
                        "messages": messages,
                        "model": requested_model or getenv("DEFAULT_MODEL"),
                    }
                )
                response = await fallback_client.forward_chat_completion(
                    request_data, stream=False
                )
                if isinstance(response, dict) and "choices" in response:
                    return response["choices"][0]["message"]["content"]
                return str(response)
            except Exception as e:
                logging.warning(
                    f"[Fallback] ezlocalai forwarding failed: {e}, trying OpenAI-compatible client..."
                )

        # Use OpenAI-compatible client (for non-ezlocalai servers or when ezlocalai forwarding fails)
        # Priority: FALLBACK_MODEL override > requested model > DEFAULT_MODEL
        model_to_use = (
            fallback_model_override or requested_model or getenv("DEFAULT_MODEL")
        )

        logging.info(
            f"[Fallback] Using OpenAI-compatible client: {fallback_client.base_url} with model {model_to_use}"
        )

        try:
            from openai import Client

            client = Client(
                api_key=fallback_client.api_key, base_url=fallback_client.base_url
            )
            response = client.chat.completions.create(
                model=model_to_use, messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"[Fallback] Fallback server request failed: {e}")
            return f"Unable to process request. Fallback server error: {str(e)}"

    def should_use_fallback(self) -> Tuple[bool, str]:
        """Check if we should use fallback server instead of local inference.

        Returns:
            Tuple of (should_use_fallback, reason)
        """
        # Check fallback based on resource thresholds
        should_fallback, reason = should_use_ezlocalai_fallback()
        if should_fallback:
            return True, reason

        resource_mgr = get_resource_manager()

        # Check if fallback is configured
        if not get_fallback_client().is_configured:
            return False, "No fallback server configured"

        # Check if LLM is available and has sufficient resources
        can_load, device, reason = resource_mgr.can_load_model(ModelType.LLM)

        if not can_load and device == "fallback":
            return True, reason

        return False, "Local resources available"

    async def forward_to_fallback(self, endpoint: str, data: dict = None, **kwargs):
        """Forward any request to the fallback server.

        This is the main entry point for fallback requests. It handles all endpoint types
        and falls back gracefully if the ezlocalai fallback is unavailable.

        Args:
            endpoint: The endpoint type ("chat", "completion", "embeddings", "transcription", "tts", "image")
            data: The request data dict
            **kwargs: Additional arguments for specific endpoint types

        Returns:
            The response from the fallback server, or None if fallback is unavailable
        """
        fallback_client = get_fallback_client()

        if not fallback_client.is_configured:
            return None

        available, reason = await fallback_client.check_availability()
        if not available:
            logging.debug(f"[Fallback] ezlocalai fallback not available: {reason}")
            return None

        logging.info(f"[Fallback] Forwarding {endpoint} request to fallback server")

        try:
            if endpoint == "chat":
                return await fallback_client.forward_chat_completion(
                    data, stream=kwargs.get("stream", False)
                )
            elif endpoint == "completion":
                return await fallback_client.forward_completion(
                    data, stream=kwargs.get("stream", False)
                )
            elif endpoint == "embeddings":
                return await fallback_client.forward_embeddings(data)
            elif endpoint == "transcription":
                return await fallback_client.forward_transcription(**kwargs)
            elif endpoint == "tts":
                return await fallback_client.forward_tts(**kwargs)
            elif endpoint == "image":
                return await fallback_client.forward_image_generation(**kwargs)
            else:
                logging.warning(f"[Fallback] Unknown endpoint type: {endpoint}")
                return None
        except Exception as e:
            logging.error(f"[Fallback] Forward to fallback failed for {endpoint}: {e}")
            return None

    async def pdf_to_audio(self, title, voice, pdf, chunk_size=200):
        # Sanitize title to prevent path traversal
        import re

        # First sanitize the input to only allow safe characters
        if not title or not isinstance(title, str):
            title = "output"
        # Remove any path separators and dangerous characters
        title = re.sub(r'[/\\:*?"<>|]', "", title)
        # Remove path traversal attempts
        title = title.replace("..", "")
        # Only allow alphanumeric, hyphen, underscore, space - strict allowlist
        title = re.sub(r"[^a-zA-Z0-9_\- ]", "", title)
        if not title:
            title = "output"
        title = title[:100]

        # Use basename to strip any remaining path components
        safe_title = os.path.basename(title)
        if not safe_title:
            safe_title = "output"

        outputs_dir = os.path.realpath(os.path.join(os.getcwd(), "outputs"))
        os.makedirs(outputs_dir, exist_ok=True)

        # Construct and normalize the path - CodeQL pattern from documentation
        fullpath = os.path.normpath(os.path.join(outputs_dir, f"{safe_title}.pdf"))
        # Verify with normalized version of path - exact CodeQL recommended pattern
        if not fullpath.startswith(outputs_dir):
            raise ValueError("Invalid path - potential path traversal")

        pdf_data = pdf.split(",")[1]
        pdf_bytes = base64.b64decode(pdf_data)
        # fullpath is verified safe - use it directly
        with open(fullpath, "wb") as pdf_file:
            pdf_file.write(pdf_bytes)

        content = ""
        if fullpath.endswith(".pdf"):
            # fullpath was already verified above, use it directly
            with pdfplumber.open(fullpath) as pdf_doc:
                content = "\n".join([page.extract_text() for page in pdf_doc.pages])
        if not content:
            return
        tts = self._get_tts()
        self.resource_manager.mark_model_in_use(ModelType.TTS, True)
        try:
            result = await tts.generate(
                text=content,
                voice=voice,
                local_uri=self.local_uri,
                output_file_name=f"{safe_title}.wav",
            )
        finally:
            self.resource_manager.mark_model_in_use(ModelType.TTS, False)
        # In voice server mode, don't destroy TTS - keep it loaded
        if not is_voice_server_mode():
            self._destroy_tts()
        return result

    async def audio_to_audio(self, voice, audio):
        audio_type = audio.split(",")[0].split(":")[1].split(";")[0]
        audio_format = audio_type.split("/")[1]
        audio = audio.split(",")[1]
        audio = base64.b64decode(audio)
        stt = self._get_stt()
        self.resource_manager.mark_model_in_use(ModelType.STT, True)
        try:
            text = stt.transcribe_audio(base64_audio=audio, audio_format=audio_format)
        finally:
            self.resource_manager.mark_model_in_use(ModelType.STT, False)
        # In voice server mode, don't destroy STT - keep it loaded
        if not is_voice_server_mode():
            self._destroy_stt()
        tts = self._get_tts()
        self.resource_manager.mark_model_in_use(ModelType.TTS, True)
        try:
            result = await tts.generate(
                text=text, voice=voice, local_uri=self.local_uri
            )
        finally:
            self.resource_manager.mark_model_in_use(ModelType.TTS, False)
        # In voice server mode, don't destroy TTS - keep it loaded
        if not is_voice_server_mode():
            self._destroy_tts()
        return result

    async def generate_image(self, prompt, response_format="url", size="512x512"):
        img = self._get_img()
        if img:
            self.resource_manager.mark_model_in_use(ModelType.IMG, True)
            try:
                img.local_uri = self.local_uri if response_format == "url" else None
                new_image = img.generate(
                    prompt=prompt,
                    size=size,
                )
            finally:
                self.resource_manager.mark_model_in_use(ModelType.IMG, False)
            self._destroy_img()
            return new_image
        return ""

    def _apply_model_config_overrides(self, data: dict) -> dict:
        """Apply model-specific config overrides if the current model has them defined.

        Overrides only apply to parameters defined in MODEL_CONFIG_OVERRIDES for the
        current model. For dict values (e.g. chat_template_kwargs), model defaults
        are merged with user-provided values, with user values taking priority.
        """
        if self.current_llm_name and self.current_llm_name in MODEL_CONFIG_OVERRIDES:
            overrides = MODEL_CONFIG_OVERRIDES[self.current_llm_name]
            for key, value in overrides.items():
                if isinstance(value, dict):
                    # Deep-merge: model defaults first, then user overrides on top
                    merged = dict(value)
                    user_val = data.get(key, {})
                    if isinstance(user_val, dict):
                        merged.update(user_val)
                    data[key] = merged
                else:
                    data[key] = value
            logging.debug(
                f"[Config] Applied model overrides for {self.current_llm_name}: {overrides}"
            )
        return data

    def _is_inference_in_progress(self) -> bool:
        """Thread-safe check if any inference is currently in progress."""
        with self._inference_count_lock:
            return self._inference_count > 0

    def _increment_inference_count(self):
        """Thread-safe increment of inference counter."""
        with self._inference_count_lock:
            self._inference_count += 1
            logging.debug(
                f"[Inference] Started - active count: {self._inference_count}"
            )

    def _decrement_inference_count(self):
        """Thread-safe decrement of inference counter."""
        with self._inference_count_lock:
            self._inference_count = max(0, self._inference_count - 1)
            logging.debug(
                f"[Inference] Completed - active count: {self._inference_count}"
            )

    async def get_response(self, data, completion_type="chat"):
        # Check if we should use fallback BEFORE allocating local resources
        should_fallback, fallback_reason = self.should_use_fallback()
        if should_fallback:
            logging.info(
                f"[Fallback] Pre-check: {fallback_reason}, attempting fallback..."
            )
            fallback_client = get_fallback_client()
            if fallback_client.is_configured:
                available, avail_reason = await fallback_client.check_availability()
                if available:
                    logging.info(
                        f"[Fallback] Using ezlocalai fallback for {completion_type}"
                    )
                    try:
                        endpoint = "chat" if completion_type == "chat" else "completion"
                        is_streaming = data.get("stream", False)
                        response = (
                            await fallback_client.forward_chat_completion(
                                data, stream=is_streaming
                            )
                            if endpoint == "chat"
                            else await fallback_client.forward_completion(
                                data, stream=is_streaming
                            )
                        )

                        # For streaming, wrap the async generator appropriately
                        if is_streaming:
                            # Convert async generator to sync generator for compatibility
                            async def async_to_sync_wrapper():
                                async for chunk in response:
                                    yield chunk

                            return async_to_sync_wrapper(), None
                        else:
                            return response, None
                    except Exception as e:
                        logging.warning(
                            f"[Fallback] ezlocalai fallback failed: {e}, falling back to local processing"
                        )
                else:
                    logging.debug(
                        f"[Fallback] ezlocalai fallback not available: {avail_reason}"
                    )

        # Cancel any pending context reset since we're handling a request
        self._cancel_context_reset()

        # Mark inference as in progress to prevent context reset during processing
        self._increment_inference_count()

        # Mark LLM as in-use in resource manager
        llm_model_type = (
            ModelType.VISION_LLM if (self.llm and self.llm.is_vision) else ModelType.LLM
        )
        self.resource_manager.mark_model_in_use(llm_model_type, True)

        try:
            return await self._get_response_internal(data, completion_type)
        finally:
            # Mark LLM as no longer in-use
            self.resource_manager.mark_model_in_use(llm_model_type, False)

            # Mark inference as complete
            self._decrement_inference_count()

            # If we're at a larger-than-optimal context and no other inferences running, schedule a reset
            if self.current_context and self.current_context > self._optimal_context:
                if not self._is_inference_in_progress():
                    self._schedule_context_reset()

    async def _get_response_internal(self, data, completion_type="chat"):
        """Internal implementation of get_response."""
        data["local_uri"] = self.local_uri
        # Apply model-specific config overrides
        data = self._apply_model_config_overrides(data)
        images = []
        if "messages" in data:
            # Process messages to extract images and handle content types
            for i, message in enumerate(data["messages"]):
                if isinstance(message.get("content"), list):
                    # Extract text content and images from list format
                    text_content = ""
                    message_images = []
                    for content_item in message["content"]:
                        if isinstance(content_item, dict):
                            if content_item.get("type") == "text":
                                text_content += content_item.get("text", "")
                            elif "image_url" in content_item:
                                message_images.append(content_item)
                            elif (
                                "video_url" in content_item
                                or content_item.get("type") == "video"
                            ):
                                # Extract video URL from different formats
                                video_url = None
                                if "video_url" in content_item:
                                    video_url = (
                                        content_item["video_url"].get("url", "")
                                        if isinstance(content_item["video_url"], dict)
                                        else content_item["video_url"]
                                    )
                                elif "video" in content_item:
                                    video_url = content_item["video"]

                                if video_url:
                                    logging.info(f"[Video] Processing video input")
                                    # Extract frames from video and convert to images
                                    video_fps = float(getenv("VIDEO_FPS", "1.0"))
                                    video_max_frames = int(
                                        getenv("VIDEO_MAX_FRAMES", "16")
                                    )
                                    frame_urls = extract_frames_from_video(
                                        video_url,
                                        fps=video_fps,
                                        max_frames=video_max_frames,
                                    )
                                    if frame_urls:
                                        # Add frame number info to help the model understand sequence
                                        for idx, frame_url in enumerate(frame_urls):
                                            message_images.append(
                                                {
                                                    "type": "image_url",
                                                    "image_url": {"url": frame_url},
                                                }
                                            )
                                        text_content = f"[Video with {len(frame_urls)} frames extracted at {video_fps} fps]\n{text_content}"
                                        logging.info(
                                            f"[Video] Extracted {len(frame_urls)} frames for vision processing"
                                        )
                                    else:
                                        logging.warning(
                                            "[Video] No frames could be extracted from video"
                                        )
                            elif "audio_url" in content_item:
                                audio_url = (
                                    content_item["audio_url"]["url"]
                                    if "url" in content_item["audio_url"]
                                    else content_item["audio_url"]
                                )
                                audio_format = "wav"
                                if audio_url.startswith("data:"):
                                    audio_url = audio_url.split(",")[1]
                                    audio_format = audio_url.split(";")[0]
                                else:
                                    audio_url = requests.get(audio_url).content
                                    audio_url = base64.b64encode(audio_url).decode(
                                        "utf-8"
                                    )
                                stt = self._get_stt()
                                self.resource_manager.mark_model_in_use(
                                    ModelType.STT, True
                                )
                                try:
                                    transcribed_audio = stt.transcribe_audio(
                                        base64_audio=audio_url,
                                        audio_format=audio_format,
                                    )
                                finally:
                                    self.resource_manager.mark_model_in_use(
                                        ModelType.STT, False
                                    )
                                self._destroy_stt()
                                text_content = f"Transcribed Audio: {transcribed_audio}\n\n{text_content}"
                        elif isinstance(content_item, str):
                            text_content += content_item

                    # Collect images for later processing
                    if message_images:
                        images.extend(message_images)

                    # For non-vision models or non-user messages, convert to string
                    # For vision models with the last user message, we'll handle this later
                    if not (
                        self.llm
                        and self.llm.is_vision
                        and message_images
                        and i == len(data["messages"]) - 1
                    ):
                        data["messages"][i]["content"] = text_content

            # Legacy handling for the old format (keeping for backward compatibility)
            # Skip if we already collected images in the modern format
            if not images and isinstance(data["messages"][-1]["content"], list):
                messages = data["messages"][-1]["content"]
                prompt = ""
                for message in messages:
                    if "text" in message:
                        prompt = message["text"]
                for message in messages:
                    if "image_url" in message:
                        images.append(message)
                    if "video_url" in message or message.get("type") == "video":
                        # Extract video URL from different formats
                        video_url = None
                        if "video_url" in message:
                            video_url = (
                                message["video_url"].get("url", "")
                                if isinstance(message["video_url"], dict)
                                else message["video_url"]
                            )
                        elif "video" in message:
                            video_url = message["video"]

                        if video_url:
                            logging.info(
                                f"[Video] Processing video input (legacy format)"
                            )
                            video_fps = float(getenv("VIDEO_FPS", "1.0"))
                            video_max_frames = int(getenv("VIDEO_MAX_FRAMES", "16"))
                            frame_urls = extract_frames_from_video(
                                video_url, fps=video_fps, max_frames=video_max_frames
                            )
                            if frame_urls:
                                for frame_url in frame_urls:
                                    images.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": frame_url},
                                        }
                                    )
                                prompt = (
                                    f"[Video with {len(frame_urls)} frames]\n{prompt}"
                                )
                                logging.info(
                                    f"[Video] Extracted {len(frame_urls)} frames for vision processing"
                                )
                    if "audio_url" in message:
                        audio_url = (
                            message["audio_url"]["url"]
                            if "url" in message["audio_url"]
                            else message["audio_url"]
                        )
                        audio_format = "wav"
                        if audio_url.startswith("data:"):
                            audio_url = audio_url.split(",")[1]
                            audio_format = audio_url.split(";")[0]
                        else:
                            audio_url = requests.get(audio_url).content
                            audio_url = base64.b64encode(audio_url).decode("utf-8")
                        stt = self._get_stt()
                        self.resource_manager.mark_model_in_use(ModelType.STT, True)
                        try:
                            transcribed_audio = stt.transcribe_audio(
                                base64_audio=audio_url, audio_format=audio_format
                            )
                        finally:
                            self.resource_manager.mark_model_in_use(
                                ModelType.STT, False
                            )
                        self._destroy_stt()
                        prompt = f"Transcribed Audio: {transcribed_audio}\n\n{prompt}"
                # Convert list content back to string for LLM compatibility
                data["messages"][-1]["content"] = prompt

        # Use current context size - don't pre-estimate tokens
        # The model's actual tokenizer will determine if context is sufficient
        # If context is exceeded, error handling will reload with larger context
        # This avoids inaccurate character-based estimation causing unnecessary reloads
        required_context = (
            self.current_context if self.current_context else self._optimal_context
        )

        # Lazy load LLM with requested model and context size
        # Determine target model
        requested_model = data.get("model")
        target_model = None

        if requested_model and self.available_models:
            # Find matching model name from available models
            for model_name in self.available_models:
                short_name = model_name.split("/")[-1].lower()
                requested_short = requested_model.split("/")[-1].lower()
                if (
                    model_name.lower() == requested_model.lower()
                    or short_name == requested_short
                ):
                    target_model = model_name
                    break

            # If requested model not found in available models, fallback to first available
            if target_model is None:
                target_model = self.available_models[0]
                logging.debug(
                    f"[LLM] Requested model '{requested_model}' not available, using '{target_model}'"
                )
        elif self.available_models:
            # No model requested, use first available
            target_model = self.available_models[0]

        # Vision model fallback: If the target model is a vision model but no images
        # are in the request, fall back to a non-vision model if one is available.
        # This optimizes resource usage since vision models are heavier.
        if target_model and not images and self._is_vision_model(target_model):
            non_vision_model = self._find_non_vision_model()
            if non_vision_model:
                logging.debug(
                    f"[LLM] Request to vision model '{target_model}' has no images, "
                    f"falling back to non-vision model '{non_vision_model}'"
                )
                target_model = non_vision_model

        # Lazy load the LLM with calculated context (estimated prompt tokens + 16k headspace)
        self._get_llm(target_model, required_context)
        data["model"] = self.current_llm_name

        if "stop" in data and data["stop"]:
            new_stop = list(self.llm.params.get("stop", []))
            if isinstance(data["stop"], list):
                new_stop.extend(data["stop"])
            else:
                new_stop.append(data["stop"])
            data["stop"] = new_stop
        if "audio_format" in data:
            base64_audio = (
                data["messages"][-1]["content"]
                if completion_type == "chat"
                else data["prompt"]
            )
            stt = self._get_stt()
            self.resource_manager.mark_model_in_use(ModelType.STT, True)
            try:
                prompt = stt.transcribe_audio(
                    base64_audio=base64_audio,
                    audio_format=data["audio_format"],
                )
            finally:
                self.resource_manager.mark_model_in_use(ModelType.STT, False)
            self._destroy_stt()
            if completion_type == "chat":
                data["messages"][-1]["content"] = prompt
            else:
                data["prompt"] = prompt
        user_message = (
            data["messages"][-1]["content"]
            if completion_type == "chat"
            else data["prompt"]
        )
        # Handle images with vision-capable LLM
        if self.llm and self.llm.is_vision and images:
            # xllamacpp expects images in base64 data URL format (PNG or JPEG)
            # Convert any remote URLs to base64 data URLs, and convert WebP/other formats to PNG
            from PIL import Image as PILImage
            from io import BytesIO

            processed_images = []
            for img in images:
                if "image_url" in img:
                    img_url = (
                        img["image_url"].get("url", "")
                        if isinstance(img["image_url"], dict)
                        else img["image_url"]
                    )
                    if img_url and not img_url.startswith("data:"):
                        # Fetch remote image and convert to base64
                        try:
                            headers = {
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                            }
                            img_response = requests.get(
                                img_url, timeout=30, headers=headers
                            )
                            img_response.raise_for_status()
                            content_type = img_response.headers.get(
                                "Content-Type", "image/jpeg"
                            )

                            # llama.cpp mmproj only supports certain formats (not WebP)
                            # Convert any non-standard format to PNG
                            if content_type in [
                                "image/webp",
                                "image/gif",
                                "image/bmp",
                                "image/tiff",
                                "image/avif",
                            ]:
                                try:
                                    pil_img = PILImage.open(
                                        BytesIO(img_response.content)
                                    )
                                    # Convert to RGB if necessary (for RGBA or palette images)
                                    if pil_img.mode in ("RGBA", "P", "LA"):
                                        pil_img = pil_img.convert("RGB")
                                    buffer = BytesIO()
                                    pil_img.save(buffer, format="PNG")
                                    img_bytes = buffer.getvalue()
                                    content_type = "image/png"
                                    logging.debug(
                                        f"[Vision] Converted {img_response.headers.get('Content-Type', 'unknown')} to PNG"
                                    )
                                except Exception as conv_err:
                                    logging.error(
                                        f"[Vision] Failed to convert image: {conv_err}"
                                    )
                                    img_bytes = img_response.content
                            else:
                                img_bytes = img_response.content

                            if not content_type.startswith("image/"):
                                content_type = "image/jpeg"
                            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                            processed_images.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{content_type};base64,{img_base64}"
                                    },
                                }
                            )
                            logging.debug(
                                f"[Vision] Converted remote image to base64 ({len(img_base64)} chars)"
                            )
                        except Exception as e:
                            logging.error(
                                f"[Vision] Failed to fetch remote image {img_url}: {e}"
                            )
                            continue
                    else:
                        # Already a data URL - check if it needs conversion
                        if img_url.startswith("data:image/webp") or img_url.startswith(
                            "data:image/gif"
                        ):
                            try:
                                # Extract base64 data and convert
                                header, encoded = img_url.split(",", 1)
                                img_bytes = base64.b64decode(encoded)
                                pil_img = PILImage.open(BytesIO(img_bytes))
                                if pil_img.mode in ("RGBA", "P", "LA"):
                                    pil_img = pil_img.convert("RGB")
                                buffer = BytesIO()
                                pil_img.save(buffer, format="PNG")
                                img_base64 = base64.b64encode(buffer.getvalue()).decode(
                                    "utf-8"
                                )
                                processed_images.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{img_base64}"
                                        },
                                    }
                                )
                                logging.debug(
                                    f"[Vision] Converted data URL WebP/GIF to PNG"
                                )
                            except Exception as conv_err:
                                logging.error(
                                    f"[Vision] Failed to convert data URL: {conv_err}"
                                )
                                processed_images.append(img)
                        else:
                            processed_images.append(img)

            if completion_type == "chat":
                # Build proper multimodal message with text + images
                user_text = data["messages"][-1]["content"]
                if isinstance(user_text, list):
                    # Extract text from list content format
                    user_text = " ".join(
                        [
                            item.get("text", "")
                            for item in user_text
                            if isinstance(item, dict) and item.get("type") == "text"
                        ]
                    )

                if processed_images:
                    # Create message with text + images in xllamacpp expected format
                    multimodal_content = [{"type": "text", "text": user_text}]
                    multimodal_content.extend(processed_images)
                    data["messages"][-1]["content"] = multimodal_content
                    logging.debug(
                        f"[Vision] Sending multimodal message with {len(processed_images)} image(s)"
                    )
                else:
                    # No images could be processed, fall back to text-only
                    data["messages"][-1]["content"] = user_text
                    logging.warning(
                        f"[Vision] No images could be processed, falling back to text-only"
                    )
        elif images and self.llm and not self.llm.is_vision:
            # Non-vision model received images - use vision model fallback
            logging.debug(
                f"[Vision Fallback] Non-vision model {self.current_llm_name} received {len(images)} image(s), using vision fallback"
            )
            user_text = (
                user_message if isinstance(user_message, str) else str(user_message)
            )

            # Get image description from vision model
            image_description = await self._describe_images_with_vision_model(
                images, user_text
            )

            if image_description:
                # Prepend image description to user message
                enhanced_message = f"[Image Description: {image_description}]\n\nUser's question: {user_text}"
                if completion_type == "chat":
                    data["messages"][-1]["content"] = enhanced_message
                else:
                    data["prompt"] = enhanced_message
                logging.debug(
                    "[Vision Fallback] Enhanced prompt with image description"
                )
            else:
                logging.warning(
                    "[Vision Fallback] Could not get image description, proceeding without images"
                )

        # Helper function to detect context size errors and retry with larger context
        def _is_context_error(error_msg: str) -> bool:
            error_lower = error_msg.lower()
            return any(
                pattern in error_lower
                for pattern in [
                    "context size",
                    "context length",
                    "exceeds",
                    "too long",
                    "token limit",
                    "max_tokens",
                    "maximum context",
                ]
            )

        def _estimate_prompt_tokens(messages_or_prompt, completion_type: str) -> int:
            """Estimate prompt tokens using character count approximation.

            We use chars/3.5 with a 10% buffer, which balances accuracy vs safety:
            - English text averages ~4 chars/token
            - Code/technical content averages ~3-3.5 chars/token
            - Chat templates add modest overhead (role markers, special tokens)
            - Over-estimating by too much forces unnecessary context scaling which
              pushes layers to CPU, dramatically hurting performance
            - Under-estimating is handled by the retry mechanism in the caller
            """
            total_chars = 0
            if completion_type == "chat" and isinstance(messages_or_prompt, list):
                for msg in messages_or_prompt:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        total_chars += len(content)
                    elif isinstance(content, list):
                        # Multimodal content
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    total_chars += len(item.get("text", ""))
                                elif "image_url" in item:
                                    # Images add significant tokens for vision models
                                    # Estimate ~1000 tokens per image
                                    total_chars += 4000
                            elif isinstance(item, str):
                                total_chars += len(item)
                    # Add role overhead (special tokens, markers)
                    total_chars += 50  # More realistic overhead per message
            else:
                # Completion mode or string content
                total_chars = len(str(messages_or_prompt))

            # Moderate estimate: chars/3.5 + 10% buffer for chat template overhead
            # This avoids over-estimating by 2x which forces context scaling and CPU offload
            estimated_tokens = int((total_chars / 3.5) * 1.1)
            return estimated_tokens

        async def _try_inference_with_context_retry(
            chat_mode: bool, data: dict
        ) -> dict:
            """Try inference, and if context error occurs, reload model with larger context and retry.

            For streaming requests, we pre-estimate tokens and ensure sufficient context
            BEFORE starting the stream, since streaming errors occur lazily during iteration
            and cannot be retried mid-stream.
            """
            max_retries = 3
            current_context = self.current_context or 16384
            is_streaming = data.get("stream", False)

            # For streaming requests, pre-estimate tokens and ensure context size
            # This prevents the lazy context error during stream iteration
            if is_streaming:
                messages_or_prompt = (
                    data.get("messages") if chat_mode else data.get("prompt", "")
                )
                estimated_tokens = _estimate_prompt_tokens(
                    messages_or_prompt, "chat" if chat_mode else "completion"
                )
                required_context = calculate_context_size(estimated_tokens)

                logging.info(
                    f"[LLM] Streaming pre-check: estimated {estimated_tokens:,} tokens, "
                    f"required context {required_context:,}, current context {current_context:,}"
                )

                if required_context > current_context:
                    logging.info(
                        f"[LLM] Streaming request: estimated {estimated_tokens:,} tokens, "
                        f"pre-loading {required_context//1024}k context (current: {current_context//1024}k)"
                    )
                    self._ensure_context_size(required_context)
                    current_context = required_context
            else:
                logging.info(
                    f"[LLM] Non-streaming request, stream={data.get('stream')}"
                )

            for attempt in range(max_retries):
                try:
                    if chat_mode:
                        logging.info(
                            f"[LLM] Calling llm.chat with stream={data.get('stream', False)}, context={self.current_context}"
                        )
                        result = self.llm.chat(**data)
                        logging.info(f"[LLM] llm.chat returned type: {type(result)}")
                        return result
                    else:
                        return self.llm.completion(**data)
                except Exception as e:
                    error_msg = str(e)
                    if _is_context_error(error_msg) and attempt < max_retries - 1:
                        # Try to extract n_prompt_tokens from error message
                        # Format: "... [n_prompt_tokens=21922, n_ctx=16384]"
                        import re

                        prompt_tokens_match = re.search(
                            r"n_prompt_tokens=(\d+)", error_msg
                        )
                        if prompt_tokens_match:
                            needed_tokens = int(prompt_tokens_match.group(1))
                            logging.debug(
                                f"[LLM] Extracted n_prompt_tokens={needed_tokens} from error"
                            )
                        else:
                            # Fallback: try to find any large number
                            numbers = re.findall(r"(\d+)", error_msg)
                            if numbers:
                                # Use the largest number as required context
                                needed_tokens = max(int(n) for n in numbers)
                            else:
                                # Double current context as fallback
                                needed_tokens = current_context * 2

                        # Calculate new context: actual tokens needed + 16k headspace
                        new_context = calculate_context_size(needed_tokens)

                        if new_context > current_context:
                            logging.warning(
                                f"[LLM] Context error detected ({needed_tokens} prompt tokens), reloading with {new_context//1024}k context..."
                            )
                            self._ensure_context_size(new_context)
                            current_context = new_context
                            continue

                    # Not a context error or max retries reached, raise
                    raise

            # Should not reach here, but just in case
            if chat_mode:
                return self.llm.chat(**data)
            else:
                return self.llm.completion(**data)

        # Check if local LLM is available, if not use fallback server
        if self.llm is None:
            logging.warning("[LLM] No local model available, using fallback server...")
            if completion_type == "chat":
                response = await self.fallback_inference(data["messages"])
                # Wrap in expected format
                response = {
                    "choices": [{"message": {"content": response}}],
                    "model": "fallback",
                }
            else:
                response = await self.fallback_inference(
                    [{"role": "user", "content": data.get("prompt", "")}]
                )
                response = {"choices": [{"text": response}], "model": "fallback"}
        elif completion_type == "chat":
            try:
                response = await _try_inference_with_context_retry(
                    chat_mode=True, data=data
                )
            except Exception as e:
                import traceback

                logging.error(f"[LLM] Chat completion failed: {e}")
                logging.error(f"[LLM] Full traceback: {traceback.format_exc()}")
                logging.error(f"[LLM] Data that caused failure: {data}")
                response = await self.fallback_inference(data["messages"])
        else:
            try:
                response = await _try_inference_with_context_retry(
                    chat_mode=False, data=data
                )
            except Exception as e:
                import traceback

                logging.error(f"[LLM] Completion failed: {e}")
                logging.error(f"[LLM] Full traceback: {traceback.format_exc()}")
                logging.error(f"[LLM] Data that caused failure: {data}")
                response = await self.fallback_inference(
                    [{"role": "user", "content": data.get("prompt", "")}]
                )
        generated_image = None
        if "temperature" not in data:
            data["temperature"] = 0.5
        if "top_p" not in data:
            data["top_p"] = 0.9
        # IMG is lazy loaded - try to get it if IMG_MODEL is configured
        # Skip image generation for streaming responses (response is a generator, not dict)
        is_streaming_response = data.get("stream", False)
        img = (
            self._get_img()
            if getenv("IMG_MODEL") and not is_streaming_response
            else None
        )
        if img_import_success and img:
            user_message = (
                data["messages"][-1]["content"]
                if completion_type == "chat"
                else data["prompt"]
            )
            if isinstance(user_message, list):
                # Extract text from list format
                user_message = ""
                for item in user_message:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            user_message = item.get("text", "")
                            break
                        elif "text" in item:
                            user_message = item["text"]
                            break
                for message in user_message if isinstance(user_message, list) else []:
                    if "image_url" in message:
                        if "url" in message["image_url"]:
                            if not message["image_url"]["url"].startswith("data:"):
                                user_message += (
                                    "Uploaded Image:"
                                    + message["image_url"]["url"]
                                    + "\n"
                                )
            response_text = (
                response["choices"][0]["text"]
                if completion_type != "chat"
                else response["choices"][0]["message"]["content"]
            )
            if "data:" in user_message:
                user_message = user_message.replace(
                    user_message.split("data:")[1].split("'")[0], ""
                )
            img_gen_prompt = f"Users message: {user_message} \n\n{'The user uploaded an image, one does not need generated unless the user is specifically asking.' if images else ''} **The assistant is acting as sentiment analysis expert and only responds with a concise YES or NO answer on if the user would like an image as visual or a picture generated. No other explanation is needed!**\nWould the user potentially like an image generated based on their message?\nAssistant: "
            logging.debug(f"[IMG] Decision maker prompt: {img_gen_prompt}")
            try:
                create_img = self.llm.chat(
                    messages=[{"role": "system", "content": img_gen_prompt}],
                    max_tokens=10,
                    temperature=data["temperature"],
                    top_p=data["top_p"],
                )
            except:
                create_img = await self.fallback_inference(
                    [{"role": "system", "content": img_gen_prompt}]
                )
            create_img = str(create_img["choices"][0]["message"]["content"]).lower()
            logging.debug(f"[IMG] Decision maker response: {create_img}")
            if "yes" in create_img or "es," in create_img:
                img_prompt = f"**The assistant is acting as a Stable Diffusion Prompt Generator.**\n\nUsers message: {user_message} \nAssistant response: {response_text} \n\nImportant rules to follow:\n- Describe subjects in detail, specify image type (e.g., digital illustration), art style (e.g., steampunk), and background. Include art inspirations (e.g., Art Station, specific artists). Detail lighting, camera (type, lens, view), and render (resolution, style). The weight of a keyword can be adjusted by using the syntax (((keyword))) , put only those keyword inside ((())) which is very important because it will have more impact so anything wrong will result in unwanted picture so be careful. Realistic prompts: exclude artist, specify lens. Separate with double lines. Max 60 words, avoiding 'real' for fantastical.\n- Based on the message from the user and response of the assistant, you will need to generate one detailed stable diffusion image generation prompt based on the context of the conversation to accompany the assistant response.\n- The prompt can only be up to 60 words long, so try to be concise while using enough descriptive words to make a proper prompt.\n- Following all rules will result in a $2000 tip that you can spend on anything!\n- Must be in markdown code block to be parsed out and only provide prompt in the code block, nothing else.\nStable Diffusion Prompt Generator: "
                try:
                    image_generation_prompt = self.llm.chat(
                        messages=[{"role": "system", "content": img_prompt}],
                        max_tokens=100,
                        temperature=data["temperature"],
                        top_p=data["top_p"],
                    )
                except:
                    image_generation_prompt = await self.fallback_inference(
                        [{"role": "system", "content": img_prompt}]
                    )
                image_generation_prompt = str(
                    image_generation_prompt["choices"][0]["message"]["content"]
                )
                logging.debug(
                    f"[IMG] Image generation response: {image_generation_prompt}"
                )
                if "```markdown" in image_generation_prompt:
                    image_generation_prompt = image_generation_prompt.split(
                        "```markdown"
                    )[1]
                    image_generation_prompt = image_generation_prompt.split("```")[0]
                self.resource_manager.mark_model_in_use(ModelType.IMG, True)
                try:
                    generated_image = self.img.generate(prompt=image_generation_prompt)
                finally:
                    self.resource_manager.mark_model_in_use(ModelType.IMG, False)
            # Destroy IMG model after use to free VRAM (even if no image was generated)
            self._destroy_img()
        audio_response = None
        if "voice" in data:
            text_response = (
                response["choices"][0]["text"]
                if completion_type != "chat"
                else response["choices"][0]["message"]["content"]
            )
            language = data["language"] if "language" in data else "en"
            tts = self._get_tts()
            self.resource_manager.mark_model_in_use(ModelType.TTS, True)
            try:
                audio_response = await tts.generate(
                    text=text_response,
                    voice=data["voice"],
                    language=language,
                    local_uri=self.local_uri,
                )
            finally:
                self.resource_manager.mark_model_in_use(ModelType.TTS, False)
            self._destroy_tts()
            if completion_type != "chat":
                response["choices"][0]["text"] = f"{text_response}\n{audio_response}"
            else:
                response["choices"][0]["message"][
                    "content"
                ] = f"{text_response}\n{audio_response}"
        if generated_image:
            if completion_type != "chat":
                response["choices"][0]["text"] += f"\n\n{generated_image}"
            else:
                response["choices"][0]["message"]["content"] += f"\n\n{generated_image}"

        # Only log JSON if response is not a generator (streaming mode)
        is_streaming = (
            hasattr(response, "__next__")
            or hasattr(response, "__iter__")
            and not isinstance(response, (dict, list))
        )

        if not is_streaming:
            logging.debug(f"[ezlocalai] {json.dumps(response, indent=2)}")
            # Keep the model loaded - no need to reload after each request
            # The higher context model works fine for smaller prompts too
        else:
            logging.debug(f"[ezlocalai] Streaming response generated")
            # For streaming, wrap the generator to handle cleanup after consumption
            original_response = response
            using_large = self._using_large_model
            pipes_self = self  # Capture self for use in wrapper
            data_copy = data.copy()  # Capture data for potential retry

            def streaming_wrapper():
                try:
                    for chunk in original_response:
                        yield chunk
                except Exception as e:
                    error_msg = str(e)
                    # Check if this is a context size error
                    if _is_context_error(error_msg):
                        # Extract token count from error if available
                        import re

                        prompt_tokens_match = re.search(
                            r"n_prompt_tokens=(\d+)", error_msg
                        )
                        if prompt_tokens_match:
                            needed_tokens = int(prompt_tokens_match.group(1))
                        else:
                            # Try to find any number that looks like a token count
                            numbers = re.findall(r"(\d{4,})", error_msg)  # 4+ digits
                            needed_tokens = (
                                max(int(n) for n in numbers) if numbers else 0
                            )

                        logging.error(
                            f"[STREAMING] Context size error during streaming. "
                            f"Prompt required {needed_tokens:,} tokens but context was insufficient. "
                            f"The model will be reloaded with larger context for the next request."
                        )
                        # Pre-load larger context for next request
                        if needed_tokens > 0:
                            new_context = calculate_context_size(needed_tokens)
                            pipes_self._ensure_context_size(new_context)
                    # Re-raise the error to let caller handle it
                    raise
                finally:
                    # Close the original LLM generator so GeneratorExit propagates
                    # to _chat_stream(), which sets cancel_event to free the slot.
                    if hasattr(original_response, "close"):
                        original_response.close()
                # No cleanup needed - keep the model loaded for subsequent requests

            response = streaming_wrapper()

        return response, audio_response
