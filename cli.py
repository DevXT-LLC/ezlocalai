#!/usr/bin/env python3
"""
ezlocalai CLI - Run local AI inference with ease.

This lightweight CLI manages Docker containers for local LLM, TTS, STT, and image generation.
It automatically detects GPU availability and runs the appropriate container.

Usage:
    ezlocalai start [--model MODEL] [--uri URI] [--api-key KEY] [--ngrok TOKEN]
    ezlocalai stop
    ezlocalai restart [--model MODEL] [--uri URI] [--api-key KEY] [--ngrok TOKEN]
    ezlocalai status
    ezlocalai logs [-f]
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Version
__version__ = "1.0.0"

# Configuration
DOCKER_IMAGE = "joshxt/ezlocalai:latest"
DOCKER_IMAGE_CUDA = "ezlocalai:cuda"  # Built locally, not from DockerHub
CONTAINER_NAME = "ezlocalai"
DEFAULT_PORT = 8091
UI_PORT = 8502
STATE_DIR = Path.home() / ".ezlocalai"
STATE_DIR.mkdir(parents=True, exist_ok=True)
ENV_FILE = STATE_DIR / ".env"
LOG_FILE = STATE_DIR / "ezlocalai.log"
REPO_URL = "https://github.com/DevXT-LLC/ezlocalai.git"
REPO_DIR = STATE_DIR / "repo"


class CLIError(RuntimeError):
    """Raised for recoverable CLI errors."""


def print_banner():
    """Print the ezLocalai banner ascii art banner"""
    print(
        """
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║  ███████╗███████╗██╗      ██████╗  ██████╗ █████╗ ██╗      █████╗ ██╗  ║
║  ██╔════╝╚══███╔╝██║     ██╔═══██╗██╔════╝██╔══██╗██║     ██╔══██╗██║  ║
║  █████╗    ███╔╝ ██║     ██║   ██║██║     ███████║██║     ███████║██║  ║
║  ██╔══╝   ███╔╝  ██║     ██║   ██║██║     ██╔══██║██║     ██╔══██║██║  ║
║  ███████╗███████╗███████╗╚██████╔╝╚██████╗██║  ██║███████╗██║  ██║██║  ║
║  ╚══════╝╚══════╝╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
"""
    )


def is_tool_installed(tool: str) -> bool:
    """Check if a command-line tool is installed."""
    try:
        result = subprocess.run(
            [tool, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def is_docker_running() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def has_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except FileNotFoundError:
        pass
    return False


def has_nvidia_container_toolkit() -> bool:
    """Check if NVIDIA Container Toolkit is installed."""
    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "nvidia/cuda:12.4.1-base-ubuntu22.04",
                "nvidia-smi",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_nvidia_gpu_info() -> Optional[str]:
    """Get NVIDIA GPU information."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def prompt_user(prompt: str, default: str = "") -> str:
    """Prompt user for input with optional default."""
    if default:
        user_input = input(f"{prompt} (default: {default}): ").strip()
    else:
        user_input = input(f"{prompt}: ").strip()
    return user_input if user_input else default


def clone_or_update_repo() -> bool:
    """Clone or update the ezlocalai repository for building CUDA image."""
    if REPO_DIR.exists():
        print("📦 Updating ezlocalai repository...")
        result = subprocess.run(
            ["git", "pull"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"⚠️  Failed to update repo: {result.stderr}")
            # Try to continue with existing repo
            return True
        print("✅ Repository updated")
        return True
    else:
        print("📦 Cloning ezlocalai repository...")
        result = subprocess.run(
            ["git", "clone", REPO_URL, str(REPO_DIR)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"❌ Failed to clone repo: {result.stderr}")
            return False
        print("✅ Repository cloned")
        return True


def build_cuda_image() -> bool:
    """Build the CUDA Docker image from source using docker-compose."""
    if not clone_or_update_repo():
        return False

    print("\n🔨 Building CUDA image (this may take 10-20 minutes)...")
    print("   Building from: docker-compose-cuda.yml")

    # Build using docker-compose (handles complex builds better)
    result = subprocess.run(
        ["docker", "compose", "-f", "docker-compose-cuda.yml", "build"],
        cwd=REPO_DIR,
        check=False,
    )

    if result.returncode != 0:
        print("❌ Failed to build CUDA image")
        return False

    # Tag the image with our expected name
    # docker-compose names it based on folder: repo-ezlocalai
    print("   Tagging image as ezlocalai:cuda...")
    tag_result = subprocess.run(
        ["docker", "tag", "repo-ezlocalai:latest", DOCKER_IMAGE_CUDA],
        check=False,
    )

    if tag_result.returncode != 0:
        print("⚠️  Failed to tag image, trying alternative name...")
        # Try with the folder name from REPO_DIR
        folder_name = REPO_DIR.name
        alt_name = f"{folder_name}-ezlocalai:latest"
        tag_result = subprocess.run(
            ["docker", "tag", alt_name, DOCKER_IMAGE_CUDA],
            check=False,
        )

    print("✅ CUDA image built successfully")
    return True


def cuda_image_exists() -> bool:
    """Check if the CUDA image exists locally."""
    result = subprocess.run(
        ["docker", "images", "-q", DOCKER_IMAGE_CUDA],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def install_docker_linux() -> bool:
    """Attempt to install Docker on Linux."""
    system = platform.system().lower()
    if system != "linux":
        return False

    install = prompt_user(
        "Docker is not installed. Would you like to install it? (y/n)", "y"
    )
    if install.lower() != "y":
        return False

    print("\n🔧 Installing Docker...")

    # Detect package manager and install
    if is_tool_installed("apt-get"):
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y docker.io",
            "sudo systemctl start docker",
            "sudo systemctl enable docker",
            "sudo usermod -aG docker $USER",
        ]
    elif is_tool_installed("dnf"):
        commands = [
            "sudo dnf install -y docker",
            "sudo systemctl start docker",
            "sudo systemctl enable docker",
            "sudo usermod -aG docker $USER",
        ]
    elif is_tool_installed("yum"):
        commands = [
            "sudo yum install -y docker",
            "sudo systemctl start docker",
            "sudo systemctl enable docker",
            "sudo usermod -aG docker $USER",
        ]
    elif is_tool_installed("pacman"):
        commands = [
            "sudo pacman -Sy --noconfirm docker",
            "sudo systemctl start docker",
            "sudo systemctl enable docker",
            "sudo usermod -aG docker $USER",
        ]
    else:
        print("❌ Unsupported package manager. Please install Docker manually.")
        return False

    try:
        for cmd in commands:
            print(f"  Running: {cmd}")
            subprocess.run(cmd, shell=True, check=True)

        print("\n✅ Docker installed successfully!")
        print(
            "⚠️  You may need to log out and back in for group changes to take effect."
        )
        print("   Or run: newgrp docker")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to install Docker: {e}")
        return False


def install_nvidia_container_toolkit() -> bool:
    """Attempt to install NVIDIA Container Toolkit on Linux."""
    system = platform.system().lower()
    if system != "linux":
        return False

    install = prompt_user(
        "\n🎮 NVIDIA GPU detected but Container Toolkit not found.\n"
        "Would you like to install NVIDIA Container Toolkit for GPU acceleration? (y/n)",
        "y",
    )
    if install.lower() != "y":
        return False

    print("\n🔧 Installing NVIDIA Container Toolkit...")

    try:
        # Add NVIDIA repository and install
        if is_tool_installed("apt-get"):
            commands = [
                "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
                "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list",
                "sudo apt-get update",
                "sudo apt-get install -y nvidia-container-toolkit",
                "sudo nvidia-ctk runtime configure --runtime=docker",
                "sudo systemctl restart docker",
            ]
        elif is_tool_installed("dnf") or is_tool_installed("yum"):
            pkg_mgr = "dnf" if is_tool_installed("dnf") else "yum"
            commands = [
                "curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo",
                f"sudo {pkg_mgr} install -y nvidia-container-toolkit",
                "sudo nvidia-ctk runtime configure --runtime=docker",
                "sudo systemctl restart docker",
            ]
        else:
            print(
                "❌ Unsupported package manager. Please install NVIDIA Container Toolkit manually:"
            )
            print(
                "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            )
            return False

        for cmd in commands:
            print(f"  Running: {cmd}")
            subprocess.run(cmd, shell=True, check=True)

        print("\n✅ NVIDIA Container Toolkit installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to install NVIDIA Container Toolkit: {e}")
        print("   You can still run ezlocalai on CPU.")
        return False


def check_prerequisites() -> tuple[bool, bool]:
    """
    Check and install prerequisites.

    Returns:
        Tuple of (docker_available, gpu_available)
    """
    system = platform.system().lower()

    # Check Docker
    if not is_tool_installed("docker"):
        print("❌ Docker is not installed.")
        if system == "linux":
            if not install_docker_linux():
                print("\n📦 Please install Docker manually:")
                print("   https://docs.docker.com/engine/install/")
                sys.exit(1)
        else:
            print("\n📦 Please install Docker Desktop:")
            if system == "darwin":
                print("   https://docs.docker.com/desktop/install/mac-install/")
            elif system == "windows":
                print("   https://docs.docker.com/desktop/install/windows-install/")
            else:
                print("   https://docs.docker.com/engine/install/")
            sys.exit(1)

    # Check if Docker daemon is running
    if not is_docker_running():
        print("❌ Docker daemon is not running.")
        print("   Please start Docker and try again.")
        if system == "linux":
            print("   Try: sudo systemctl start docker")
        sys.exit(1)

    print("✅ Docker is installed and running")

    # Check GPU
    gpu_available = False
    if has_nvidia_gpu():
        gpu_info = get_nvidia_gpu_info()
        print(f"✅ NVIDIA GPU detected: {gpu_info}")

        if has_nvidia_container_toolkit():
            print("✅ NVIDIA Container Toolkit is installed")
            gpu_available = True
        else:
            if system == "linux":
                if install_nvidia_container_toolkit():
                    gpu_available = has_nvidia_container_toolkit()
                else:
                    print("⚠️  Running on CPU (GPU acceleration not available)")
            else:
                print("⚠️  NVIDIA Container Toolkit not detected.")
                print("   For GPU acceleration, install it from:")
                print(
                    "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
                )
                print("   Continuing with CPU mode...")
    else:
        print("ℹ️  No NVIDIA GPU detected, running on CPU")

    return True, gpu_available


def is_container_running() -> bool:
    """Check if ezlocalai container is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name={CONTAINER_NAME}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(result.stdout.strip())
    except FileNotFoundError:
        return False


def get_container_status() -> Optional[str]:
    """Get container status (running, exited, etc.)."""
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--format",
                "{{.Status}}",
                "-f",
                f"name={CONTAINER_NAME}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.stdout.strip() else None
    except FileNotFoundError:
        return None


def load_env_file() -> dict:
    """Load environment variables from state file."""
    env_vars = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                # Remove quotes
                value = value.strip().strip('"').strip("'")
                env_vars[key.strip()] = value
    return env_vars


def save_env_file(env_vars: dict) -> None:
    """Save environment variables to state file."""
    lines = []
    for key, value in sorted(env_vars.items()):
        lines.append(f'{key}="{value}"')
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_default_env() -> dict:
    """Get default environment variables."""
    return {
        "EZLOCALAI_URL": f"http://localhost:{DEFAULT_PORT}",
        "EZLOCALAI_API_KEY": "",
        "DEFAULT_MODEL": "unsloth/Qwen3-VL-4B-Instruct-GGUF,unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "WHISPER_MODEL": "base",
        "IMG_MODEL": "",
        "NGROK_TOKEN": "",
        "MAIN_GPU": "0",
        "MAX_CONCURRENT_REQUESTS": "2",
        "MAX_QUEUE_SIZE": "100",
        "REQUEST_TIMEOUT": "300",
    }


def start_container(
    model: Optional[str] = None,
    uri: Optional[str] = None,
    api_key: Optional[str] = None,
    ngrok: Optional[str] = None,
    use_gpu: bool = False,
) -> None:
    """Start the ezlocalai container."""

    # Check if already running
    if is_container_running():
        print(f"✅ ezlocalai is already running!")
        print(f"   API: http://localhost:{DEFAULT_PORT}")
        print(f"   UI:  http://localhost:{UI_PORT}")
        return

    # Load existing env or defaults
    env_vars = get_default_env()
    saved_env = load_env_file()
    env_vars.update(saved_env)

    # Apply command line overrides
    if model:
        env_vars["DEFAULT_MODEL"] = model
    if uri:
        env_vars["EZLOCALAI_URL"] = uri
    if api_key:
        env_vars["EZLOCALAI_API_KEY"] = api_key
    if ngrok:
        env_vars["NGROK_TOKEN"] = ngrok

    # Save updated env
    save_env_file(env_vars)

    # Prepare data directories
    data_dir = STATE_DIR / "data"
    dirs = ["models", "outputs", "voices", "hf"]
    for d in dirs:
        (data_dir / d).mkdir(parents=True, exist_ok=True)

    # Remove existing stopped container
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True,
        check=False,
    )

    # Select image
    image = DOCKER_IMAGE_CUDA if use_gpu else DOCKER_IMAGE

    # Build docker run command
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        CONTAINER_NAME,
        "-p",
        f"{DEFAULT_PORT}:{DEFAULT_PORT}",
        "-p",
        f"{UI_PORT}:{UI_PORT}",
        "-v",
        f"{data_dir / 'models'}:/app/models",
        "-v",
        f"{data_dir / 'outputs'}:/app/outputs",
        "-v",
        f"{data_dir / 'voices'}:/app/voices",
        "-v",
        f"{data_dir / 'hf'}:/root/.cache/huggingface/hub",
        "--restart",
        "unless-stopped",
    ]

    # Add GPU flag if available
    if use_gpu:
        cmd.extend(["--gpus", "all"])

    # Add environment variables
    for key, value in env_vars.items():
        if value:  # Only add non-empty values
            cmd.extend(["-e", f"{key}={value}"])

    # Add image
    cmd.append(image)

    # Handle image: pull for CPU, build for CUDA
    if use_gpu:
        # CUDA image must be built locally (too large for DockerHub)
        if not cuda_image_exists():
            print("\n🔨 CUDA image not found, building from source...")
            if not build_cuda_image():
                print("❌ Failed to build CUDA image. Falling back to CPU mode.")
                use_gpu = False
                image = DOCKER_IMAGE
                cmd[-1] = image  # Update image in command
                # Remove --gpus flag
                if "--gpus" in cmd:
                    idx = cmd.index("--gpus")
                    cmd.pop(idx)  # Remove --gpus
                    cmd.pop(idx)  # Remove "all"
        # Note: We don't auto-rebuild on start - use 'ezlocalai update' to rebuild
    else:
        # CPU image: pull from DockerHub
        print(f"\n📦 Pulling latest image: {image}")
        pull_result = subprocess.run(
            ["docker", "pull", image],
            capture_output=True,
            text=True,
            check=False,
        )
        if pull_result.returncode != 0:
            print(f"⚠️  Failed to pull latest image, using cached version if available")

    # Start container
    mode = "GPU" if use_gpu else "CPU"
    print(f"\n🚀 Starting ezlocalai ({mode} mode)...")
    print(f"   Model: {env_vars.get('DEFAULT_MODEL', 'default')}")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print(f"❌ Failed to start container: {result.stderr}")
        sys.exit(1)

    # Wait for container to be healthy
    print("\n⏳ Waiting for server to be ready...")
    max_wait = 300  # 5 minutes for model download
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            import urllib.request

            req = urllib.request.Request(f"http://localhost:{DEFAULT_PORT}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    print("\n✅ ezlocalai is ready!")
                    print(f"\n   🌐 API: http://localhost:{DEFAULT_PORT}")
                    print(f"   🖥️  UI:  http://localhost:{UI_PORT}")
                    print(f"\n   📖 API Docs: http://localhost:{DEFAULT_PORT}/docs")
                    return
        except Exception:
            pass

        # Show progress
        elapsed = int(time.time() - start_time)
        if elapsed % 10 == 0 and elapsed > 0:
            print(f"   Still loading... ({elapsed}s)")
        time.sleep(2)

    print("\n⚠️  Container started but server not responding yet.")
    print("   This is normal for first-time model downloads.")
    print(f"   Check logs with: ezlocalai logs")
    print(f"\n   🌐 API: http://localhost:{DEFAULT_PORT}")
    print(f"   🖥️  UI:  http://localhost:{UI_PORT}")


def stop_container() -> None:
    """Stop the ezlocalai container."""
    if not is_container_running():
        print("ℹ️  ezlocalai is not running")
        return

    print("🛑 Stopping ezlocalai...")
    result = subprocess.run(
        ["docker", "stop", CONTAINER_NAME],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        print("✅ ezlocalai stopped")
    else:
        print(f"❌ Failed to stop container: {result.stderr}")


def show_status() -> None:
    """Show ezlocalai status."""
    status = get_container_status()

    # Load config (saved or defaults)
    env_vars = load_env_file()
    if not env_vars:
        env_vars = get_default_env()

    if not status:
        print("ℹ️  ezlocalai container not found")
        print("   Run 'ezlocalai start' to start the server")
    elif is_container_running():
        print(f"✅ ezlocalai is running")
        print(f"   Status: {status}")
        print(f"\n   🌐 API: http://localhost:{DEFAULT_PORT}")
        print(f"   🖥️  UI:  http://localhost:{UI_PORT}")

        # Show loaded model from API
        try:
            import urllib.request
            import json

            req = urllib.request.Request(f"http://localhost:{DEFAULT_PORT}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                models = [m.get("id") for m in data.get("data", [])]
                if models:
                    print(f"\n   🧠 Active model: {models[0]}")
        except Exception:
            pass
    else:
        print(f"❌ ezlocalai is not running")
        print(f"   Status: {status}")
        print("   Run 'ezlocalai start' to start the server")

    # Always show configuration
    print(f"\n   ⚙️  Configuration:")

    # LLM models
    models = env_vars.get("DEFAULT_MODEL", "unsloth/Qwen3-VL-4B-Instruct-GGUF")
    configured = [m.strip() for m in models.split(",")]
    print(f"      LLM models:")
    for m in configured:
        print(f"        - {m}")

    # Whisper
    whisper = env_vars.get("WHISPER_MODEL", "base")
    if whisper:
        print(f"      Speech-to-text: {whisper}")
    else:
        print(f"      Speech-to-text: disabled")

    # Image model
    img_model = env_vars.get("IMG_MODEL", "")
    if img_model:
        print(f"      Image generation: {img_model}")
    else:
        print(f"      Image generation: disabled")

    print(f"\n   💡 To change settings:")
    print(f"      ezlocalai start --model <model>")
    print(f"      Or edit ~/.ezlocalai/.env")


def show_logs(follow: bool = False) -> None:
    """Show container logs."""
    if not get_container_status():
        print("ℹ️  ezlocalai container not found")
        return

    cmd = ["docker", "logs"]
    if follow:
        cmd.append("-f")
    cmd.extend(["--tail", "100", CONTAINER_NAME])

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        pass


def update_images() -> None:
    """Pull latest CPU image and rebuild CUDA image."""
    print("📦 Updating ezlocalai...")

    # Pull CPU image from DockerHub
    print(f"\n📥 Pulling {DOCKER_IMAGE}...")
    result = subprocess.run(
        ["docker", "pull", DOCKER_IMAGE],
        check=False,
    )
    if result.returncode != 0:
        print(f"   ⚠️  Failed to pull CPU image")
    else:
        print(f"   ✅ CPU image updated")

    # Build CUDA image from source
    if has_nvidia_gpu():
        print(f"\n🔨 Building CUDA image from source...")
        if build_cuda_image():
            print("   ✅ CUDA image rebuilt")
        else:
            print("   ⚠️  Failed to build CUDA image")
    else:
        print("\nℹ️  No NVIDIA GPU detected, skipping CUDA image build")

    print("\n✅ Update complete!")
    print("   Run 'ezlocalai restart' to use the new version.")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="ezlocalai - Local AI inference made easy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ezlocalai start                           Start with default settings
  ezlocalai start --model unsloth/Qwen3-4B  Start with specific model
  ezlocalai restart                         Restart the server
  ezlocalai stop                            Stop the server
  ezlocalai status                          Show server status
  ezlocalai logs -f                         Follow logs

Environment:
  Configuration is stored in ~/.ezlocalai/.env
  Data (models, outputs) is stored in ~/.ezlocalai/data/
""",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"ezlocalai {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start ezlocalai server")
    start_parser.add_argument(
        "--model",
        "-m",
        help="Model(s) to load (comma-separated for multiple)",
        default=None,
    )
    start_parser.add_argument(
        "--uri",
        help="Server URI (default: http://localhost:8091)",
        default=None,
    )
    start_parser.add_argument(
        "--api-key",
        help="API key for authentication",
        default=None,
    )
    start_parser.add_argument(
        "--ngrok",
        help="Ngrok token for public URL",
        default=None,
    )

    # Stop command
    subparsers.add_parser("stop", help="Stop ezlocalai server")

    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart ezlocalai server")
    restart_parser.add_argument(
        "--model",
        "-m",
        help="Model(s) to load (comma-separated for multiple)",
        default=None,
    )
    restart_parser.add_argument(
        "--uri",
        help="Server URI (default: http://localhost:8091)",
        default=None,
    )
    restart_parser.add_argument(
        "--api-key",
        help="API key for authentication",
        default=None,
    )
    restart_parser.add_argument(
        "--ngrok",
        help="Ngrok token for public URL",
        default=None,
    )

    # Status command
    subparsers.add_parser("status", help="Show server status")

    # Update command
    subparsers.add_parser("update", help="Pull latest Docker image")

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show server logs")
    logs_parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Follow log output",
    )

    args = parser.parse_args()

    if not args.command:
        print_banner()
        parser.print_help()
        sys.exit(0)

    # Print banner for main commands
    if args.command in ("start", "restart"):
        print_banner()

    # Check prerequisites for start/restart
    gpu_available = False
    if args.command in ("start", "restart"):
        _, gpu_available = check_prerequisites()
        print()

    # Execute command
    if args.command == "start":
        start_container(
            model=args.model,
            uri=args.uri,
            api_key=args.api_key,
            ngrok=args.ngrok,
            use_gpu=gpu_available,
        )

    elif args.command == "stop":
        stop_container()

    elif args.command == "restart":
        stop_container()
        time.sleep(2)
        start_container(
            model=args.model,
            uri=args.uri,
            api_key=args.api_key,
            ngrok=args.ngrok,
            use_gpu=gpu_available,
        )

    elif args.command == "status":
        show_status()

    elif args.command == "update":
        update_images()

    elif args.command == "logs":
        show_logs(follow=args.follow)


if __name__ == "__main__":
    main()
