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
    ezlocalai prompt "your prompt" [-m MODEL] [-temp TEMPERATURE] [-tp TOP_P] [-image PATH]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# Version
__version__ = "1.0.5"

# Configuration
DOCKER_IMAGE = "joshxt/ezlocalai:latest"
DOCKER_IMAGE_CUDA = "ezlocalai:cuda"  # Built locally, not from DockerHub
DOCKER_IMAGE_ROCM = "ezlocalai:rocm"  # Built locally for AMD GPUs
CONTAINER_NAME = "ezlocalai"
DEFAULT_PORT = 8091
STATE_DIR = Path.home() / ".ezlocalai"
STATE_DIR.mkdir(parents=True, exist_ok=True)
ENV_FILE = STATE_DIR / ".env"
LOG_FILE = STATE_DIR / "ezlocalai.log"
REPO_URL = "https://github.com/DevXT-LLC/ezlocalai.git"
REPO_DIR = STATE_DIR / "repo"


def is_ezlocalai_folder(folder: Path) -> bool:
    """Check if the given folder is the ezlocalai source folder.

    Detects the ezlocalai folder by checking for key files that exist
    in the source repository but not in typical installation locations.
    """
    key_files = [
        "docker-compose.yml",
        "docker-compose-cuda.yml",
        "Dockerfile",
        "cuda.Dockerfile",
        "app.py",  # Main app file
    ]
    key_dirs = [
        "ezlocalai",  # Python module folder
    ]
    files_exist = all((folder / f).exists() for f in key_files)
    dirs_exist = all((folder / d).is_dir() for d in key_dirs)
    return files_exist and dirs_exist


def get_ezlocalai_source_dir() -> Optional[Path]:
    """Get the ezlocalai source directory if running from within it.

    Returns the current working directory if it's the ezlocalai folder,
    otherwise returns None.
    """
    cwd = Path.cwd()
    if is_ezlocalai_folder(cwd):
        return cwd
    return None


class CLIError(RuntimeError):
    """Raised for recoverable CLI errors."""


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
                "nvidia/cuda:12.8.1-base-ubuntu24.04",
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


def has_amd_gpu() -> bool:
    """Check if AMD GPU is available via ROCm."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except FileNotFoundError:
        pass
    # Also check for /dev/kfd which indicates ROCm-capable hardware
    if Path("/dev/kfd").exists() and Path("/dev/dri").exists():
        return True
    return False


def has_rocm_support() -> bool:
    """Check if ROCm is properly installed and functional."""
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and "Agent" in result.stdout:
            return True
    except FileNotFoundError:
        pass
    return False


def get_amd_gpu_info() -> Optional[str]:
    """Get AMD GPU information."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            # Parse the output to get GPU name
            for line in result.stdout.splitlines():
                if "GPU" in line or "Card" in line:
                    return line.strip()
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    # Fallback: try lspci
    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "VGA" in line and ("AMD" in line or "Radeon" in line):
                    return line.split(":")[-1].strip()
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


def clone_or_update_repo() -> Path:
    """Clone or update the ezlocalai repository for building CUDA image.

    Returns the path to the ezlocalai source directory (either local or cloned).
    """
    # First check if we're running from within the ezlocalai folder
    local_source = get_ezlocalai_source_dir()
    if local_source:
        print("📦 Using local ezlocalai source folder...")
        print(f"   Path: {local_source}")
        return local_source

    # Fall back to cloning/updating the repo
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
            return REPO_DIR
        print("✅ Repository updated")
        return REPO_DIR
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
            return None
        print("✅ Repository cloned")
        return REPO_DIR


def build_cuda_image() -> bool:
    """Build the CUDA Docker image from source using docker-compose."""
    source_dir = clone_or_update_repo()
    if not source_dir:
        return False

    print("\n🔨 Building CUDA image (this may take 10-20 minutes)...")
    print("   Building from: docker-compose-cuda.yml")
    print(f"   Source directory: {source_dir}")

    # Build using docker-compose (handles complex builds better)
    result = subprocess.run(
        ["docker", "compose", "-f", "docker-compose-cuda.yml", "build"],
        cwd=source_dir,
        check=False,
    )

    if result.returncode != 0:
        print("❌ Failed to build CUDA image")
        return False

    # Tag the image with our expected name
    # docker-compose names it based on folder name
    print("   Tagging image as ezlocalai:cuda...")

    # Determine the expected image name based on folder
    folder_name = source_dir.name
    expected_names = [
        f"{folder_name}-ezlocalai:latest",
        "repo-ezlocalai:latest",
        f"{folder_name}_ezlocalai:latest",
    ]

    tagged = False
    for expected_name in expected_names:
        tag_result = subprocess.run(
            ["docker", "tag", expected_name, DOCKER_IMAGE_CUDA],
            capture_output=True,
            check=False,
        )
        if tag_result.returncode == 0:
            tagged = True
            break

    if not tagged:
        print("⚠️  Could not tag image, trying to find it...")
        # List images and try to find one that matches
        list_result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if list_result.returncode == 0:
            for line in list_result.stdout.splitlines():
                if "ezlocalai" in line.lower() and "cuda" not in line:
                    subprocess.run(
                        ["docker", "tag", line.strip(), DOCKER_IMAGE_CUDA],
                        check=False,
                    )
                    tagged = True
                    break

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


def rocm_image_exists() -> bool:
    """Check if the ROCm image exists locally."""
    result = subprocess.run(
        ["docker", "images", "-q", DOCKER_IMAGE_ROCM],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def build_rocm_image() -> bool:
    """Build the ROCm Docker image from source using docker-compose."""
    source_dir = clone_or_update_repo()
    if not source_dir:
        return False

    print("\n🔨 Building ROCm image (this may take 10-20 minutes)...")
    print("   Building from: docker-compose-rocm.yml")
    print(f"   Source directory: {source_dir}")

    # Build using docker-compose (handles complex builds better)
    result = subprocess.run(
        ["docker", "compose", "-f", "docker-compose-rocm.yml", "build"],
        cwd=source_dir,
        check=False,
    )

    if result.returncode != 0:
        print("❌ Failed to build ROCm image")
        return False

    # Tag the image with our expected name
    # docker-compose names it based on folder name
    print("   Tagging image as ezlocalai:rocm...")

    # Determine the expected image name based on folder
    folder_name = source_dir.name
    expected_names = [
        f"{folder_name}-ezlocalai:latest",
        "repo-ezlocalai:latest",
        f"{folder_name}_ezlocalai:latest",
    ]

    tagged = False
    for expected_name in expected_names:
        tag_result = subprocess.run(
            ["docker", "tag", expected_name, DOCKER_IMAGE_ROCM],
            capture_output=True,
            check=False,
        )
        if tag_result.returncode == 0:
            tagged = True
            break

    if not tagged:
        print("⚠️  Could not tag image, trying to find it...")
        # List images and try to find one that matches
        list_result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if list_result.returncode == 0:
            for line in list_result.stdout.splitlines():
                if "ezlocalai" in line.lower() and "rocm" not in line:
                    subprocess.run(
                        ["docker", "tag", line.strip(), DOCKER_IMAGE_ROCM],
                        check=False,
                    )
                    tagged = True
                    break

    print("✅ ROCm image built successfully")
    return True


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


def check_prerequisites() -> tuple[bool, str]:
    """
    Check and install prerequisites.

    Returns:
        Tuple of (docker_available, gpu_type) where gpu_type is 'nvidia', 'amd', or 'cpu'
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

    # Check for NVIDIA GPU first
    gpu_type = "cpu"
    if has_nvidia_gpu():
        gpu_info = get_nvidia_gpu_info()
        print(f"✅ NVIDIA GPU detected: {gpu_info}")

        if has_nvidia_container_toolkit():
            print("✅ NVIDIA Container Toolkit is installed")
            gpu_type = "nvidia"
        else:
            if system == "linux":
                if install_nvidia_container_toolkit():
                    if has_nvidia_container_toolkit():
                        gpu_type = "nvidia"
                else:
                    print("⚠️  Running on CPU (GPU acceleration not available)")
            else:
                print("⚠️  NVIDIA Container Toolkit not detected.")
                print("   For GPU acceleration, install it from:")
                print(
                    "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
                )
                print("   Continuing with CPU mode...")
    # Check for AMD GPU if no NVIDIA GPU
    elif has_amd_gpu():
        gpu_info = get_amd_gpu_info()
        print(f"✅ AMD GPU detected: {gpu_info}")

        if has_rocm_support():
            print("✅ ROCm is installed and functional")
            gpu_type = "amd"
        else:
            print("⚠️  ROCm not detected or not functional.")
            if system == "linux":
                print("   For AMD GPU acceleration, install ROCm:")
                print(
                    "   https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
                )
            print("   Continuing with CPU mode...")
    else:
        print("ℹ️  No GPU detected, running on CPU")

    return True, gpu_type


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
        "DEFAULT_MODEL": "unsloth/Qwen3-4B-Instruct-2507-GGUF,unsloth/Qwen3-VL-4B-Instruct-GGUF",
        "WHISPER_MODEL": "large-v3",
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
    gpu_type: str = "cpu",
) -> None:
    """Start the ezlocalai container."""

    # Check if already running
    if is_container_running():
        print(f"✅ ezlocalai is already running!")
        print(f"   API: http://localhost:{DEFAULT_PORT}")
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

    # Select image based on GPU type
    if gpu_type == "nvidia":
        image = DOCKER_IMAGE_CUDA
    elif gpu_type == "amd":
        image = DOCKER_IMAGE_ROCM
    else:
        image = DOCKER_IMAGE

    # Build docker run command
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        CONTAINER_NAME,
        "-p",
        f"{DEFAULT_PORT}:{DEFAULT_PORT}",
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

    # Add GPU-specific flags
    if gpu_type == "nvidia":
        cmd.extend(["--gpus", "all"])
    elif gpu_type == "amd":
        # ROCm requires access to specific devices
        cmd.extend(
            [
                "--device=/dev/kfd",
                "--device=/dev/dri",
                "--group-add",
                "video",
                "--group-add",
                "render",
                "--security-opt",
                "seccomp=unconfined",
            ]
        )

    # Add environment variables
    for key, value in env_vars.items():
        if value:  # Only add non-empty values
            cmd.extend(["-e", f"{key}={value}"])

    # Add ROCm-specific environment variables
    if gpu_type == "amd":
        # HSA_OVERRIDE_GFX_VERSION helps with newer APUs like Radeon 880M
        if "HSA_OVERRIDE_GFX_VERSION" not in env_vars:
            cmd.extend(["-e", "HSA_OVERRIDE_GFX_VERSION=11.0.0"])

    # Add image
    cmd.append(image)

    # Handle image: pull for CPU, build for CUDA/ROCm
    if gpu_type == "nvidia":
        # CUDA image must be built locally (too large for DockerHub)
        if not cuda_image_exists():
            print("\n🔨 CUDA image not found, building from source...")
            if not build_cuda_image():
                print("❌ Failed to build CUDA image. Falling back to CPU mode.")
                gpu_type = "cpu"
                image = DOCKER_IMAGE
                cmd[-1] = image  # Update image in command
                # Remove --gpus flag
                if "--gpus" in cmd:
                    idx = cmd.index("--gpus")
                    cmd.pop(idx)  # Remove --gpus
                    cmd.pop(idx)  # Remove "all"
    elif gpu_type == "amd":
        # ROCm image must be built locally
        if not rocm_image_exists():
            print("\n🔨 ROCm image not found, building from source...")
            if not build_rocm_image():
                print("❌ Failed to build ROCm image. Falling back to CPU mode.")
                gpu_type = "cpu"
                image = DOCKER_IMAGE
                cmd[-1] = image  # Update image in command
                # Remove ROCm-specific flags
                rocm_flags = [
                    "--device=/dev/kfd",
                    "--device=/dev/dri",
                    "--group-add",
                    "video",
                    "--group-add",
                    "render",
                    "--security-opt",
                    "seccomp=unconfined",
                ]
                for flag in rocm_flags:
                    if flag in cmd:
                        cmd.remove(flag)
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
    mode_names = {"nvidia": "NVIDIA GPU", "amd": "AMD GPU", "cpu": "CPU"}
    mode = mode_names.get(gpu_type, "CPU")
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


def stop_container() -> None:
    """Stop and remove the ezlocalai container."""
    if not is_container_running():
        # Check if stopped container exists and remove it
        status = get_container_status()
        if status:
            print("🧹 Removing stopped container...")
            subprocess.run(
                ["docker", "rm", "-f", CONTAINER_NAME],
                capture_output=True,
                check=False,
            )
            print("✅ Container removed")
        else:
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
        # Also remove the container to clean up
        subprocess.run(
            ["docker", "rm", "-f", CONTAINER_NAME],
            capture_output=True,
            check=False,
        )
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
    models = env_vars.get("DEFAULT_MODEL", None)
    if not models:
        # Get models from the list
        env = get_default_env()
        models = env.get("DEFAULT_MODEL", "unsloth/Qwen3-4B-Instruct-2507-GGUF")
    configured = [m.strip() for m in models.split(",")]
    print(f"      LLM models:")
    for m in configured:
        print(f"        - {m}")

    # Whisper
    whisper = env_vars.get("WHISPER_MODEL", "large-v3")
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
    """Pull latest CPU image and rebuild CUDA/ROCm image."""
    print("📦 Updating ezlocalai...")

    # Check if we're in the ezlocalai source folder
    local_source = get_ezlocalai_source_dir()
    if local_source:
        print(f"ℹ️  Running from ezlocalai source folder: {local_source}")

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

    # Build CUDA image from source if NVIDIA GPU available
    if has_nvidia_gpu():
        print(f"\n🔨 Building CUDA image from source...")
        if build_cuda_image():
            print("   ✅ CUDA image rebuilt")
        else:
            print("   ⚠️  Failed to build CUDA image")
    else:
        print("\nℹ️  No NVIDIA GPU detected, skipping CUDA image build")

    # Build ROCm image from source if AMD GPU available
    if has_amd_gpu():
        print(f"\n🔨 Building ROCm image from source...")
        if build_rocm_image():
            print("   ✅ ROCm image rebuilt")
        else:
            print("   ⚠️  Failed to build ROCm image")
    else:
        print("\nℹ️  No AMD GPU detected, skipping ROCm image build")

    print("\n✅ Update complete!")
    print("   Run 'ezlocalai restart' to use the new version.")


def encode_image_to_base64(image_path: str) -> str:
    """Encode a local image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """Get the MIME type based on file extension."""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_types.get(ext, "image/jpeg")


def get_video_mime_type(video_path: str) -> str:
    """Get the MIME type based on video file extension."""
    ext = Path(video_path).suffix.lower()
    mime_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".m4v": "video/mp4",
        ".wmv": "video/x-ms-wmv",
        ".flv": "video/x-flv",
    }
    return mime_types.get(ext, "video/mp4")


def is_video_file(path: str) -> bool:
    """Check if the path is a video file based on extension."""
    ext = Path(path).suffix.lower()
    video_extensions = {".mp4", ".webm", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}
    return ext in video_extensions


def is_url(path: str) -> bool:
    """Check if the path is a URL."""
    return path.startswith("http://") or path.startswith("https://")


def send_prompt(
    prompt_text: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    image_path: Optional[str] = None,
    video_path: Optional[str] = None,
    show_stats: bool = False,
) -> None:
    """Send a prompt to the ezlocalai server and print the response."""
    # Check if server is running
    if not is_container_running():
        print("❌ ezlocalai is not running. Start it with: ezlocalai start")
        sys.exit(1)

    # Load env to get API key and default model if set
    env_vars = load_env_file()
    api_key = env_vars.get("EZLOCALAI_API_KEY", "")

    # Get default model from running server if not specified
    if not model:
        try:
            models_url = f"http://localhost:{DEFAULT_PORT}/v1/models"
            req = urllib.request.Request(models_url)
            with urllib.request.urlopen(req, timeout=10) as response:
                models_data = json.loads(response.read().decode("utf-8"))
                if models_data.get("data"):
                    model = models_data["data"][0].get("id")
        except Exception:
            # Fall back to configured default model
            default_models = env_vars.get("DEFAULT_MODEL", "")
            if default_models:
                model = default_models.split(",")[0].strip()

    # Build the messages array
    content = []

    # Handle image if provided
    if image_path:
        if is_url(image_path):
            # Image URL - use directly
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_path},
                }
            )
        else:
            # Local file - encode to base64
            image_file = Path(image_path)
            if not image_file.exists():
                print(f"❌ Image file not found: {image_path}")
                sys.exit(1)

            mime_type = get_image_mime_type(image_path)
            base64_image = encode_image_to_base64(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                }
            )

    # Handle video if provided
    if video_path:
        if is_url(video_path):
            # Video URL - pass directly, server will handle frame extraction
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": video_path},
                }
            )
            print(f"📹 Processing video from URL...")
        else:
            # Local file - check it exists and pass the path
            # The server will extract frames from the local path
            video_file = Path(video_path)
            if not video_file.exists():
                print(f"❌ Video file not found: {video_path}")
                sys.exit(1)

            # For local files, we encode to base64 data URL
            print(f"📹 Processing video: {video_path}")
            mime_type = get_video_mime_type(video_path)
            base64_video = encode_image_to_base64(
                video_path
            )  # Same function works for any file
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:{mime_type};base64,{base64_video}"},
                }
            )

    # Add the text prompt
    content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content}]

    # Build the request payload
    payload = {
        "messages": messages,
        "stream": True,
    }

    # Add optional parameters
    if model:
        payload["model"] = model
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    # Build headers
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Make the request
    url = f"http://localhost:{DEFAULT_PORT}/v1/chat/completions"

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        start_time = time.time()
        print()  # Start on a new line for streaming output

        # Track streaming statistics
        completion_tokens = 0
        prompt_tokens = 0
        actual_model = model or "unknown"
        full_content = ""

        with urllib.request.urlopen(req, timeout=300) as response:
            # Process Server-Sent Events (SSE) stream
            buffer = ""
            for chunk in iter(lambda: response.read(1024).decode("utf-8"), ""):
                if not chunk:
                    break
                buffer += chunk

                # Process complete SSE messages
                while "\n\n" in buffer or "\r\n\r\n" in buffer:
                    # Split on either \n\n or \r\n\r\n
                    if "\r\n\r\n" in buffer:
                        message, buffer = buffer.split("\r\n\r\n", 1)
                    else:
                        message, buffer = buffer.split("\n\n", 1)

                    # Process each line in the message
                    for line in message.split("\n"):
                        line = line.strip()
                        if line.startswith("data: "):
                            json_data = line[6:]  # Remove "data: " prefix

                            # Check for stream end
                            if json_data == "[DONE]":
                                continue

                            try:
                                chunk_data = json.loads(json_data)

                                # Get model from first chunk
                                if chunk_data.get("model"):
                                    actual_model = chunk_data["model"]

                                # Extract and print delta content
                                choices = chunk_data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        print(content, end="", flush=True)
                                        full_content += content
                                        completion_tokens += (
                                            1  # Approximate token count
                                        )

                                # Get usage from final chunk if available
                                usage = chunk_data.get("usage")
                                if usage:
                                    prompt_tokens = usage.get("prompt_tokens", 0)
                                    completion_tokens = usage.get(
                                        "completion_tokens", completion_tokens
                                    )

                            except json.JSONDecodeError:
                                pass  # Skip malformed JSON

        elapsed = time.time() - start_time
        print()  # End with newline

        if not full_content:
            print("⚠️  Empty response received")

        # Show stats if requested
        if show_stats:
            total_tokens = prompt_tokens + completion_tokens

            print(f"\n{'─' * 50}")
            print(f"📊 Statistics")
            print(f"{'─' * 50}")
            print(f"   Model: {actual_model}")
            print(f"   Prompt tokens: {prompt_tokens:,}")
            print(f"   Completion tokens: {completion_tokens:,}")
            print(f"   Total tokens: {total_tokens:,}")
            print(f"   Total time: {elapsed:.1f}s")

            if completion_tokens > 0:
                overall_speed = completion_tokens / elapsed
                print(f"   Overall speed: {overall_speed:.1f} tok/s")

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        print(f"❌ HTTP Error {e.code}: {e.reason}")
        if error_body:
            try:
                error_json = json.loads(error_body)
                print(f"   {error_json.get('detail', error_body)}")
            except json.JSONDecodeError:
                print(f"   {error_body}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"❌ Connection error: {e.reason}")
        print("   Is ezlocalai running? Check with: ezlocalai status")
        sys.exit(1)
    except TimeoutError:
        print("❌ Request timed out after 300 seconds")
        sys.exit(1)


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
  ezlocalai prompt "Hello, world!"          Send a prompt to the AI
  ezlocalai prompt "What's in this image?" -image ./photo.jpg
  ezlocalai prompt "Describe this video" -video ./clip.mp4

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

    # Prompt command
    prompt_parser = subparsers.add_parser(
        "prompt", help="Send a prompt to the AI and get a response"
    )
    prompt_parser.add_argument(
        "text",
        help="The prompt text to send to the AI",
    )
    prompt_parser.add_argument(
        "-m",
        "--model",
        help="Model to use for the prompt",
        default=None,
    )
    prompt_parser.add_argument(
        "-temp",
        "--temperature",
        type=float,
        help="Temperature for response generation (0.0-2.0)",
        default=None,
    )
    prompt_parser.add_argument(
        "-tp",
        "--top-p",
        type=float,
        help="Top-p (nucleus) sampling parameter (0.0-1.0)",
        default=None,
    )
    prompt_parser.add_argument(
        "-image",
        "--image",
        help="Path to an image file or URL to include with the prompt",
        default=None,
    )
    prompt_parser.add_argument(
        "-video",
        "--video",
        help="Path to a video file or URL to include with the prompt (frames will be extracted)",
        default=None,
    )
    prompt_parser.add_argument(
        "-stats",
        "--stats",
        action="store_true",
        help="Show statistics (tokens, speed, timing) after the response",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Check prerequisites for start/restart
    gpu_type = "cpu"
    if args.command in ("start", "restart"):
        _, gpu_type = check_prerequisites()
        print()

    # Execute command
    if args.command == "start":
        start_container(
            model=args.model,
            uri=args.uri,
            api_key=args.api_key,
            ngrok=args.ngrok,
            gpu_type=gpu_type,
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
            gpu_type=gpu_type,
        )

    elif args.command == "status":
        show_status()

    elif args.command == "update":
        update_images()

    elif args.command == "logs":
        show_logs(follow=args.follow)

    elif args.command == "prompt":
        send_prompt(
            prompt_text=args.text,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            image_path=args.image,
            video_path=args.video,
            show_stats=args.stats,
        )


if __name__ == "__main__":
    main()
