"""
Configuration settings for the LLaVA implementation.
"""

import os
import torch
from pathlib import Path
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
ASSETS_DIR = PROJECT_ROOT / "assets"
EXAMPLES_DIR = PROJECT_ROOT / "examples"

# Model settings
MODEL_NAME = "liuhaotian/llava-v1.5-7b"
MODEL_REVISION = "main"

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    logger.info("CUDA not available, using CPU")

# Generation settings
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

# API settings
API_HOST = "0.0.0.0"
API_PORT = 7860
API_WORKERS = 1
API_RELOAD = True

# Gradio settings
GRADIO_THEME = "soft"
GRADIO_TITLE = "LLaVA Chat"
GRADIO_DESCRIPTION = """
A powerful multimodal AI assistant that can understand and discuss images.
Upload any image and chat with LLaVA about it!
"""

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"

# Create necessary directories
for directory in [ASSETS_DIR, EXAMPLES_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 