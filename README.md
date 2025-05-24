# LLaVA Implementation: Large Language and Vision Assistant

This project implements the architecture from the research paper "Visual Instruction Tuning" (NeurIPS 2023, Oral Presentation) by Liu et al., which introduces LLaVA - a multimodal model that combines vision and language capabilities.

## Overview

LLaVA (Large Language and Vision Assistant) is a state-of-the-art multimodal model that connects a vision encoder with a large language model for general-purpose visual and language understanding. This implementation focuses on the core architecture and demonstrates how to:

1. Connect a CLIP ViT-L/14 vision encoder to a language model using a projection layer
2. Process images and text together for multimodal understanding
3. Generate text responses based on visual inputs

## Architecture

The LLaVA architecture consists of three main components:
- **Vision Encoder**: CLIP ViT-L/14 (336px) for processing images
- **Projection Layer**: A two-layer MLP that connects the vision encoder to the language model
- **Language Model**: A large language model (e.g., Vicuna) for text generation

## Features

- Image and text processing in a unified framework
- Visual question answering capabilities
- Image captioning and description
- Visual reasoning

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llava_implementation.git
cd llava_implementation

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from models.llava import LLaVA

# Initialize the model
model = LLaVA(
    vision_model_path="openai/clip-vit-large-patch14-336",
    language_model_path="lmsys/vicuna-7b-v1.5",
    device="cuda"  # or "cpu"
)

# Process an image and generate a response
response = model.generate_from_image("path/to/image.jpg", "What can you see in this image?")
print(response)
```

## Project Structure

```
llava_implementation/
├── configs/            # Configuration files
├── data/               # Data handling utilities
├── models/             # Model implementation
├── notebooks/          # Jupyter notebooks for demonstrations
├── scripts/            # Training and evaluation scripts
├── utils/              # Utility functions
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

## References

- [Visual Instruction Tuning (Liu et al., 2023)](https://arxiv.org/abs/2304.08485)
- [Improved Baselines with Visual Instruction Tuning (Liu et al., 2023)](https://arxiv.org/abs/2310.03744)
- [LLaVA GitHub Repository](https://github.com/haotian-liu/LLaVA)

## License

This project is licensed under the MIT License - see the LICENSE file for details.