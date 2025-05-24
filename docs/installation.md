# Installation Guide

This guide provides detailed instructions for installing and setting up the LLaVA implementation.

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for inference)
- Git

## Basic Installation

1. Clone the repository:
```bash
git clone https://github.com/Prashant-ambati/llava-implementation.git
cd llava-implementation
```

2. Create a virtual environment:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n llava python=3.10
conda activate llava
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## GPU Acceleration

For GPU acceleration, make sure you have the appropriate CUDA version installed. You can check your CUDA version with:

```bash
nvidia-smi
```

If you need to install a specific version of PyTorch compatible with your CUDA version, visit the [PyTorch installation page](https://pytorch.org/get-started/locally/).

## Memory-Efficient Installation

If you have limited GPU memory, you can use quantization to reduce memory usage. Install the additional dependencies:

```bash
pip install bitsandbytes
```

When running the model, use the `--load-8bit` or `--load-4bit` flags to enable quantization.

## Troubleshooting

### Common Issues

1. **CUDA out of memory error**:
   - Try using a smaller model (e.g., 7B instead of 13B)
   - Enable quantization with `--load-8bit` or `--load-4bit`
   - Reduce batch size or input resolution

2. **Missing dependencies**:
   - Make sure you've installed all dependencies with `pip install -r requirements.txt`
   - Some packages might require system libraries; check the error message for details

3. **Model download issues**:
   - Ensure you have a stable internet connection
   - Try using a VPN if you're having issues accessing Hugging Face models
   - Consider downloading the models manually and specifying the local path

### Getting Help

If you encounter any issues not covered here, please [open an issue](https://github.com/Prashant-ambati/llava-implementation/issues) on GitHub.