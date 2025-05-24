# LLaVA Architecture

This document describes the architecture of the LLaVA (Large Language and Vision Assistant) model.

## Overview

LLaVA is a multimodal model that connects a vision encoder with a large language model for general-purpose visual and language understanding. The model is based on the paper ["Visual Instruction Tuning"](https://arxiv.org/abs/2304.08485) by Liu et al., which was presented at NeurIPS 2023.

## Components

The LLaVA architecture consists of three main components:

1. **Vision Encoder**: CLIP ViT-L/14 (336px)
2. **Projection Layer**: A two-layer MLP
3. **Language Model**: Vicuna (7B or 13B)

### Vision Encoder

The vision encoder is responsible for processing images and extracting visual features. LLaVA uses the CLIP ViT-L/14 model with a resolution of 336x336 pixels. This model was pretrained on a large dataset of image-text pairs and has strong zero-shot capabilities.

Key specifications:
- Model: CLIP ViT-L/14
- Resolution: 336x336 pixels
- Hidden size: 1024
- Output: Image embeddings of shape [batch_size, 1, hidden_size]

### Projection Layer

The projection layer connects the vision encoder to the language model. It transforms the visual features from the vision encoder's embedding space to the language model's embedding space.

Key specifications:
- Type: Two-layer MLP with GELU activation
- Input dimension: 1024 (vision encoder hidden size)
- Hidden dimension: 4096
- Output dimension: 4096 (language model hidden size for 7B) or 5120 (for 13B)
- Activation: GELU
- Dropout: 0.1

### Language Model

The language model is responsible for generating text based on the combined visual and textual inputs. LLaVA uses Vicuna, an instruction-tuned chatbot based on LLaMA.

Key specifications:
- Model: Vicuna (7B or 13B)
- Context length: 2048 tokens
- Architecture: Decoder-only transformer

## Data Flow

1. An image is processed by the vision encoder to extract visual features.
2. The visual features are transformed by the projection layer to match the language model's embedding space.
3. The projected visual features are prepended to the text embeddings.
4. The combined embeddings are processed by the language model to generate a response.

## Training Process

LLaVA is trained in two stages:

1. **Feature Alignment Stage**: Only the projection layer is updated, while the vision encoder and language model are kept frozen. This stage aligns the visual features with the language model's embedding space.

2. **Visual Instruction Tuning Stage**: Both the projection layer and language model are updated. This stage teaches the model to follow multimodal instructions.

## Model Variants

The implementation supports different model sizes:

1. **LLaVA-7B**: Uses Vicuna-7B as the language model
   - Language model hidden size: 4096
   - Projection output dimension: 4096

2. **LLaVA-13B**: Uses Vicuna-13B as the language model
   - Language model hidden size: 5120
   - Projection output dimension: 5120

## Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Vision Encoder │────▶│ Projection Layer│────▶│ Language Model  │────▶ Text Output
│  (CLIP ViT-L/14)│     │   (MLP 2-layer) │     │    (Vicuna)     │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ▲                                               ▲
        │                                               │
        │                                               │
    Image Input                                    Text Input
```