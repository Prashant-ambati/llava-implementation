"""
Utility functions for data processing in LLaVA.
"""

import os
import json
import torch
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from transformers import CLIPImageProcessor


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")


def process_image(
    image: Union[str, Image.Image],
    image_processor: CLIPImageProcessor,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Process an image for input to the vision model.
    
    Args:
        image: PIL Image object or path to image file
        image_processor: CLIP image processor
        device: Device to load the processed image on
        
    Returns:
        Processed image tensor
    """
    if isinstance(image, str):
        image = load_image(image)
    
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    return inputs


def pad_image(image: Image.Image, target_size: Tuple[int, int] = (336, 336)) -> Image.Image:
    """
    Pad an image to the target size while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        target_size: Target size (width, height)
        
    Returns:
        Padded image
    """
    width, height = image.size
    target_width, target_height = target_size
    
    # Calculate padding
    ratio = min(target_width / width, target_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # Resize image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create new image with padding
    new_image = Image.new("RGB", target_size, (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))
    
    return new_image


def load_conversation_data(json_path: str) -> List[Dict]:
    """
    Load conversation data from a JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        List of conversation dictionaries
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise ValueError(f"Error loading JSON data: {e}")


def format_conversation(
    conversation: List[Dict[str, str]],
    system_prompt: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Format a conversation for the LLaVA model.
    
    Args:
        conversation: List of conversation messages
        system_prompt: Optional system prompt to prepend
        
    Returns:
        Formatted conversation
    """
    formatted_conv = []
    
    # Add system prompt if provided
    if system_prompt:
        formatted_conv.append({"role": "system", "content": system_prompt})
    
    # Add conversation messages
    for message in conversation:
        if "role" in message and "content" in message:
            formatted_conv.append({
                "role": message["role"],
                "content": message["content"]
            })
    
    return formatted_conv


def create_image_text_pair(
    image_path: str,
    text: str,
    image_processor: CLIPImageProcessor,
    tokenizer,
    max_length: int = 512,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create an image-text pair for training or inference.
    
    Args:
        image_path: Path to the image file
        text: Text prompt
        image_processor: CLIP image processor
        tokenizer: Language model tokenizer
        max_length: Maximum text length
        device: Device to load tensors on
        
    Returns:
        Tuple of (image_tensor, text_tensor)
    """
    # Process image
    image = load_image(image_path)
    image_inputs = image_processor(images=image, return_tensors="pt").to(device)
    
    # Process text
    text_inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    ).to(device)
    
    return image_inputs, text_inputs