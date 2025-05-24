"""
Visualization utilities for LLaVA.
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from typing import List, Dict, Tuple, Optional, Union
import cv2


def display_image_with_caption(
    image_path: str,
    caption: str,
    figsize: Tuple[int, int] = (10, 10)
) -> None:
    """
    Display an image with a caption.
    
    Args:
        image_path: Path to the image file
        caption: Caption text
        figsize: Figure size
    """
    image = Image.open(image_path).convert('RGB')
    
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.title(caption)
    plt.tight_layout()
    plt.show()


def visualize_attention(
    image_path: str,
    attention_weights: torch.Tensor,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Visualize attention weights on an image.
    
    Args:
        image_path: Path to the image file
        attention_weights: Attention weights tensor
        figsize: Figure size
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Normalize attention weights
    if attention_weights.dim() > 2:
        # Average across heads and layers if necessary
        attention_weights = attention_weights.mean(dim=(0, 1))
    
    attention_weights = attention_weights.detach().cpu().numpy()
    attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
    
    # Resize attention map to image size
    attention_map = cv2.resize(attention_weights, (image.width, image.height))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image
    alpha = 0.5
    overlay = heatmap * alpha + image_np * (1 - alpha)
    overlay = overlay.astype(np.uint8)
    
    # Display original image and attention overlay
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_comparison_grid(
    image_path: str,
    responses: List[Dict[str, str]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Create a comparison grid of different model responses.
    
    Args:
        image_path: Path to the image file
        responses: List of dictionaries with 'model' and 'response' keys
        output_path: Optional path to save the figure
        figsize: Figure size
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Add image
    ax1 = plt.subplot2grid((len(responses) + 1, 3), (0, 0), colspan=3)
    ax1.imshow(image)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # Add responses
    for i, resp in enumerate(responses):
        ax = plt.subplot2grid((len(responses) + 1, 3), (i + 1, 0), colspan=3)
        ax.text(0.5, 0.5, f"{resp['model']}: {resp['response']}", 
                wrap=True, horizontalalignment='center', 
                verticalalignment='center', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    plt.show()


def add_caption_to_image(
    image_path: str,
    caption: str,
    output_path: str,
    font_size: int = 20,
    font_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0)
) -> None:
    """
    Add a caption to an image and save it.
    
    Args:
        image_path: Path to the input image
        caption: Caption text
        output_path: Path to save the output image
        font_size: Font size
        font_color: Font color (RGB)
        bg_color: Background color (RGB)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Create a new image with space for the caption
    caption_height = font_size + 20  # Add some padding
    new_image = Image.new('RGB', (image.width, image.height + caption_height), bg_color)
    new_image.paste(image, (0, 0))
    
    # Add caption
    draw = ImageDraw.Draw(new_image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text position
    text_width = draw.textlength(caption, font=font)
    text_position = ((image.width - text_width) // 2, image.height + 10)
    
    # Draw text
    draw.text(text_position, caption, font=font, fill=font_color)
    
    # Save image
    new_image.save(output_path)