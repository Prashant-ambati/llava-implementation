"""
Test script for LLaVA model.
"""

import os
import sys
import argparse
import torch
from PIL import Image
import requests
from io import BytesIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llava import LLaVA
from utils.visualization import display_image_with_caption


def parse_args():
    parser = argparse.ArgumentParser(description="Test LLaVA Model")
    parser.add_argument("--vision-model", type=str, default="openai/clip-vit-large-patch14-336",
                        help="Vision model name or path")
    parser.add_argument("--language-model", type=str, default="lmsys/vicuna-7b-v1.5",
                        help="Language model name or path")
    parser.add_argument("--image-url", type=str, 
                        default="https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg",
                        help="URL of the image to test")
    parser.add_argument("--prompt", type=str, default="What's in this image?",
                        help="Text prompt")
    parser.add_argument("--load-8bit", action="store_true",
                        help="Load language model in 8-bit mode")
    parser.add_argument("--load-4bit", action="store_true",
                        help="Load language model in 4-bit mode")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run the model on")
    return parser.parse_args()


def download_image(url):
    """Download an image from a URL."""
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def save_image(image, filename):
    """Save an image to a file."""
    image.save(filename)
    return filename


def main():
    args = parse_args()
    
    # Download and save the image
    print(f"Downloading image from {args.image_url}...")
    image = download_image(args.image_url)
    filename = "temp_test_image.jpg"
    save_image(image, filename)
    
    # Initialize model
    print(f"Initializing LLaVA model...")
    model = LLaVA(
        vision_model_path=args.vision_model,
        language_model_path=args.language_model,
        device=args.device,
        load_in_8bit=args.load_8bit,
        load_in_4bit=args.load_4bit
    )
    
    print(f"Model initialized on {model.device}")
    
    # Generate response
    print(f"Generating response for prompt: {args.prompt}")
    
    response = model.generate_from_image(
        image_path=filename,
        prompt=args.prompt,
        max_new_tokens=256,
        temperature=0.7
    )
    
    print(f"\nResponse: {response}")
    
    # Display the image with caption
    display_image_with_caption(filename, response)
    
    # Clean up
    if os.path.exists(filename):
        os.remove(filename)


if __name__ == "__main__":
    main()