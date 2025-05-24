"""
Main script for LLaVA model.
"""

import os
import argparse
import torch
from PIL import Image

from models.llava import LLaVA
from utils.data_utils import load_image, process_image
from utils.visualization import display_image_with_caption


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA: Large Language and Vision Assistant")
    parser.add_argument("--vision-model", type=str, default="openai/clip-vit-large-patch14-336",
                        help="Vision model name or path")
    parser.add_argument("--language-model", type=str, default="lmsys/vicuna-7b-v1.5",
                        help="Language model name or path")
    parser.add_argument("--image-path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--prompt", type=str, default="What's in this image?",
                        help="Text prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--load-8bit", action="store_true",
                        help="Load language model in 8-bit mode")
    parser.add_argument("--load-4bit", action="store_true",
                        help="Load language model in 4-bit mode")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run the model on")
    parser.add_argument("--save-output", type=str, default=None,
                        help="Path to save the output image with caption")
    parser.add_argument("--display", action="store_true",
                        help="Display the image with caption")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    
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
    print(f"Generating response for image: {args.image_path}")
    print(f"Prompt: {args.prompt}")
    
    response = model.generate_from_image(
        image_path=args.image_path,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    print(f"\nResponse: {response}")
    
    # Display or save the image with caption
    if args.display:
        display_image_with_caption(args.image_path, response)
    
    if args.save_output:
        from utils.visualization import add_caption_to_image
        add_caption_to_image(args.image_path, response, args.save_output)
        print(f"Output image saved to: {args.save_output}")


if __name__ == "__main__":
    main()