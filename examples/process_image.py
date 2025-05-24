"""
Example script for processing images with the LLaVA model.
"""

import argparse
from pathlib import Path
from PIL import Image

from src.models.llava_model import LLaVAModel
from src.configs.settings import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from src.utils.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)

def process_image(
    image_path: str,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P
) -> str:
    """
    Process an image with the LLaVA model.
    
    Args:
        image_path: Path to the input image
        prompt: Text prompt for the model
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        str: Model response
    """
    try:
        # Load image
        image = Image.open(image_path)
        logger.info(f"Loaded image from {image_path}")
        
        # Initialize model
        model = LLaVAModel()
        logger.info("Model initialized")
        
        # Generate response
        response = model(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        logger.info("Generated response")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

def main():
    """Main function to process images from command line."""
    parser = argparse.ArgumentParser(description="Process images with LLaVA model")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("prompt", type=str, help="Text prompt for the model")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                      help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                      help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P,
                      help="Top-p sampling parameter")
    parser.add_argument("--output", type=str, help="Path to save the response")
    
    args = parser.parse_args()
    
    try:
        # Process image
        response = process_image(
            image_path=args.image_path,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # Print or save response
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(response)
            logger.info(f"Saved response to {output_path}")
        else:
            print("\nModel Response:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 