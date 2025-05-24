"""
Demo script for LLaVA model.
"""

import os
import sys
import argparse
from PIL import Image
import torch
import gradio as gr

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llava import LLaVA
from utils.visualization import display_image_with_caption


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA Demo")
    parser.add_argument("--vision-model", type=str, default="openai/clip-vit-large-patch14-336",
                        help="Vision model name or path")
    parser.add_argument("--language-model", type=str, default="lmsys/vicuna-7b-v1.5",
                        help="Language model name or path")
    parser.add_argument("--load-8bit", action="store_true", help="Load language model in 8-bit mode")
    parser.add_argument("--load-4bit", action="store_true", help="Load language model in 4-bit mode")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to run the model on (default: cuda if available, otherwise cpu)")
    parser.add_argument("--share", action="store_true", help="Share the Gradio interface")
    return parser.parse_args()


def initialize_model(args):
    """Initialize the LLaVA model."""
    print(f"Initializing LLaVA model...")
    print(f"Vision model: {args.vision_model}")
    print(f"Language model: {args.language_model}")
    
    model = LLaVA(
        vision_model_path=args.vision_model,
        language_model_path=args.language_model,
        device=args.device,
        load_in_8bit=args.load_8bit,
        load_in_4bit=args.load_4bit
    )
    
    print(f"Model initialized on {model.device}")
    return model


def gradio_interface(model):
    """Create a Gradio interface for the model."""
    
    def process_example(image, prompt, temperature, max_tokens):
        if image is None:
            return "Please upload an image."
        
        # Save the image temporarily
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        # Generate response
        response = model.generate_from_image(
            image_path=temp_path,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_tokens
        )
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return response
    
    # Define examples
    examples = [
        [None, "What's in this image?", 0.7, 256],
        [None, "Describe this image in detail.", 0.7, 512],
        [None, "What objects can you see in this image?", 0.7, 256],
        [None, "Is there anything unusual or interesting in this image?", 0.7, 256],
    ]
    
    # Create the interface
    demo = gr.Interface(
        fn=process_example,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
            gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature"),
            gr.Slider(minimum=16, maximum=1024, value=256, step=16, label="Max Tokens")
        ],
        outputs=gr.Textbox(label="Response"),
        title="LLaVA: Large Language and Vision Assistant",
        description="Upload an image and enter a prompt to get a response from the LLaVA model.",
        examples=examples,
        allow_flagging="never"
    )
    
    return demo


def main():
    args = parse_args()
    model = initialize_model(args)
    demo = gradio_interface(model)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()