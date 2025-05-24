"""
Gradio interface for the LLaVA model.
"""

import gradio as gr
from PIL import Image
import os
import tempfile
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import traceback
import sys

from ..configs.settings import (
    GRADIO_THEME,
    GRADIO_TITLE,
    GRADIO_DESCRIPTION,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    API_HOST,
    API_PORT,
    API_WORKERS,
    API_RELOAD
)
from ..models.llava_model import LLaVAModel
from ..utils.logging import setup_logging, get_logger

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(title="LLaVA Web Interface")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model = None

def initialize_model():
    global model
    try:
        logger.info("Initializing LLaVA model...")
        # Use a smaller model variant and enable memory optimizations
        model = LLaVAModel(
            vision_model_path="openai/clip-vit-base-patch32",  # Smaller vision model
            language_model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Smaller language model
            device="cpu",  # Force CPU for Hugging Face Spaces
            projection_hidden_dim=2048  # Reduce projection layer size
        )
        
        # Enable memory optimizations
        torch.cuda.empty_cache()  # Clear any cached memory
        if hasattr(model, 'language_model'):
            model.language_model.config.use_cache = False  # Disable KV cache
        
        logger.info(f"Model initialized on {model.device}")
        return True
    except Exception as e:
        error_msg = f"Error initializing model: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        print(error_msg, file=sys.stderr)
        return False

def process_image(
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P
) -> str:
    """
    Process an image with the LLaVA model.
    
    Args:
        image: Input image
        prompt: Text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        str: Model response
    """
    if not model:
        error_msg = "Error: Model not initialized"
        logger.error(error_msg)
        return error_msg
    
    if image is None:
        error_msg = "Error: No image provided"
        logger.error(error_msg)
        return error_msg
    
    if not prompt or not prompt.strip():
        error_msg = "Error: No prompt provided"
        logger.error(error_msg)
        return error_msg
    
    temp_path = None
    try:
        logger.info(f"Processing image with prompt: {prompt[:100]}...")
        
        # Save the uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        # Clear memory before processing
        torch.cuda.empty_cache()
        
        # Generate response with reduced memory usage
        with torch.inference_mode():  # More memory efficient than no_grad
            response = model(
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )

        logger.info("Successfully generated response")
        return response

    except Exception as e:
        error_msg = f"Error processing image: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        print(error_msg, file=sys.stderr)
        return f"Error: {str(e)}"
    
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary file: {str(e)}")
        
        # Clear memory after processing
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error clearing CUDA cache: {str(e)}")

def create_interface() -> gr.Blocks:
    """Create and return the Gradio interface."""
    with gr.Blocks(theme=GRADIO_THEME) as interface:
        gr.Markdown(f"""# {GRADIO_TITLE}

{GRADIO_DESCRIPTION}

## Example Prompts

Try these prompts to get started:
- "What can you see in this image?"
- "Describe this scene in detail"
- "What emotions does this image convey?"
- "What's happening in this picture?"
- "Can you identify any objects or people in this image?"

## Usage Instructions

1. Upload an image using the image uploader
2. Enter your prompt in the text box
3. (Optional) Adjust the generation parameters
4. Click "Generate Response" to get LLaVA's analysis
""")
        
        with gr.Row():
            with gr.Column():
                # Input components
                image_input = gr.Image(type="pil", label="Upload Image")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="What can you see in this image?",
                    lines=3
                )
                
                with gr.Accordion("Generation Parameters", open=False):
                    max_tokens = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        value=DEFAULT_MAX_NEW_TOKENS,
                        step=64,
                        label="Max New Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=DEFAULT_TEMPERATURE,
                        step=0.1,
                        label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=DEFAULT_TOP_P,
                        step=0.1,
                        label="Top P"
                    )
                
                generate_btn = gr.Button("Generate Response", variant="primary")
            
            with gr.Column():
                # Output component
                output = gr.Textbox(
                    label="Response",
                    lines=10,
                    show_copy_button=True
                )
        
        # Set up event handlers with explicit types
        generate_btn.click(
            fn=process_image,
            inputs=[
                image_input,
                prompt_input,
                max_tokens,
                temperature,
                top_p
            ],
            outputs=output,
            api_name="process_image"
        )
    
    return interface

# Create Gradio app
demo = create_interface()

# Mount Gradio app
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    """Run the FastAPI application."""
    import uvicorn
    
    # Initialize model
    if not initialize_model():
        logger.error("Failed to initialize model. Exiting...")
        sys.exit(1)
    
    # Start the server
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        workers=API_WORKERS,
        reload=API_RELOAD,
        log_level="info"
    )

if __name__ == "__main__":
    main() 