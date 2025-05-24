"""
Gradio interface for the LLaVA model.
"""

import gradio as gr
from PIL import Image

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

# Initialize model
model = LLaVAModel()

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
    try:
        logger.info(f"Processing image with prompt: {prompt[:100]}...")
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
        logger.error(f"Error processing image: {str(e)}")
        return f"Error: {str(e)}"

def create_interface() -> gr.Interface:
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
        
        # Set up event handlers
        generate_btn.click(
            fn=process_image,
            inputs=[
                image_input,
                prompt_input,
                max_tokens,
                temperature,
                top_p
            ],
            outputs=output
        )
    
    return interface

def main():
    """Run the Gradio interface."""
    interface = create_interface()
    interface.launch(
        server_name=API_HOST,
        server_port=API_PORT,
        share=True,
        show_error=True,
        show_api=False
    )

if __name__ == "__main__":
    main() 