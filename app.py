from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
from typing import Optional
from pydantic import BaseModel
import torch
import gradio as gr
from models.llava import LLaVA

# Initialize model globally
model = None

def initialize_model():
    global model
    try:
        model = LLaVA(
            vision_model_path="openai/clip-vit-large-patch14-336",
            language_model_path="lmsys/vicuna-7b-v1.5",
            device="cuda" if torch.cuda.is_available() else "cpu",
            load_in_8bit=True
        )
        print(f"Model initialized on {model.device}")
        return True
    except Exception as e:
        print(f"Error initializing model: {e}")
        return False

def process_image(image, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    if not model:
        return "Error: Model not initialized"
    
    try:
        # Save the uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name

        # Generate response
        response = model.generate_from_image(
            image_path=temp_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )

        # Clean up temporary file
        os.unlink(temp_path)
        return response

    except Exception as e:
        return f"Error processing image: {str(e)}"

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="LLaVA Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # LLaVA Chat
        Upload an image and chat with LLaVA about it. This model can understand and describe images, answer questions about them, and engage in visual conversations.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Image")
                prompt_input = gr.Textbox(
                    label="Ask about the image",
                    placeholder="What can you see in this image?",
                    lines=3
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    max_tokens = gr.Slider(
                        minimum=32,
                        maximum=512,
                        value=256,
                        step=32,
                        label="Max New Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        label="Top P"
                    )
                
                submit_btn = gr.Button("Generate Response", variant="primary")
            
            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="Model Response",
                    lines=10,
                    show_copy_button=True
                )
        
        # Set up the submit action
        submit_btn.click(
            fn=process_image,
            inputs=[image_input, prompt_input, max_tokens, temperature, top_p],
            outputs=output
        )
        
        # Add examples
        gr.Examples(
            examples=[
                ["examples/cat.jpg", "What can you see in this image?"],
                ["examples/landscape.jpg", "Describe this scene in detail."],
                ["examples/food.jpg", "What kind of food is this and how would you describe it?"]
            ],
            inputs=[image_input, prompt_input]
        )
    
    return demo

# Create FastAPI app
app = FastAPI(title="LLaVA Web Interface")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Gradio app
demo = create_interface()

# Mount Gradio app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # Initialize model
    if initialize_model():
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=7860)  # Hugging Face Spaces uses port 7860
    else:
        print("Failed to initialize model. Exiting...") 