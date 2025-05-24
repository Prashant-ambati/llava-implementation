"""
LLaVA model implementation.
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

from ..configs.settings import MODEL_NAME, MODEL_REVISION, DEVICE
from ..utils.logging import get_logger

logger = get_logger(__name__)

class LLaVAModel:
    """LLaVA model wrapper class."""
    
    def __init__(self):
        """Initialize the LLaVA model and processor."""
        try:
            logger.info(f"Initializing LLaVA model from {MODEL_NAME}")
            logger.info(f"Using device: {DEVICE}")
            
            # Initialize processor
            self.processor = AutoProcessor.from_pretrained(
                MODEL_NAME,
                revision=MODEL_REVISION,
                trust_remote_code=True
            )
            
            # Set model dtype based on device
            model_dtype = torch.float32 if DEVICE == "cpu" else torch.float16
            
            # Initialize model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                revision=MODEL_REVISION,
                torch_dtype=model_dtype,
                device_map="auto" if DEVICE == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Move model to device if not using device_map
            if DEVICE == "cpu":
                self.model = self.model.to(DEVICE)
            
            logger.info("Model initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def generate_response(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response for the given image and prompt.
        
        Args:
            image: Input image as PIL Image
            prompt: Text prompt for the model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            str: Generated response
        """
        try:
            # Prepare inputs
            inputs = self.processor(
                prompt,
                image,
                return_tensors="pt"
            ).to(DEVICE)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
            
            # Decode and return response
            response = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            logger.debug(f"Generated response: {response[:100]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def __call__(self, *args, **kwargs):
        """Convenience method to call generate_response."""
        return self.generate_response(*args, **kwargs) 