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
        logger.info(f"Initializing LLaVA model from {MODEL_NAME}")
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("Model initialization complete")
    
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