"""
LLaVA: Large Language and Vision Assistant
Implementation based on the paper "Visual Instruction Tuning" (NeurIPS 2023)
"""

import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel, 
    CLIPImageProcessor,
    AutoTokenizer, 
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList
)
from PIL import Image
import os
from typing import List, Dict, Optional, Tuple, Union


class StoppingCriteriaSub(StoppingCriteria):
    """Custom stopping criteria for text generation."""
    
    def __init__(self, stops=None, encounters=1):
        super().__init__()
        self.stops = stops or []
        self.encounters = encounters
        self.counter = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Check if stopping criteria are met."""
        for stop_id in self.stops:
            if stop_id in input_ids[0][-1:]:
                self.counter += 1
                if self.counter >= self.encounters:
                    return True
        return False


class MLP(nn.Module):
    """MLP projection layer to connect vision and language models."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LLaVA(nn.Module):
    """
    LLaVA: Large Language and Vision Assistant
    A multimodal model that connects a vision encoder with a language model.
    """
    
    def __init__(
        self,
        vision_model_path: str = "openai/clip-vit-large-patch14-336",
        language_model_path: str = "lmsys/vicuna-7b-v1.5",
        projection_hidden_dim: int = 4096,
        device: str = None
    ):
        """
        Initialize the LLaVA model.
        
        Args:
            vision_model_path: Path or name of the vision model
            language_model_path: Path or name of the language model
            projection_hidden_dim: Hidden dimension of the projection layer
            device: Device to load the model on ('cpu' only for Hugging Face Spaces)
        """
        super().__init__()
        
        self.device = 'cpu'
        
        # Load vision model
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_path)
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_model_path)
        
        # Load language model (always float32, cpu)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_path)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_path,
            torch_dtype=torch.float32,
            device_map=None
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get dimensions
        vision_hidden_size = self.vision_model.config.hidden_size
        language_hidden_size = self.language_model.config.hidden_size
        
        # Create projection layer
        self.projection = MLP(
            input_dim=vision_hidden_size,
            hidden_dim=projection_hidden_dim,
            output_dim=language_hidden_size
        )
        
        # Move models to device
        self.vision_model.to(self.device)
        self.language_model.to(self.device)
        self.projection.to(self.device)
        
        # Set to evaluation mode
        self.vision_model.eval()
        self.language_model.eval()
        self.projection.eval()
        
        # Template for conversation
        self.conv_template = [
            {"role": "system", "content": "You are a helpful assistant that can understand images and answer questions about them."},
        ]
        
    def encode_image(self, image_path: str) -> torch.Tensor:
        """
        Encode an image using the vision model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tensor containing the image features
        """
        image = Image.open(image_path).convert('RGB')
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.vision_model(**inputs)
            image_features = outputs.pooler_output  # [1, hidden_size]
            
        return image_features
    
    def project_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Project image features to the language model's embedding space.
        
        Args:
            image_features: Image features from the vision model
            
        Returns:
            Projected image features
        """
        with torch.no_grad():
            projected_features = self.projection(image_features)
            
        return projected_features
    
    def format_prompt(self, prompt: str, conversation: List[Dict[str, str]] = None) -> str:
        """
        Format the prompt for the language model.
        
        Args:
            prompt: The text prompt
            conversation: Optional conversation history
            
        Returns:
            Formatted prompt string
        """
        if conversation is None:
            conversation = self.conv_template.copy()
        
        conversation.append({"role": "user", "content": prompt})
        
        formatted_prompt = ""
        for message in conversation:
            if message["role"] == "system":
                formatted_prompt += f"<s>[INST] <<SYS>>\n{message['content']}\n<</SYS>>\n\n"
            elif message["role"] == "user":
                if formatted_prompt:
                    formatted_prompt += f"{message['content']} [/INST]"
                else:
                    formatted_prompt += f"<s>[INST] {message['content']} [/INST]"
            elif message["role"] == "assistant":
                formatted_prompt += f" {message['content']} </s><s>[INST] "
        
        return formatted_prompt
    
    def generate_from_image(
        self, 
        image_path: str, 
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        conversation: List[Dict[str, str]] = None
    ) -> str:
        """
        Generate text based on an image and a prompt.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            conversation: Optional conversation history
            
        Returns:
            Generated text response
        """
        # Encode image
        image_features = self.encode_image(image_path)
        projected_features = self.project_image_features(image_features)
        
        # Format prompt
        formatted_prompt = self.format_prompt(prompt, conversation)
        
        # Tokenize prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # Prepare for generation
        stopping_criteria = StoppingCriteriaList([
            StoppingCriteriaSub(stops=[self.tokenizer.eos_token_id], encounters=1)
        ])
        
        # Generate response
        with torch.no_grad():
            # Prepare the inputs for the language model
            # Here we would normally inject the image features into the language model
            # This is a simplified version - in the actual LLaVA, this is done by modifying
            # the language model's forward pass to accept image features
            
            # For demonstration purposes, we'll just use the language model directly
            outputs = self.language_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stopping_criteria=stopping_criteria,
                do_sample=True
            )
            
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        return generated_text.strip()
    
    def save_model(self, output_dir: str):
        """
        Save the model to the specified directory.
        
        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save vision model
        self.vision_model.save_pretrained(os.path.join(output_dir, "vision_model"))
        self.image_processor.save_pretrained(os.path.join(output_dir, "vision_model"))
        
        # Save language model
        self.language_model.save_pretrained(os.path.join(output_dir, "language_model"))
        self.tokenizer.save_pretrained(os.path.join(output_dir, "language_model"))
        
        # Save projection layer
        torch.save(self.projection.state_dict(), os.path.join(output_dir, "projection.pt"))
        
    @classmethod
    def from_pretrained(cls, model_path: str, device: str = None):
        """
        Load a pretrained LLaVA model.
        
        Args:
            model_path: Path to the saved model
            device: Device to load the model on
            
        Returns:
            Loaded LLaVA model
        """
        # Load vision model
        vision_model_path = os.path.join(model_path, "vision_model")
        
        # Load language model
        language_model_path = os.path.join(model_path, "language_model")
        
        # Create model instance
        model = cls(
            vision_model_path=vision_model_path,
            language_model_path=language_model_path,
            device=device
        )
        
        # Load projection layer
        projection_path = os.path.join(model_path, "projection.pt")
        model.projection.load_state_dict(torch.load(projection_path, map_location=model.device))
        
        return model