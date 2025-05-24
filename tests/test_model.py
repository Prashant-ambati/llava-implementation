"""
Tests for the LLaVA model implementation.
"""

import pytest
from PIL import Image
import torch

from src.models.llava_model import LLaVAModel
from src.configs.settings import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P

@pytest.fixture
def model():
    """Fixture to provide a model instance."""
    return LLaVAModel()

@pytest.fixture
def sample_image():
    """Fixture to provide a sample image."""
    # Create a simple test image
    return Image.new('RGB', (224, 224), color='red')

def test_model_initialization(model):
    """Test that the model initializes correctly."""
    assert model is not None
    assert model.processor is not None
    assert model.model is not None

def test_model_device(model):
    """Test that the model is on the correct device."""
    assert next(model.model.parameters()).device.type in ['cuda', 'cpu']

def test_generate_response(model, sample_image):
    """Test that the model can generate responses."""
    prompt = "What color is this image?"
    response = model.generate_response(
        image=sample_image,
        prompt=prompt,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P
    )
    
    assert isinstance(response, str)
    assert len(response) > 0

def test_generate_response_with_invalid_image(model):
    """Test that the model handles invalid images correctly."""
    with pytest.raises(Exception):
        model.generate_response(
            image=None,
            prompt="What color is this image?",
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P
        )

def test_generate_response_with_empty_prompt(model, sample_image):
    """Test that the model handles empty prompts correctly."""
    with pytest.raises(Exception):
        model.generate_response(
            image=sample_image,
            prompt="",
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P
        ) 