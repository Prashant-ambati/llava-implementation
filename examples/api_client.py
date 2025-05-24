"""
Example API client for the LLaVA model.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import requests
from PIL import Image
import base64
from io import BytesIO

def encode_image(image_path: str) -> str:
    """
    Encode an image to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        str: Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(
    api_url: str,
    image_path: str,
    prompt: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None
) -> Dict[str, Any]:
    """
    Process an image using the LLaVA API.
    
    Args:
        api_url: URL of the API endpoint
        image_path: Path to the input image
        prompt: Text prompt for the model
        max_new_tokens: Optional maximum tokens to generate
        temperature: Optional sampling temperature
        top_p: Optional top-p sampling parameter
        
    Returns:
        Dict containing the API response
    """
    # Prepare the request payload
    payload = {
        "image": encode_image(image_path),
        "prompt": prompt
    }
    
    # Add optional parameters if provided
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    
    try:
        # Send the request
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        raise

def save_response(response: Dict[str, Any], output_path: Optional[str] = None):
    """
    Save or print the API response.
    
    Args:
        response: API response dictionary
        output_path: Optional path to save the response
    """
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(response, f, indent=2)
        print(f"Saved response to {output_path}")
    else:
        print("\nAPI Response:")
        print("-" * 50)
        print(json.dumps(response, indent=2))
        print("-" * 50)

def main():
    """Main function to process images using the API."""
    parser = argparse.ArgumentParser(description="Process images using LLaVA API")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("prompt", type=str, help="Text prompt for the model")
    parser.add_argument("--api-url", type=str, default="http://localhost:7860/api/process",
                      help="URL of the API endpoint")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, help="Top-p sampling parameter")
    parser.add_argument("--output", type=str, help="Path to save the response")
    
    args = parser.parse_args()
    
    try:
        # Process image
        response = process_image(
            api_url=args.api_url,
            image_path=args.image_path,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # Save or print response
        save_response(response, args.output)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 