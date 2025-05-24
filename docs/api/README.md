# LLaVA API Documentation

## Overview

The LLaVA API provides a simple interface for interacting with the LLaVA model through a Gradio web interface. The API allows users to upload images and receive AI-generated responses about the image content.

## API Endpoints

### Web Interface

The main interface is served at the root URL (`/`) and provides the following components:

#### Input Components

1. **Image Upload**
   - Type: Image uploader
   - Format: PIL Image
   - Purpose: Upload an image for analysis

2. **Prompt Input**
   - Type: Text input
   - Purpose: Enter questions or prompts about the image
   - Default placeholder: "What can you see in this image?"

3. **Generation Parameters**
   - Max New Tokens (64-2048, default: 512)
   - Temperature (0.1-1.0, default: 0.7)
   - Top P (0.1-1.0, default: 0.9)

#### Output Components

1. **Response**
   - Type: Text output
   - Purpose: Displays the model's response
   - Features: Copy button, scrollable

## Usage Examples

### Basic Usage

1. Upload an image using the image uploader
2. Enter a prompt in the text input
3. Click "Generate Response"
4. View the response in the output box

### Example Prompts

- "What can you see in this image?"
- "Describe this scene in detail"
- "What emotions does this image convey?"
- "What's happening in this picture?"
- "Can you identify any objects or people in this image?"

## Error Handling

The API handles various error cases:

1. **Invalid Images**
   - Returns an error message if the image is invalid or corrupted
   - Supports common image formats (JPEG, PNG, etc.)

2. **Empty Prompts**
   - Returns an error message if no prompt is provided
   - Prompts should be non-empty strings

3. **Model Errors**
   - Returns descriptive error messages for model-related issues
   - Includes logging for debugging

## Configuration

The API can be configured through environment variables or the settings file:

- `API_HOST`: Server host (default: "0.0.0.0")
- `API_PORT`: Server port (default: 7860)
- `GRADIO_THEME`: Interface theme (default: "soft")
- `DEFAULT_MAX_NEW_TOKENS`: Default token limit (default: 512)
- `DEFAULT_TEMPERATURE`: Default temperature (default: 0.7)
- `DEFAULT_TOP_P`: Default top-p value (default: 0.9)

## Development

### Running Locally

```bash
python src/api/app.py
```

### Running Tests

```bash
pytest tests/
```

### Code Style

The project follows PEP 8 guidelines. To check your code:

```bash
flake8 src/
black src/
```

## Security Considerations

1. The API is designed for public use but should be deployed behind appropriate security measures
2. Input validation is performed on all user inputs
3. Large file uploads are handled safely
4. Error messages are sanitized to prevent information leakage

## Rate Limiting

Currently, no rate limiting is implemented. Consider implementing rate limiting for production deployments.

## Future Improvements

1. Add authentication
2. Implement rate limiting
3. Add batch processing capabilities
4. Support for video input
5. Real-time streaming responses 