# Usage Examples

This document provides examples of how to use the LLaVA implementation for various tasks.

## Basic Usage

### Initializing the Model

```python
from models.llava import LLaVA

# Initialize the model
model = LLaVA(
    vision_model_path="openai/clip-vit-large-patch14-336",
    language_model_path="lmsys/vicuna-7b-v1.5",
    device="cuda"  # or "cpu"
)
```

### Processing a Single Image

```python
# Generate a response for an image
response = model.generate_from_image(
    image_path="path/to/image.jpg",
    prompt="What can you see in this image?",
    max_new_tokens=256,
    temperature=0.7
)

print(response)
```

### Memory-Efficient Initialization

```python
# Initialize with 8-bit quantization
model = LLaVA(
    vision_model_path="openai/clip-vit-large-patch14-336",
    language_model_path="lmsys/vicuna-7b-v1.5",
    device="cuda",
    load_in_8bit=True  # Enable 8-bit quantization
)

# Or with 4-bit quantization
model = LLaVA(
    vision_model_path="openai/clip-vit-large-patch14-336",
    language_model_path="lmsys/vicuna-7b-v1.5",
    device="cuda",
    load_in_4bit=True  # Enable 4-bit quantization
)
```

## Advanced Usage

### Multi-turn Conversation

```python
# Initialize conversation
conversation = [
    {"role": "system", "content": "You are a helpful assistant that can understand images and answer questions about them."}
]

# First turn
image_path = "path/to/image.jpg"
prompt = "What's in this image?"

response = model.generate_from_image(
    image_path=image_path,
    prompt=prompt,
    conversation=conversation.copy()
)

print(f"User: {prompt}")
print(f"Assistant: {response}\n")

# Update conversation
conversation.append({"role": "user", "content": prompt})
conversation.append({"role": "assistant", "content": response})

# Second turn
prompt = "Can you describe it in more detail?"

response = model.generate_from_image(
    image_path=image_path,
    prompt=prompt,
    conversation=conversation.copy()
)

print(f"User: {prompt}")
print(f"Assistant: {response}")
```

### Saving and Loading Models

```python
# Save the model
model.save_model("path/to/save/model")

# Load the model
loaded_model = LLaVA.from_pretrained("path/to/save/model")
```

## Command-Line Interface

### Basic Demo

```bash
python scripts/demo.py --vision-model openai/clip-vit-large-patch14-336 --language-model lmsys/vicuna-7b-v1.5
```

### Testing with a Specific Image

```bash
python scripts/test_model.py --vision-model openai/clip-vit-large-patch14-336 --language-model lmsys/vicuna-7b-v1.5 --image-url https://example.com/image.jpg --prompt "What's in this image?"
```

### Evaluating on VQA

```bash
python scripts/evaluate_vqa.py --vision-model openai/clip-vit-large-patch14-336 --language-model lmsys/vicuna-7b-v1.5 --questions-file path/to/questions.json --image-folder path/to/images --output-file results.json
```

## Using the Jupyter Notebook

The repository includes a Jupyter notebook (`notebooks/LLaVA_Demo.ipynb`) that demonstrates the model's capabilities. To use it:

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open the notebook and follow the instructions.

## Web Interface

To launch the Gradio web interface:

```bash
python scripts/demo.py --vision-model openai/clip-vit-large-patch14-336 --language-model lmsys/vicuna-7b-v1.5 --share
```

This will start a local web server and provide a URL that you can use to access the interface.