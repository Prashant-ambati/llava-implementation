# LLaVA Scripts

This directory contains various scripts for working with the LLaVA model.

## Available Scripts

- `demo.py`: Launches a Gradio web interface for interacting with the LLaVA model.
- `evaluate_vqa.py`: Evaluates the LLaVA model on visual question answering datasets.
- `test_model.py`: A simple script to test the LLaVA model on a single image.

## Usage Examples

### Demo

Launch the Gradio web interface:

```bash
python scripts/demo.py --vision-model openai/clip-vit-large-patch14-336 --language-model lmsys/vicuna-7b-v1.5 --load-8bit
```

### Evaluate VQA

Evaluate the model on a VQA dataset:

```bash
python scripts/evaluate_vqa.py --vision-model openai/clip-vit-large-patch14-336 --language-model lmsys/vicuna-7b-v1.5 --questions-file path/to/questions.json --image-folder path/to/images --output-file results.json --load-8bit
```

### Test Model

Test the model on a single image:

```bash
python scripts/test_model.py --vision-model openai/clip-vit-large-patch14-336 --language-model lmsys/vicuna-7b-v1.5 --image-url https://example.com/image.jpg --prompt "What's in this image?" --load-8bit
```

## Options

Most scripts support the following options:

- `--vision-model`: Path or name of the vision model (default: "openai/clip-vit-large-patch14-336")
- `--language-model`: Path or name of the language model (default: "lmsys/vicuna-7b-v1.5")
- `--load-8bit`: Load the language model in 8-bit precision (reduces memory usage)
- `--load-4bit`: Load the language model in 4-bit precision (further reduces memory usage)
- `--device`: Device to run the model on (default: cuda if available, otherwise cpu)

See the individual script help messages for more specific options:

```bash
python scripts/script_name.py --help
```