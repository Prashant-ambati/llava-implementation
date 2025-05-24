"""
Script to evaluate LLaVA on Visual Question Answering datasets.
"""

import os
import sys
import json
import argparse
from tqdm import tqdm
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llava import LLaVA
from utils.eval_utils import evaluate_vqa, visualize_results, compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA on VQA")
    parser.add_argument("--vision-model", type=str, default="openai/clip-vit-large-patch14-336",
                        help="Vision model name or path")
    parser.add_argument("--language-model", type=str, default="lmsys/vicuna-7b-v1.5",
                        help="Language model name or path")
    parser.add_argument("--questions-file", type=str, required=True,
                        help="Path to questions JSON file")
    parser.add_argument("--image-folder", type=str, required=True,
                        help="Path to folder containing images")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to save results")
    parser.add_argument("--load-8bit", action="store_true", 
                        help="Load language model in 8-bit mode")
    parser.add_argument("--load-4bit", action="store_true", 
                        help="Load language model in 4-bit mode")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run the model on")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize results")
    parser.add_argument("--num-examples", type=int, default=5,
                        help="Number of examples to visualize")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check if files exist
    if not os.path.exists(args.questions_file):
        raise FileNotFoundError(f"Questions file not found: {args.questions_file}")
    
    if not os.path.exists(args.image_folder):
        raise FileNotFoundError(f"Image folder not found: {args.image_folder}")
    
    # Initialize model
    print(f"Initializing LLaVA model...")
    model = LLaVA(
        vision_model_path=args.vision_model,
        language_model_path=args.language_model,
        device=args.device,
        load_in_8bit=args.load_8bit,
        load_in_4bit=args.load_4bit
    )
    
    print(f"Model initialized on {model.device}")
    
    # Evaluate model
    print(f"Evaluating on {args.questions_file}...")
    eval_results = evaluate_vqa(
        model=model,
        questions_file=args.questions_file,
        image_folder=args.image_folder,
        output_file=args.output_file,
        max_new_tokens=args.max_new_tokens
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Number of questions: {eval_results['num_questions']}")
    
    if eval_results['accuracy'] is not None:
        print(f"Accuracy: {eval_results['accuracy']:.4f}")
    
    # Compute additional metrics
    metrics = compute_metrics(eval_results['results'])
    print("\nAdditional Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize results if requested
    if args.visualize:
        print(f"\nVisualizing {args.num_examples} examples...")
        visualize_results(
            results=eval_results['results'],
            num_examples=args.num_examples,
            image_folder=args.image_folder
        )


if __name__ == "__main__":
    main()