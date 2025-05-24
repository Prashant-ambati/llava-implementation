"""
Utility functions for evaluating LLaVA models.
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def evaluate_vqa(
    model,
    questions_file: str,
    image_folder: str,
    output_file: Optional[str] = None,
    max_new_tokens: int = 100
) -> Dict:
    """
    Evaluate the model on visual question answering.
    
    Args:
        model: LLaVA model
        questions_file: Path to the questions JSON file
        image_folder: Path to the folder containing images
        output_file: Optional path to save results
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Dictionary with evaluation results
    """
    # Load questions
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    results = []
    
    # Process each question
    for q in tqdm(questions, desc="Evaluating VQA"):
        image_path = os.path.join(image_folder, q['image'])
        question_text = q['question']
        
        # Generate answer
        try:
            answer = model.generate_from_image(
                image_path=image_path,
                prompt=question_text,
                max_new_tokens=max_new_tokens
            )
            
            result = {
                'question_id': q.get('question_id', None),
                'image': q['image'],
                'question': question_text,
                'answer': answer,
                'gt_answer': q.get('answer', None)
            }
            
            results.append(result)
        except Exception as e:
            print(f"Error processing question {q.get('question_id', '')}: {e}")
    
    # Save results if output file is provided
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    
    # Calculate accuracy if ground truth answers are available
    accuracy = None
    if all('gt_answer' in r and r['gt_answer'] is not None for r in results):
        correct = 0
        for r in results:
            # Simple exact match accuracy
            if r['answer'].lower() == r['gt_answer'].lower():
                correct += 1
        
        accuracy = correct / len(results) if results else 0
    
    return {
        'results': results,
        'accuracy': accuracy,
        'num_questions': len(results)
    }


def visualize_results(
    results: List[Dict],
    num_examples: int = 5,
    figsize: Tuple[int, int] = (15, 10),
    image_folder: str = None
) -> None:
    """
    Visualize VQA results.
    
    Args:
        results: List of result dictionaries
        num_examples: Number of examples to visualize
        figsize: Figure size
        image_folder: Path to the folder containing images
    """
    # Select a subset of results
    if len(results) > num_examples:
        indices = np.random.choice(len(results), num_examples, replace=False)
        selected_results = [results[i] for i in indices]
    else:
        selected_results = results
    
    # Create figure
    fig, axes = plt.subplots(len(selected_results), 1, figsize=figsize)
    if len(selected_results) == 1:
        axes = [axes]
    
    # Plot each example
    for i, result in enumerate(selected_results):
        # Load image
        if image_folder:
            image_path = os.path.join(image_folder, result['image'])
            img = Image.open(image_path).convert('RGB')
            axes[i].imshow(img)
        
        # Set title and text
        title = f"Q: {result['question']}"
        text = f"A: {result['answer']}"
        if 'gt_answer' in result and result['gt_answer']:
            text += f"\nGT: {result['gt_answer']}"
        
        axes[i].set_title(title)
        axes[i].text(0, -0.5, text, transform=axes[i].transAxes, fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def compute_metrics(results: List[Dict]) -> Dict:
    """
    Compute evaluation metrics.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Check if ground truth answers are available
    has_gt = all('gt_answer' in r and r['gt_answer'] is not None for r in results)
    
    if has_gt:
        # Exact match accuracy
        correct = 0
        for r in results:
            if r['answer'].lower() == r['gt_answer'].lower():
                correct += 1
        
        metrics['exact_match_accuracy'] = correct / len(results) if results else 0
        
        # Token overlap (simple BLEU-like metric)
        total_overlap = 0
        for r in results:
            pred_tokens = set(r['answer'].lower().split())
            gt_tokens = set(r['gt_answer'].lower().split())
            
            if gt_tokens:  # Avoid division by zero
                overlap = len(pred_tokens.intersection(gt_tokens)) / len(gt_tokens)
                total_overlap += overlap
        
        metrics['token_overlap'] = total_overlap / len(results) if results else 0
    
    # Response length statistics
    lengths = [len(r['answer'].split()) for r in results]
    metrics['avg_response_length'] = sum(lengths) / len(lengths) if lengths else 0
    metrics['min_response_length'] = min(lengths) if lengths else 0
    metrics['max_response_length'] = max(lengths) if lengths else 0
    
    return metrics