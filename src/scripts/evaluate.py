"""Evaluation script for handwriting generation models."""

import argparse
import os
from typing import Dict, Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import create_data_module
from src.evaluation.evaluator import HandwritingEvaluator
from src.models.gan import HandwritingGAN
from src.models.vae import HandwritingVAE
from src.utils.core import get_device, set_seed
from src.utils.text import create_text_prompts


def load_model(config: DictConfig, checkpoint_path: str) -> torch.nn.Module:
    """Load trained model from checkpoint.
    
    Args:
        config: Model configuration.
        checkpoint_path: Path to checkpoint file.
        
    Returns:
        Loaded model.
    """
    # Determine model type
    model_type = config.model._target_.split(".")[-1].lower()
    
    if model_type == "handwritinggan":
        model = HandwritingGAN(config.model)
    elif model_type == "handwritingvae":
        model = HandwritingVAE(config.model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract model state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove "model." prefix if present
        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                model_state_dict[key[6:]] = value
            else:
                model_state_dict[key] = value
        model.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    return model


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate handwriting generation model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    else:
        set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(config, args.checkpoint)
    model.to(device)
    model.eval()
    
    # Create data module
    data_module = create_data_module(config.data)
    
    # Create evaluator
    evaluator = HandwritingEvaluator(model, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get text prompts for generation evaluation
    text_prompts = create_text_prompts()[:5]  # Use first 5 prompts
    
    print("Starting evaluation...")
    
    # Generate comprehensive report
    results = evaluator.generate_report(
        dataloader=data_module.test_dataloader(),
        text_prompts=text_prompts,
        save_dir=args.output_dir
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print("\nReconstruction Metrics:")
    for metric, value in results["reconstruction_metrics"].items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\nDiversity Metrics:")
    for metric, value in results["diversity_metrics"].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
