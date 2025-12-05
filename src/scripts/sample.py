"""Sampling script for handwriting generation models."""

import argparse
import os
from typing import List, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from src.models.gan import HandwritingGAN
from src.models.vae import HandwritingVAE
from src.utils.core import get_device, set_seed
from src.utils.sampling import HandwritingSampler


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
    """Main sampling function."""
    parser = argparse.ArgumentParser(description="Generate handwriting samples")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, default="Hello, world!", help="Text to generate handwriting for")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for generation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="samples", help="Output directory")
    parser.add_argument("--save_grid", action="store_true", help="Save sample grid")
    parser.add_argument("--save_individual", action="store_true", help="Save individual samples")
    
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
    
    # Create sampler
    sampler = HandwritingSampler(model, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples for text: '{args.text}'")
    
    samples = sampler.sample_with_guidance(
        text=args.text,
        guidance_scale=args.guidance_scale,
        num_samples=args.num_samples,
        seed=args.seed
    )
    
    # Save samples
    if args.save_grid:
        grid_path = os.path.join(args.output_dir, "sample_grid.png")
        sampler.create_sample_grid(samples, args.text, grid_path)
        print(f"Sample grid saved to: {grid_path}")
    
    if args.save_individual:
        saved_paths = sampler.save_samples_as_images(
            samples, 
            args.text, 
            args.output_dir
        )
        print(f"Individual samples saved to: {args.output_dir}")
        print(f"Saved {len(saved_paths)} files")
    
    # Generate demo samples if no specific text provided
    if args.text == "Hello, world!" and args.num_samples == 4:
        print("Generating demo samples...")
        demo_results = sampler.generate_demo_samples(
            num_samples=args.num_samples,
            save_dir=os.path.join(args.output_dir, "demo")
        )
        print(f"Demo samples saved to: {os.path.join(args.output_dir, 'demo')}")
    
    print("Sampling completed!")


if __name__ == "__main__":
    main()
