#!/usr/bin/env python3
"""
Example script demonstrating the modernized handwriting generation system.

This script shows how to:
1. Train a handwriting generation model
2. Generate samples from the trained model
3. Evaluate model performance
4. Use the interactive demo

Run this script to see the complete workflow in action.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from src.models.gan import HandwritingGAN
from src.models.vae import HandwritingVAE
from src.data.dataset import create_data_module
from src.training.trainer import create_training_module, create_callbacks, create_logger
from src.evaluation.evaluator import HandwritingEvaluator
from src.utils.core import set_seed, get_device, create_directories
from src.utils.sampling import HandwritingSampler
from src.utils.text import create_text_prompts


def setup_example_environment():
    """Set up the example environment."""
    print("Setting up example environment...")
    
    # Create necessary directories
    directories = ["data", "outputs", "checkpoints", "logs", "assets", "demo_samples"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Environment setup complete!")


def train_example_model(model_type: str = "gan", epochs: int = 5):
    """Train an example model for demonstration."""
    print(f"\nTraining {model_type.upper()} model for {epochs} epochs...")
    
    # Load configuration
    config_path = f"configs/model/{model_type}.yaml"
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        return None
    
    config = OmegaConf.load(config_path)
    
    # Add training configuration
    config.update({
        "seed": 42,
        "device": "auto",
        "precision": "32",
        "batch_size": 16,
        "max_epochs": epochs,
        "learning_rate": 0.0002,
        "log_every_n_steps": 10,
        "validate_every_n_epochs": 2,
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs",
        "use_wandb": False,
        "data": {
            "image_size": (64, 256),
            "font_size": 24,
            "train_samples": 200,
            "val_samples": 50,
            "test_samples": 50,
            "batch_size": 16,
            "num_workers": 2,
            "pin_memory": True
        }
    })
    
    # Set seed
    set_seed(config.seed)
    
    # Create directories
    create_directories(config)
    
    # Create data module
    data_module = create_data_module(config.data)
    
    # Create training module
    training_module = create_training_module(config)
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    # Create logger
    logger = create_logger(config)
    
    # Create trainer
    import pytorch_lightning as pl
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        devices=1,
        accelerator="auto",
        precision=config.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.validate_every_n_epochs,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=False,  # Set to True for quick testing
    )
    
    # Train model
    try:
        trainer.fit(training_module, data_module)
        
        # Save final model
        checkpoint_path = f"checkpoints/{model_type}_example_model.ckpt"
        trainer.save_checkpoint(checkpoint_path)
        print(f"Model saved to: {checkpoint_path}")
        
        return checkpoint_path
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return None


def generate_example_samples(checkpoint_path: str, model_type: str = "gan"):
    """Generate example samples from trained model."""
    print(f"\nGenerating samples from {model_type.upper()} model...")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Load configuration
    config_path = f"configs/model/{model_type}.yaml"
    config = OmegaConf.load(config_path)
    
    # Get device
    device = get_device()
    
    # Load model
    if model_type == "gan":
        model = HandwritingGAN(config)
    elif model_type == "vae":
        model = HandwritingVAE(config)
    else:
        print(f"Unknown model type: {model_type}")
        return
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                model_state_dict[key[6:]] = value
            else:
                model_state_dict[key] = value
        model.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Create sampler
    sampler = HandwritingSampler(model, device)
    
    # Generate samples for example texts
    example_texts = [
        "Hello, world!",
        "Deep learning is amazing.",
        "Handwriting generation works!",
    ]
    
    output_dir = "demo_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, text in enumerate(example_texts):
        print(f"Generating samples for: '{text}'")
        
        # Generate samples
        samples = sampler.sample_from_text(text, num_samples=4, seed=42)
        
        # Save samples
        text_dir = os.path.join(output_dir, f"text_{i:03d}")
        sampler.save_samples_as_images(samples, text, text_dir)
        
        # Create grid
        grid_path = os.path.join(text_dir, "grid.png")
        sampler.create_sample_grid(samples, text, grid_path)
        
        print(f"Samples saved to: {text_dir}")
    
    print(f"All samples saved to: {output_dir}")


def evaluate_example_model(checkpoint_path: str, model_type: str = "gan"):
    """Evaluate the trained model."""
    print(f"\nEvaluating {model_type.upper()} model...")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Load configuration
    config_path = f"configs/model/{model_type}.yaml"
    config = OmegaConf.load(config_path)
    
    # Add data configuration
    config.data = {
        "image_size": (64, 256),
        "font_size": 24,
        "train_samples": 200,
        "val_samples": 50,
        "test_samples": 50,
        "batch_size": 16,
        "num_workers": 2,
        "pin_memory": True
    }
    
    # Get device
    device = get_device()
    
    # Load model
    if model_type == "gan":
        model = HandwritingGAN(config)
    elif model_type == "vae":
        model = HandwritingVAE(config)
    else:
        print(f"Unknown model type: {model_type}")
        return
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                model_state_dict[key[6:]] = value
            else:
                model_state_dict[key] = value
        model.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Create data module
    data_module = create_data_module(config.data)
    
    # Create evaluator
    evaluator = HandwritingEvaluator(model, device)
    
    # Get text prompts
    text_prompts = create_text_prompts()[:3]
    
    # Evaluate
    output_dir = "evaluation_results"
    results = evaluator.generate_report(
        dataloader=data_module.test_dataloader(),
        text_prompts=text_prompts,
        save_dir=output_dir
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 40)
    
    print("\nReconstruction Metrics:")
    for metric, value in results["reconstruction_metrics"].items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\nDiversity Metrics:")
    for metric, value in results["diversity_metrics"].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nDetailed results saved to: {output_dir}")


def run_interactive_demo():
    """Run the interactive Streamlit demo."""
    print("\nStarting interactive demo...")
    print("The demo will open in your web browser.")
    print("Press Ctrl+C to stop the demo.")
    
    try:
        subprocess.run(["streamlit", "run", "demo/app.py"], check=True)
    except subprocess.CalledProcessError:
        print("Failed to start Streamlit demo. Make sure Streamlit is installed:")
        print("pip install streamlit")
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")


def main():
    """Main example function."""
    parser = argparse.ArgumentParser(description="Handwriting Generation Example")
    parser.add_argument("--model", choices=["gan", "vae"], default="gan", help="Model type to use")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--skip-training", action="store_true", help="Skip training step")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation step")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation step")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HANDWRITING GENERATION - EXAMPLE SCRIPT")
    print("=" * 60)
    
    # Setup environment
    setup_example_environment()
    
    checkpoint_path = None
    
    # Training step
    if not args.skip_training:
        checkpoint_path = train_example_model(args.model, args.epochs)
        if checkpoint_path is None:
            print("Training failed. Exiting.")
            return
    else:
        # Look for existing checkpoint
        checkpoint_path = f"checkpoints/{args.model}_example_model.ckpt"
        if not os.path.exists(checkpoint_path):
            print(f"No existing checkpoint found: {checkpoint_path}")
            print("Please train a model first or remove --skip-training flag.")
            return
    
    # Generation step
    if not args.skip_generation:
        generate_example_samples(checkpoint_path, args.model)
    
    # Evaluation step
    if not args.skip_evaluation:
        evaluate_example_model(checkpoint_path, args.model)
    
    # Interactive demo
    if args.demo:
        run_interactive_demo()
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check the 'demo_samples' directory for generated samples")
    print("2. Check the 'evaluation_results' directory for evaluation metrics")
    print("3. Run 'streamlit run demo/app.py' to start the interactive demo")
    print("4. Modify the configuration files to experiment with different settings")
    print("5. Add your own datasets by implementing custom dataset classes")


if __name__ == "__main__":
    main()
