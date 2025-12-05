"""Training script for handwriting generation models."""

import argparse
import os
from typing import Dict, Any

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import create_data_module
from src.training.trainer import create_training_module, create_callbacks, create_logger
from src.utils.core import set_seed, get_device, create_directories


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train handwriting generation model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--overrides", nargs="*", help="Override config values")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Apply overrides
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    
    # Set seed
    set_seed(config.seed)
    
    # Create directories
    create_directories(config)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Create data module
    data_module = create_data_module(config.data)
    
    # Create training module
    training_module = create_training_module(config)
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    # Create logger
    logger = create_logger(config)
    
    # Create trainer
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
    )
    
    # Train model
    if args.resume:
        trainer.fit(training_module, data_module, ckpt_path=args.resume)
    else:
        trainer.fit(training_module, data_module)
    
    # Save final model
    final_model_path = os.path.join(config.checkpoint_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
