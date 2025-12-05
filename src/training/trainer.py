"""Training utilities for handwriting generation models."""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from src.models.gan import HandwritingGAN
from src.models.vae import HandwritingVAE
from src.utils.core import EarlyStopping as CustomEarlyStopping


class GANTrainingModule(pl.LightningModule):
    """PyTorch Lightning module for GAN training.
    
    Attributes:
        model: GAN model.
        config: Training configuration.
        generator_loss: Generator loss function.
        discriminator_loss: Discriminator loss function.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize GAN training module.
        
        Args:
            config: Training configuration dictionary.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.model = HandwritingGAN(config.model)
        
        # Loss functions
        self.generator_loss = self._get_loss_function(config.model.loss.generator_loss)
        self.discriminator_loss = self._get_loss_function(config.model.loss.discriminator_loss)
        
        # EMA for generator
        if config.model.loss.use_ema:
            self.generator_ema = self._create_ema_model()
        
        # Metrics
        self.g_loss = []
        self.d_loss = []
    
    def _get_loss_function(self, loss_type: str) -> nn.Module:
        """Get loss function by type.
        
        Args:
            loss_type: Type of loss function.
            
        Returns:
            Loss function module.
        """
        if loss_type == "hinge":
            return nn.ReLU()
        elif loss_type == "ns":
            return nn.BCEWithLogitsLoss()
        elif loss_type == "wgan":
            return lambda x: -x.mean()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _create_ema_model(self) -> HandwritingGenerator:
        """Create EMA model for generator.
        
        Returns:
            EMA generator model.
        """
        ema_model = HandwritingGenerator(**self.config.model.generator)
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    
    def _update_ema(self) -> None:
        """Update EMA model."""
        if hasattr(self, 'generator_ema'):
            decay = self.config.model.loss.ema_decay
            for ema_param, param in zip(self.generator_ema.parameters(), self.model.generator.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step for GAN.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Dictionary containing losses and metrics.
        """
        real_images = batch["image"]
        text = batch["text"]
        batch_size = real_images.size(0)
        
        # Generate noise
        noise = torch.randn(batch_size, self.config.model.generator.latent_dim, device=self.device)
        
        # Train Discriminator
        d_loss = self._train_discriminator(real_images, noise, text)
        
        # Train Generator
        g_loss = self._train_generator(noise, text)
        
        # Update EMA
        self._update_ema()
        
        # Log losses
        self.log("train/d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/g_loss", g_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {"d_loss": d_loss, "g_loss": g_loss}
    
    def _train_discriminator(
        self, 
        real_images: torch.Tensor, 
        noise: torch.Tensor, 
        text: List[str]
    ) -> torch.Tensor:
        """Train discriminator.
        
        Args:
            real_images: Real images.
            noise: Random noise.
            text: Text strings.
            
        Returns:
            Discriminator loss.
        """
        # Generate fake images
        with torch.no_grad():
            fake_images = self.model(noise, text)
        
        # Real images
        real_pred = self.model.discriminator(real_images, self.model.text_encoder(text))
        real_loss = self.discriminator_loss(1 - real_pred).mean()
        
        # Fake images
        fake_pred = self.model.discriminator(fake_images, self.model.text_encoder(text))
        fake_loss = self.discriminator_loss(1 + fake_pred).mean()
        
        # Gradient penalty
        if self.config.model.loss.lambda_gp > 0:
            gp_loss = self._gradient_penalty(real_images, fake_images, text)
            d_loss = real_loss + fake_loss + self.config.model.loss.lambda_gp * gp_loss
        else:
            d_loss = real_loss + fake_loss
        
        # Optimize discriminator
        self.optimizers()[1].zero_grad()
        self.manual_backward(d_loss)
        self.optimizers()[1].step()
        
        return d_loss
    
    def _train_generator(self, noise: torch.Tensor, text: List[str]) -> torch.Tensor:
        """Train generator.
        
        Args:
            noise: Random noise.
            text: Text strings.
            
        Returns:
            Generator loss.
        """
        # Generate fake images
        fake_images = self.model(noise, text)
        
        # Discriminator prediction
        fake_pred = self.model.discriminator(fake_images, self.model.text_encoder(text))
        
        # Generator loss
        g_loss = self.generator_loss(-fake_pred).mean()
        
        # Optimize generator
        self.optimizers()[0].zero_grad()
        self.manual_backward(g_loss)
        self.optimizers()[0].step()
        
        return g_loss
    
    def _gradient_penalty(
        self, 
        real_images: torch.Tensor, 
        fake_images: torch.Tensor, 
        text: List[str]
    ) -> torch.Tensor:
        """Calculate gradient penalty for WGAN-GP.
        
        Args:
            real_images: Real images.
            fake_images: Fake images.
            text: Text strings.
            
        Returns:
            Gradient penalty loss.
        """
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)
        
        text_embeddings = self.model.text_encoder(text)
        pred = self.model.discriminator(interpolated, text_embeddings)
        
        gradients = torch.autograd.grad(
            outputs=pred,
            inputs=interpolated,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Dictionary containing validation metrics.
        """
        real_images = batch["image"]
        text = batch["text"]
        batch_size = real_images.size(0)
        
        # Generate samples
        noise = torch.randn(batch_size, self.config.model.generator.latent_dim, device=self.device)
        fake_images = self.model(noise, text)
        
        # Calculate metrics
        d_real = self.model.discriminator(real_images, self.model.text_encoder(text))
        d_fake = self.model.discriminator(fake_images, self.model.text_encoder(text))
        
        # Log metrics
        self.log("val/d_real", d_real.mean(), on_epoch=True)
        self.log("val/d_fake", d_fake.mean(), on_epoch=True)
        
        return {"d_real": d_real.mean(), "d_fake": d_fake.mean()}
    
    def configure_optimizers(self) -> List[optim.Optimizer]:
        """Configure optimizers.
        
        Returns:
            List of optimizers for generator and discriminator.
        """
        g_optimizer = optim.Adam(
            self.model.generator.parameters(),
            lr=self.config.model.lr_g,
            betas=(self.config.model.beta1, self.config.model.beta2),
            weight_decay=self.config.model.weight_decay,
        )
        
        d_optimizer = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.config.model.lr_d,
            betas=(self.config.model.beta1, self.config.model.beta2),
            weight_decay=self.config.model.weight_decay,
        )
        
        return [g_optimizer, d_optimizer]


class VAETrainingModule(pl.LightningModule):
    """PyTorch Lightning module for VAE training.
    
    Attributes:
        model: VAE model.
        config: Training configuration.
        reconstruction_loss: Reconstruction loss function.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize VAE training module.
        
        Args:
            config: Training configuration dictionary.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.model = HandwritingVAE(config.model)
        
        # Loss functions
        self.reconstruction_loss = self._get_reconstruction_loss(config.model.loss.reconstruction_loss)
        
        # KL annealing
        self.kl_weight = config.model.loss.kl_weight
        self.use_annealing = config.model.loss.use_annealing
        self.annealing_steps = config.model.loss.annealing_steps
    
    def _get_reconstruction_loss(self, loss_type: str) -> nn.Module:
        """Get reconstruction loss function.
        
        Args:
            loss_type: Type of reconstruction loss.
            
        Returns:
            Reconstruction loss function.
        """
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "bce":
            return nn.BCELoss()
        elif loss_type == "l1":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown reconstruction loss type: {loss_type}")
    
    def _get_kl_weight(self, step: int) -> float:
        """Get KL weight with optional annealing.
        
        Args:
            step: Current training step.
            
        Returns:
            KL weight.
        """
        if self.use_annealing:
            return min(1.0, step / self.annealing_steps) * self.kl_weight
        return self.kl_weight
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step for VAE.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Dictionary containing losses and metrics.
        """
        images = batch["image"]
        text = batch["text"]
        
        # Forward pass
        outputs = self.model(images, text)
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(outputs["reconstruction"], images)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(
            1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp()
        ) / images.size(0)
        
        # Get KL weight
        kl_weight = self._get_kl_weight(self.global_step)
        
        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss
        
        # Log losses
        self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/kl_loss", kl_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/kl_weight", kl_weight, on_step=True, on_epoch=True)
        
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Dictionary containing validation metrics.
        """
        images = batch["image"]
        text = batch["text"]
        
        # Forward pass
        outputs = self.model(images, text)
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(outputs["reconstruction"], images)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(
            1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp()
        ) / images.size(0)
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        # Log losses
        self.log("val/recon_loss", recon_loss, on_epoch=True)
        self.log("val/kl_loss", kl_loss, on_epoch=True)
        self.log("val/total_loss", total_loss, on_epoch=True)
        
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
    
    def configure_optimizers(self) -> optim.Optimizer:
        """Configure optimizer.
        
        Returns:
            Optimizer for VAE.
        """
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.model.lr,
            betas=(self.config.model.beta1, self.config.model.beta2),
            weight_decay=self.config.model.weight_decay,
        )


def create_training_module(config: Dict[str, Any]) -> pl.LightningModule:
    """Create training module based on model type.
    
    Args:
        config: Training configuration.
        
    Returns:
        Appropriate training module.
    """
    model_type = config.model._target_.split(".")[-1].lower()
    
    if model_type == "handwritinggan":
        return GANTrainingModule(config)
    elif model_type == "handwritingvae":
        return VAETrainingModule(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_callbacks(config: Dict[str, Any]) -> List[pl.Callback]:
    """Create training callbacks.
    
    Args:
        config: Training configuration.
        
    Returns:
        List of training callbacks.
    """
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config.get("patience", 10),
        verbose=True,
    )
    callbacks.append(early_stopping)
    
    return callbacks


def create_logger(config: Dict[str, Any]) -> Optional[pl.loggers.Logger]:
    """Create logger for training.
    
    Args:
        config: Training configuration.
        
    Returns:
        Logger instance or None.
    """
    if config.use_wandb:
        return WandbLogger(
            project=config.wandb_project,
            entity=config.wandb_entity,
            save_dir=config.log_dir,
        )
    else:
        return TensorBoardLogger(
            save_dir=config.log_dir,
            name="handwriting_generation",
        )
