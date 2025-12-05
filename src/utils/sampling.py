"""Sampling utilities for handwriting generation."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.utils.core import get_device, to_numpy, denormalize_image
from src.utils.text import clean_text, create_text_prompts


class HandwritingSampler:
    """Sampler for generating handwriting samples.
    
    Attributes:
        model: Trained model for generation.
        device: Device for computation.
        text_encoder: Text encoder for text processing.
    """
    
    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        """Initialize sampler.
        
        Args:
            model: Trained model for generation.
            device: Device for computation.
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.model.eval()
        
        # Get text encoder
        if hasattr(model, 'text_encoder'):
            self.text_encoder = model.text_encoder
        else:
            from src.utils.text import TextEncoder
            self.text_encoder = TextEncoder()
    
    def sample_from_text(
        self, 
        text: str, 
        num_samples: int = 1,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """Generate handwriting samples from text.
        
        Args:
            text: Input text string.
            num_samples: Number of samples to generate.
            seed: Random seed for reproducibility.
            
        Returns:
            Generated handwriting images.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Clean text
        text = clean_text(text)
        
        with torch.no_grad():
            if hasattr(self.model, 'generate'):
                # VAE generation
                samples = self.model.generate([text] * num_samples, num_samples=1)
            else:
                # GAN generation
                noise = torch.randn(num_samples, self.model.generator.latent_dim, device=self.device)
                samples = self.model(noise, [text] * num_samples)
        
        return samples
    
    def sample_with_guidance(
        self, 
        text: str, 
        guidance_scale: float = 7.5,
        num_samples: int = 1,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """Generate samples with guidance (for diffusion models).
        
        Args:
            text: Input text string.
            guidance_scale: Guidance scale for generation.
            num_samples: Number of samples to generate.
            seed: Random seed for reproducibility.
            
        Returns:
            Generated handwriting images.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Clean text
        text = clean_text(text)
        
        with torch.no_grad():
            # This is a placeholder for diffusion model guidance
            # In practice, you would implement classifier-free guidance here
            if hasattr(self.model, 'generate'):
                samples = self.model.generate([text] * num_samples, num_samples=1)
            else:
                noise = torch.randn(num_samples, self.model.generator.latent_dim, device=self.device)
                samples = self.model(noise, [text] * num_samples)
        
        return samples
    
    def interpolate_between_texts(
        self, 
        text1: str, 
        text2: str, 
        num_steps: int = 10,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """Interpolate between two texts in latent space.
        
        Args:
            text1: First text string.
            text2: Second text string.
            num_steps: Number of interpolation steps.
            seed: Random seed for reproducibility.
            
        Returns:
            Interpolated handwriting images.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Clean texts
        text1 = clean_text(text1)
        text2 = clean_text(text2)
        
        with torch.no_grad():
            if hasattr(self.model, 'generate'):
                # For VAE, interpolate in latent space
                z1 = torch.randn(1, self.model.encoder.latent_dim, device=self.device)
                z2 = torch.randn(1, self.model.encoder.latent_dim, device=self.device)
                
                interpolated_samples = []
                for i in range(num_steps):
                    alpha = i / (num_steps - 1)
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    
                    # Use text1 for all interpolations (could be modified)
                    sample = self.model.decoder(z_interp, self.text_encoder([text1]))
                    interpolated_samples.append(sample)
                
                return torch.cat(interpolated_samples, dim=0)
            else:
                # For GAN, interpolate noise
                noise1 = torch.randn(1, self.model.generator.latent_dim, device=self.device)
                noise2 = torch.randn(1, self.model.generator.latent_dim, device=self.device)
                
                interpolated_samples = []
                for i in range(num_steps):
                    alpha = i / (num_steps - 1)
                    noise_interp = (1 - alpha) * noise1 + alpha * noise2
                    
                    # Alternate between texts
                    text = text1 if i < num_steps // 2 else text2
                    sample = self.model(noise_interp, [text])
                    interpolated_samples.append(sample)
                
                return torch.cat(interpolated_samples, dim=0)
    
    def create_sample_grid(
        self, 
        samples: torch.Tensor, 
        text: str,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """Create a grid visualization of samples.
        
        Args:
            samples: Generated samples tensor.
            text: Text used for generation.
            save_path: Path to save the figure.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure.
        """
        num_samples = samples.size(0)
        cols = min(4, num_samples)
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Denormalize samples
        samples = denormalize_image(samples)
        samples = torch.clamp(samples, 0, 1)
        
        for i in range(num_samples):
            row = i // cols
            col = i % cols
            
            # Convert to numpy
            sample = to_numpy(samples[i])
            
            # Plot
            if sample.ndim == 3:
                axes[row, col].imshow(sample.transpose(1, 2, 0), cmap='gray')
            else:
                axes[row, col].imshow(sample, cmap='gray')
            
            axes[row, col].set_title(f"Sample {i+1}")
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        fig.suptitle(f"Generated Handwriting: '{text}'", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def save_samples_as_images(
        self, 
        samples: torch.Tensor, 
        text: str,
        save_dir: str,
        prefix: str = "sample"
    ) -> List[str]:
        """Save samples as individual image files.
        
        Args:
            samples: Generated samples tensor.
            text: Text used for generation.
            save_dir: Directory to save images.
            prefix: Prefix for filenames.
            
        Returns:
            List of saved file paths.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Denormalize samples
        samples = denormalize_image(samples)
        samples = torch.clamp(samples, 0, 1)
        
        saved_paths = []
        
        for i, sample in enumerate(samples):
            # Convert to numpy
            sample_np = to_numpy(sample)
            
            # Convert to PIL Image
            if sample_np.ndim == 3:
                sample_np = sample_np.transpose(1, 2, 0)
            
            sample_np = (sample_np * 255).astype(np.uint8)
            
            if sample_np.ndim == 2:
                image = Image.fromarray(sample_np, mode='L')
            else:
                image = Image.fromarray(sample_np)
            
            # Save image
            filename = f"{prefix}_{i:03d}.png"
            filepath = os.path.join(save_dir, filename)
            image.save(filepath)
            saved_paths.append(filepath)
        
        # Save text
        text_file = os.path.join(save_dir, "text.txt")
        with open(text_file, "w") as f:
            f.write(text)
        
        return saved_paths
    
    def generate_demo_samples(
        self, 
        num_samples: int = 16,
        save_dir: str = "demo_samples"
    ) -> Dict[str, Any]:
        """Generate demo samples with various texts.
        
        Args:
            num_samples: Number of samples to generate per text.
            save_dir: Directory to save samples.
            
        Returns:
            Dictionary containing generated samples and metadata.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Get example texts
        texts = create_text_prompts()
        
        all_samples = {}
        
        for i, text in enumerate(texts[:5]):  # Limit to first 5 texts
            print(f"Generating samples for: '{text}'")
            
            # Generate samples
            samples = self.sample_from_text(text, num_samples=num_samples)
            
            # Save samples
            text_dir = os.path.join(save_dir, f"text_{i:03d}")
            saved_paths = self.save_samples_as_images(samples, text, text_dir)
            
            # Create grid
            grid_path = os.path.join(text_dir, "grid.png")
            self.create_sample_grid(samples, text, grid_path)
            
            all_samples[text] = {
                "samples": samples,
                "saved_paths": saved_paths,
                "grid_path": grid_path,
            }
        
        return all_samples
    
    def interactive_sampling(
        self, 
        text: str,
        num_samples: int = 4,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Interactive sampling with various parameters.
        
        Args:
            text: Input text string.
            num_samples: Number of samples to generate.
            guidance_scale: Guidance scale for generation.
            seed: Random seed for reproducibility.
            
        Returns:
            Dictionary containing samples and parameters.
        """
        # Generate samples
        samples = self.sample_with_guidance(
            text=text,
            guidance_scale=guidance_scale,
            num_samples=num_samples,
            seed=seed
        )
        
        # Create visualization
        fig = self.create_sample_grid(samples, text)
        
        return {
            "samples": samples,
            "text": text,
            "num_samples": num_samples,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "figure": fig,
        }
