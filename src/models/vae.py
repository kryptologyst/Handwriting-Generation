"""VAE model for handwriting generation."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """Encoder block with downsampling.
    
    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        use_spectral_norm: Whether to use spectral normalization.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_spectral_norm: bool = False,
    ):
        """Initialize encoder block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            use_spectral_norm: Whether to use spectral normalization.
        """
        super().__init__()
        
        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if use_spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsampling
        self.downsample = nn.AvgPool2d(2)
        
        # Activation
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        # First convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # Second convolution
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        # Downsample
        x = self.downsample(x)
        
        return x


class DecoderBlock(nn.Module):
    """Decoder block with upsampling.
    
    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """Initialize decoder block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        
        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Activation
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        # Upsample
        x = self.upsample(x)
        
        # First convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # Second convolution
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        return x


class HandwritingEncoder(nn.Module):
    """Encoder for handwriting VAE.
    
    Attributes:
        input_channels: Number of input channels.
        latent_dim: Dimension of latent space.
        hidden_dims: List of hidden dimensions for each block.
        use_spectral_norm: Whether to use spectral normalization.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 128,
        hidden_dims: List[int] = [32, 64, 128, 256],
        use_spectral_norm: bool = False,
    ):
        """Initialize encoder.
        
        Args:
            input_channels: Number of input channels.
            latent_dim: Dimension of latent space.
            hidden_dims: List of hidden dimensions for each block.
            use_spectral_norm: Whether to use spectral normalization.
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder blocks
        self.blocks = nn.ModuleList()
        
        prev_channels = input_channels
        for hidden_dim in hidden_dims:
            block = EncoderBlock(
                in_channels=prev_channels,
                out_channels=hidden_dim,
                use_spectral_norm=use_spectral_norm,
            )
            self.blocks.append(block)
            prev_channels = hidden_dim
        
        # Calculate final feature map size
        # Assuming input size of 64x256, after 4 downsampling operations: 4x16
        self.final_height = 4
        self.final_width = 16
        
        # Latent projection
        self.mu_projection = nn.Linear(
            hidden_dims[-1] * self.final_height * self.final_width, 
            latent_dim
        )
        self.logvar_projection = nn.Linear(
            hidden_dims[-1] * self.final_height * self.final_width, 
            latent_dim
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space.
        
        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width).
            
        Returns:
            Tuple of (mu, logvar) for latent distribution.
        """
        # Apply encoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Project to latent space
        mu = self.mu_projection(x)
        logvar = self.logvar_projection(x)
        
        return mu, logvar


class HandwritingDecoder(nn.Module):
    """Decoder for handwriting VAE.
    
    Attributes:
        latent_dim: Dimension of latent space.
        text_embedding_dim: Dimension of text embeddings.
        hidden_dims: List of hidden dimensions for each block.
        output_channels: Number of output channels.
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        text_embedding_dim: int = 256,
        hidden_dims: List[int] = [256, 128, 64, 32],
        output_channels: int = 1,
    ):
        """Initialize decoder.
        
        Args:
            latent_dim: Dimension of latent space.
            text_embedding_dim: Dimension of text embeddings.
            hidden_dims: List of hidden dimensions for each block.
            output_channels: Number of output channels.
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dims = hidden_dims
        self.output_channels = output_channels
        
        # Input projection
        input_dim = latent_dim + text_embedding_dim
        self.input_projection = nn.Linear(input_dim, hidden_dims[0] * 4 * 16)
        
        # Decoder blocks
        self.blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims)):
            in_channels = hidden_dims[i]
            out_channels = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else output_channels
            
            block = DecoderBlock(in_channels=in_channels, out_channels=out_channels)
            self.blocks.append(block)
        
        # Final activation
        self.final_activation = nn.Sigmoid()
    
    def forward(self, z: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Decode latent code to handwriting image.
        
        Args:
            z: Latent code of shape (batch_size, latent_dim).
            text_embeddings: Text embeddings of shape (batch_size, text_embedding_dim).
            
        Returns:
            Reconstructed handwriting image of shape (batch_size, output_channels, height, width).
        """
        batch_size = z.size(0)
        
        # Concatenate latent code and text embeddings
        x = torch.cat([z, text_embeddings], dim=1)
        
        # Project to initial feature map
        x = self.input_projection(x)
        x = x.view(batch_size, self.hidden_dims[0], 4, 16)
        
        # Apply decoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Final activation
        x = self.final_activation(x)
        
        return x


class HandwritingVAE(nn.Module):
    """VAE model for handwriting generation.
    
    Attributes:
        encoder: Encoder network.
        decoder: Decoder network.
        text_encoder: Text encoder for text embeddings.
    """
    
    def __init__(self, config: Dict[str, any]):
        """Initialize VAE model.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__()
        
        # Initialize components
        self.encoder = HandwritingEncoder(**config.encoder)
        self.decoder = HandwritingDecoder(**config.decoder)
        
        # Text encoder
        from src.utils.text import TextEncoder
        self.text_encoder = TextEncoder(embedding_dim=config.decoder.text_embedding_dim)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution.
            logvar: Log variance of latent distribution.
            
        Returns:
            Sampled latent code.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor, text: List[str]) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width).
            text: List of text strings.
            
        Returns:
            Dictionary containing reconstruction, mu, logvar, and z.
        """
        # Encode text
        text_embeddings = self.text_encoder(text)
        
        # Encode images
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decoder(z, text_embeddings)
        
        return {
            "reconstruction": reconstruction,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }
    
    def generate(self, text: List[str], num_samples: int = 1) -> torch.Tensor:
        """Generate handwriting from text.
        
        Args:
            text: List of text strings.
            num_samples: Number of samples to generate per text.
            
        Returns:
            Generated handwriting images.
        """
        batch_size = len(text)
        
        # Encode text
        text_embeddings = self.text_encoder(text)
        
        # Sample from prior
        z = torch.randn(batch_size * num_samples, self.encoder.latent_dim, device=text_embeddings.device)
        
        # Repeat text embeddings
        text_embeddings = text_embeddings.repeat_interleave(num_samples, dim=0)
        
        # Decode
        generated_images = self.decoder(z, text_embeddings)
        
        return generated_images
