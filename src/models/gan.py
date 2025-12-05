"""GAN model for handwriting generation."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SelfAttention(nn.Module):
    """Self-attention layer for GAN models.
    
    Attributes:
        in_channels: Number of input channels.
        attention_dim: Dimension of attention space.
    """
    
    def __init__(self, in_channels: int, attention_dim: int = 64):
        """Initialize self-attention layer.
        
        Args:
            in_channels: Number of input channels.
            attention_dim: Dimension of attention space.
        """
        super().__init__()
        self.in_channels = in_channels
        self.attention_dim = attention_dim
        
        self.query_conv = nn.Conv2d(in_channels, attention_dim, 1)
        self.key_conv = nn.Conv2d(in_channels, attention_dim, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            Output tensor with self-attention applied.
        """
        batch_size, channels, height, width = x.size()
        
        # Compute attention
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x


class GeneratorBlock(nn.Module):
    """Generator block with upsampling and optional self-attention.
    
    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        use_spectral_norm: Whether to use spectral normalization.
        use_self_attention: Whether to use self-attention.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_spectral_norm: bool = True,
        use_self_attention: bool = False,
    ):
        """Initialize generator block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            use_spectral_norm: Whether to use spectral normalization.
            use_self_attention: Whether to use self-attention.
        """
        super().__init__()
        
        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
        # Convolution layers
        conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if use_spectral_norm:
            conv1 = spectral_norm(conv1)
            conv2 = spectral_norm(conv2)
        
        self.conv1 = conv1
        self.conv2 = conv2
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Self-attention
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.attention = SelfAttention(out_channels)
        
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
        
        # Self-attention
        if self.use_self_attention:
            x = self.attention(x)
        
        x = self.activation(x)
        
        return x


class HandwritingGenerator(nn.Module):
    """Generator for handwriting generation.
    
    Attributes:
        latent_dim: Dimension of latent noise.
        text_embedding_dim: Dimension of text embeddings.
        hidden_dims: List of hidden dimensions for each block.
        output_channels: Number of output channels.
        use_spectral_norm: Whether to use spectral normalization.
        use_self_attention: Whether to use self-attention.
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        text_embedding_dim: int = 256,
        hidden_dims: List[int] = [512, 256, 128],
        output_channels: int = 1,
        use_spectral_norm: bool = True,
        use_self_attention: bool = True,
    ):
        """Initialize generator.
        
        Args:
            latent_dim: Dimension of latent noise.
            text_embedding_dim: Dimension of text embeddings.
            hidden_dims: List of hidden dimensions for each block.
            output_channels: Number of output channels.
            use_spectral_norm: Whether to use spectral normalization.
            use_self_attention: Whether to use self-attention.
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dims = hidden_dims
        self.output_channels = output_channels
        
        # Input projection
        input_dim = latent_dim + text_embedding_dim
        self.input_projection = nn.Linear(input_dim, hidden_dims[0] * 4 * 8)
        
        # Generator blocks
        self.blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims)):
            in_channels = hidden_dims[i]
            out_channels = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else output_channels
            
            block = GeneratorBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                use_spectral_norm=use_spectral_norm,
                use_self_attention=use_self_attention and i == len(hidden_dims) - 2,
            )
            self.blocks.append(block)
        
        # Final activation
        self.final_activation = nn.Tanh()
    
    def forward(self, noise: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Generate handwriting from noise and text.
        
        Args:
            noise: Random noise tensor of shape (batch_size, latent_dim).
            text_embeddings: Text embeddings of shape (batch_size, text_embedding_dim).
            
        Returns:
            Generated handwriting image of shape (batch_size, output_channels, height, width).
        """
        batch_size = noise.size(0)
        
        # Concatenate noise and text embeddings
        x = torch.cat([noise, text_embeddings], dim=1)
        
        # Project to initial feature map
        x = self.input_projection(x)
        x = x.view(batch_size, self.hidden_dims[0], 4, 8)
        
        # Apply generator blocks
        for block in self.blocks:
            x = block(x)
        
        # Final activation
        x = self.final_activation(x)
        
        return x


class DiscriminatorBlock(nn.Module):
    """Discriminator block with downsampling and optional self-attention.
    
    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        use_spectral_norm: Whether to use spectral normalization.
        use_self_attention: Whether to use self-attention.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_spectral_norm: bool = True,
        use_self_attention: bool = False,
    ):
        """Initialize discriminator block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            use_spectral_norm: Whether to use spectral normalization.
            use_self_attention: Whether to use self-attention.
        """
        super().__init__()
        
        # Convolution layers
        conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if use_spectral_norm:
            conv1 = spectral_norm(conv1)
            conv2 = spectral_norm(conv2)
        
        self.conv1 = conv1
        self.conv2 = conv2
        
        # Downsampling
        self.downsample = nn.AvgPool2d(2)
        
        # Self-attention
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.attention = SelfAttention(out_channels)
        
        # Activation
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        # First convolution
        x = self.conv1(x)
        x = self.activation(x)
        
        # Second convolution
        x = self.conv2(x)
        x = self.activation(x)
        
        # Self-attention
        if self.use_self_attention:
            x = self.attention(x)
        
        # Downsample
        x = self.downsample(x)
        
        return x


class HandwritingDiscriminator(nn.Module):
    """Discriminator for handwriting generation.
    
    Attributes:
        input_channels: Number of input channels.
        text_embedding_dim: Dimension of text embeddings.
        hidden_dims: List of hidden dimensions for each block.
        use_spectral_norm: Whether to use spectral normalization.
        use_self_attention: Whether to use self-attention.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        text_embedding_dim: int = 256,
        hidden_dims: List[int] = [128, 256, 512],
        use_spectral_norm: bool = True,
        use_self_attention: bool = True,
    ):
        """Initialize discriminator.
        
        Args:
            input_channels: Number of input channels.
            text_embedding_dim: Dimension of text embeddings.
            hidden_dims: List of hidden dimensions for each block.
            use_spectral_norm: Whether to use spectral normalization.
            use_self_attention: Whether to use self-attention.
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dims = hidden_dims
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(input_channels, hidden_dims[0], 3, padding=1)
        if use_spectral_norm:
            self.initial_conv = spectral_norm(self.initial_conv)
        
        # Discriminator blocks
        self.blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims)):
            in_channels = hidden_dims[i]
            out_channels = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else hidden_dims[i]
            
            block = DiscriminatorBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                use_spectral_norm=use_spectral_norm,
                use_self_attention=use_self_attention and i == 1,
            )
            self.blocks.append(block)
        
        # Text conditioning
        self.text_projection = nn.Linear(text_embedding_dim, hidden_dims[-1])
        
        # Final classification
        self.final_conv = nn.Conv2d(hidden_dims[-1] * 2, 1, 1)
        if use_spectral_norm:
            self.final_conv = spectral_norm(self.final_conv)
    
    def forward(self, x: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Discriminate real vs fake handwriting.
        
        Args:
            x: Input image tensor of shape (batch_size, input_channels, height, width).
            text_embeddings: Text embeddings of shape (batch_size, text_embedding_dim).
            
        Returns:
            Discriminator output of shape (batch_size, 1, height, width).
        """
        # Initial convolution
        x = self.initial_conv(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        
        # Apply discriminator blocks
        for block in self.blocks:
            x = block(x)
        
        # Text conditioning
        text_features = self.text_projection(text_embeddings)
        text_features = text_features.view(text_features.size(0), text_features.size(1), 1, 1)
        text_features = text_features.expand_as(x)
        
        # Concatenate image and text features
        x = torch.cat([x, text_features], dim=1)
        
        # Final classification
        x = self.final_conv(x)
        
        return x


class HandwritingGAN(nn.Module):
    """Complete GAN model for handwriting generation.
    
    Attributes:
        generator: Generator network.
        discriminator: Discriminator network.
        text_encoder: Text encoder for text embeddings.
    """
    
    def __init__(self, config: Dict[str, any]):
        """Initialize GAN model.
        
        Args:
            config: Configuration dictionary.
        """
        super().__init__()
        
        # Initialize components
        self.generator = HandwritingGenerator(**config.generator)
        self.discriminator = HandwritingDiscriminator(**config.discriminator)
        
        # Text encoder
        from src.utils.text import TextEncoder
        self.text_encoder = TextEncoder(embedding_dim=config.generator.text_embedding_dim)
    
    def forward(self, noise: torch.Tensor, text: List[str]) -> torch.Tensor:
        """Generate handwriting from noise and text.
        
        Args:
            noise: Random noise tensor.
            text: List of text strings.
            
        Returns:
            Generated handwriting images.
        """
        # Encode text
        text_embeddings = self.text_encoder(text)
        
        # Generate images
        generated_images = self.generator(noise, text_embeddings)
        
        return generated_images
