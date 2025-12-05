"""Tests for handwriting generation models."""

import pytest
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.models.gan import HandwritingGAN, HandwritingGenerator, HandwritingDiscriminator
from src.models.vae import HandwritingVAE, HandwritingEncoder, HandwritingDecoder
from src.utils.core import set_seed, get_device, count_parameters
from src.utils.text import clean_text, TextEncoder


class TestGAN:
    """Test GAN model components."""
    
    def test_generator_forward(self):
        """Test generator forward pass."""
        generator = HandwritingGenerator(
            latent_dim=100,
            text_embedding_dim=256,
            hidden_dims=[512, 256, 128],
            output_channels=1
        )
        
        batch_size = 4
        noise = torch.randn(batch_size, 100)
        text_embeddings = torch.randn(batch_size, 256)
        
        output = generator(noise, text_embeddings)
        
        assert output.shape == (batch_size, 1, 64, 256)
        assert output.min() >= -1.0
        assert output.max() <= 1.0
    
    def test_discriminator_forward(self):
        """Test discriminator forward pass."""
        discriminator = HandwritingDiscriminator(
            input_channels=1,
            text_embedding_dim=256,
            hidden_dims=[128, 256, 512]
        )
        
        batch_size = 4
        images = torch.randn(batch_size, 1, 64, 256)
        text_embeddings = torch.randn(batch_size, 256)
        
        output = discriminator(images, text_embeddings)
        
        assert output.shape == (batch_size, 1, 4, 16)
    
    def test_gan_forward(self):
        """Test complete GAN forward pass."""
        config = OmegaConf.create({
            "generator": {
                "latent_dim": 100,
                "text_embedding_dim": 256,
                "hidden_dims": [512, 256, 128],
                "output_channels": 1,
                "use_spectral_norm": True,
                "use_self_attention": True
            },
            "discriminator": {
                "input_channels": 1,
                "text_embedding_dim": 256,
                "hidden_dims": [128, 256, 512],
                "use_spectral_norm": True,
                "use_self_attention": True
            }
        })
        
        model = HandwritingGAN(config)
        
        batch_size = 2
        noise = torch.randn(batch_size, 100)
        text = ["Hello", "World"]
        
        output = model(noise, text)
        
        assert output.shape == (batch_size, 1, 64, 256)


class TestVAE:
    """Test VAE model components."""
    
    def test_encoder_forward(self):
        """Test encoder forward pass."""
        encoder = HandwritingEncoder(
            input_channels=1,
            latent_dim=128,
            hidden_dims=[32, 64, 128, 256]
        )
        
        batch_size = 4
        images = torch.randn(batch_size, 1, 64, 256)
        
        mu, logvar = encoder(images)
        
        assert mu.shape == (batch_size, 128)
        assert logvar.shape == (batch_size, 128)
    
    def test_decoder_forward(self):
        """Test decoder forward pass."""
        decoder = HandwritingDecoder(
            latent_dim=128,
            text_embedding_dim=256,
            hidden_dims=[256, 128, 64, 32],
            output_channels=1
        )
        
        batch_size = 4
        z = torch.randn(batch_size, 128)
        text_embeddings = torch.randn(batch_size, 256)
        
        output = decoder(z, text_embeddings)
        
        assert output.shape == (batch_size, 1, 64, 256)
        assert output.min() >= 0.0
        assert output.max() <= 1.0
    
    def test_vae_forward(self):
        """Test complete VAE forward pass."""
        config = OmegaConf.create({
            "encoder": {
                "input_channels": 1,
                "latent_dim": 128,
                "hidden_dims": [32, 64, 128, 256],
                "use_spectral_norm": False
            },
            "decoder": {
                "latent_dim": 128,
                "text_embedding_dim": 256,
                "hidden_dims": [256, 128, 64, 32],
                "output_channels": 1
            }
        })
        
        model = HandwritingVAE(config)
        
        batch_size = 2
        images = torch.randn(batch_size, 1, 64, 256)
        text = ["Hello", "World"]
        
        outputs = model(images, text)
        
        assert "reconstruction" in outputs
        assert "mu" in outputs
        assert "logvar" in outputs
        assert "z" in outputs
        
        assert outputs["reconstruction"].shape == (batch_size, 1, 64, 256)
        assert outputs["mu"].shape == (batch_size, 128)
        assert outputs["logvar"].shape == (batch_size, 128)
        assert outputs["z"].shape == (batch_size, 128)
    
    def test_vae_generate(self):
        """Test VAE generation."""
        config = OmegaConf.create({
            "encoder": {
                "input_channels": 1,
                "latent_dim": 128,
                "hidden_dims": [32, 64, 128, 256],
                "use_spectral_norm": False
            },
            "decoder": {
                "latent_dim": 128,
                "text_embedding_dim": 256,
                "hidden_dims": [256, 128, 64, 32],
                "output_channels": 1
            }
        })
        
        model = HandwritingVAE(config)
        
        text = ["Hello", "World"]
        num_samples = 3
        
        generated = model.generate(text, num_samples)
        
        assert generated.shape == (len(text) * num_samples, 1, 64, 256)


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        rand1 = torch.randn(10)
        
        set_seed(42)
        rand2 = torch.randn(10)
        
        assert torch.allclose(rand1, rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        num_params = count_parameters(model)
        assert num_params == 55  # 10*5 + 5 bias
    
    def test_clean_text(self):
        """Test text cleaning."""
        dirty_text = "  Hello,   world!  \n\t  "
        clean = clean_text(dirty_text)
        assert clean == "Hello, world!"
        
        # Test special character removal
        special_text = "Hello@#$%^&*()world"
        clean_special = clean_text(special_text)
        assert clean_special == "Helloworld"
    
    def test_text_encoder(self):
        """Test text encoder."""
        encoder = TextEncoder(embedding_dim=128)
        
        texts = ["Hello", "World"]
        embeddings = encoder(texts)
        
        assert embeddings.shape == (2, 128)
        assert not torch.isnan(embeddings).any()


class TestData:
    """Test data loading and processing."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        from src.data.dataset import SyntheticHandwritingDataset
        
        texts = ["Hello", "World"]
        dataset = SyntheticHandwritingDataset(
            texts=texts,
            image_size=(64, 256),
            font_size=24,
            num_samples=10
        )
        
        assert len(dataset) == 10
        
        sample = dataset[0]
        assert "text" in sample
        assert "image" in sample
        assert "index" in sample
        
        assert sample["image"].size == (256, 64)  # PIL Image size is (width, height)
        assert sample["text"] in texts


if __name__ == "__main__":
    pytest.main([__file__])
