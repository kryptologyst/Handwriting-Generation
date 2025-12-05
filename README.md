# Handwriting Generation

A comprehensive handwriting generation system using deep learning models including GANs, VAEs, and Diffusion models.

## Features

- **Multiple Model Architectures**: GAN, VAE, and Diffusion models for handwriting generation
- **Text-to-Image Generation**: Generate realistic handwritten text from input strings
- **Modern Tech Stack**: PyTorch 2.0+, PyTorch Lightning, OmegaConf, and more
- **Comprehensive Evaluation**: Multiple metrics including FID, SSIM, PSNR, and diversity measures
- **Interactive Demo**: Streamlit-based web application for easy experimentation
- **Production Ready**: Proper configuration management, logging, and checkpointing

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Handwriting-Generation.git
cd Handwriting-Generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e ".[dev]"
```

### Training

Train a GAN model:
```bash
python src/scripts/train.py --config configs/config.yaml
```

Train a VAE model:
```bash
python src/scripts/train.py --config configs/config.yaml model=vae
```

### Sampling

Generate handwriting samples:
```bash
python src/scripts/sample.py --checkpoint checkpoints/gan_model.ckpt --text "Hello, world!" --num_samples 4
```

### Evaluation

Evaluate model performance:
```bash
python src/scripts/evaluate.py --checkpoint checkpoints/gan_model.ckpt --output_dir evaluation_results
```

### Demo Application

Launch the interactive demo:
```bash
streamlit run demo/app.py
```

## Project Structure

```
handwriting-generation/
├── src/                    # Source code
│   ├── models/             # Model architectures
│   │   ├── gan.py         # GAN implementation
│   │   ├── vae.py         # VAE implementation
│   │   └── diffusion.py   # Diffusion model (optional)
│   ├── data/              # Data handling
│   │   └── dataset.py     # Dataset classes
│   ├── training/          # Training utilities
│   │   └── trainer.py     # PyTorch Lightning modules
│   ├── evaluation/        # Evaluation metrics
│   │   └── evaluator.py   # Evaluation utilities
│   ├── utils/             # Utility functions
│   │   ├── core.py        # Core utilities
│   │   ├── text.py        # Text processing
│   │   └── sampling.py    # Sampling utilities
│   └── scripts/           # Command-line scripts
│       ├── train.py       # Training script
│       ├── sample.py      # Sampling script
│       └── evaluate.py    # Evaluation script
├── configs/               # Configuration files
│   ├── config.yaml        # Main configuration
│   └── model/             # Model-specific configs
│       ├── gan.yaml       # GAN configuration
│       ├── vae.yaml       # VAE configuration
│       └── diffusion.yaml # Diffusion configuration
├── demo/                  # Demo application
│   └── app.py            # Streamlit app
├── tests/                 # Unit tests
├── assets/               # Generated samples and assets
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
└── requirements.txt      # Dependencies
```

## Configuration

The project uses OmegaConf for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration with global settings
- `configs/model/gan.yaml`: GAN-specific configuration
- `configs/model/vae.yaml`: VAE-specific configuration

### Key Configuration Options

```yaml
# Global settings
seed: 42
device: auto  # auto, cpu, cuda, mps
precision: 16-mixed

# Training
batch_size: 32
max_epochs: 100
learning_rate: 0.0002

# Model architecture
model:
  _target_: src.models.gan.HandwritingGAN
  generator:
    latent_dim: 100
    text_embedding_dim: 256
    hidden_dims: [512, 256, 128]
```

## Models

### GAN (Generative Adversarial Network)

The GAN implementation includes:
- Generator with upsampling blocks and self-attention
- Discriminator with downsampling and spectral normalization
- Multiple loss functions (hinge, NS, WGAN-GP)
- EMA (Exponential Moving Average) for stable training

### VAE (Variational Autoencoder)

The VAE implementation includes:
- Encoder with downsampling blocks
- Decoder with upsampling blocks
- Reparameterization trick for latent sampling
- KL annealing for better training dynamics

### Text Processing

- Pre-trained text encoder (DistilBERT)
- Text cleaning and normalization
- Embedding projection to model dimensions

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Reconstruction Metrics**: MSE, MAE, SSIM, PSNR
- **Diversity Metrics**: Pairwise distance analysis
- **Generation Quality**: Visual inspection and comparison grids

## Data

The project includes a synthetic handwriting dataset generator that creates training data using PIL. This allows the system to run without external datasets.

For production use, you can replace this with real handwriting datasets by implementing custom dataset classes.

## Development

### Code Quality

The project uses:
- Black for code formatting
- Ruff for linting
- Pytest for testing
- Pre-commit hooks for quality assurance

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
ruff check src/
```

## Advanced Features

### Mixed Precision Training

The system supports automatic mixed precision training for faster training and reduced memory usage.

### Device Support

Automatic device detection supports:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU fallback

### Logging and Monitoring

Integration with:
- Weights & Biases (wandb)
- TensorBoard
- MLflow

## Limitations and Considerations

- **Synthetic Data**: The current implementation uses synthetic data. For production use, real handwriting datasets are recommended.
- **Model Size**: Models are designed for demonstration purposes. Larger models may be needed for production-quality results.
- **Text Length**: Current models work best with shorter text inputs (up to 200 characters).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{handwriting_generation,
  title={Handwriting Generation: A Modern Deep Learning Approach},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Handwriting-Generation}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- PyTorch Lightning for simplifying training workflows
- The open-source community for various components and inspiration
# Handwriting-Generation
