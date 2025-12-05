"""Core utilities for handwriting generation project."""

import os
import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ('auto', 'cpu', 'cuda', 'mps').
        
    Returns:
        PyTorch device object.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save.
        save_path: Path where to save the configuration.
    """
    OmegaConf.save(config, save_path)


def create_directories(config: DictConfig) -> None:
    """Create necessary directories from configuration.
    
    Args:
        config: Configuration object containing directory paths.
    """
    directories = [
        config.data_dir,
        config.output_dir,
        config.checkpoint_dir,
        config.log_dir,
        config.assets_dir,
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> str:
    """Get human-readable model size.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Model size as a string (e.g., "1.2M", "45.6M").
    """
    num_params = count_parameters(model)
    
    if num_params >= 1e9:
        return f"{num_params / 1e9:.1f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.1f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.1f}K"
    else:
        return str(num_params)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to NumPy array.
    
    Args:
        tensor: PyTorch tensor.
        
    Returns:
        NumPy array.
    """
    return tensor.detach().cpu().numpy()


def to_tensor(array: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert NumPy array to PyTorch tensor.
    
    Args:
        array: NumPy array.
        device: Target device for the tensor.
        
    Returns:
        PyTorch tensor.
    """
    tensor = torch.from_numpy(array)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def denormalize_image(
    tensor: torch.Tensor, 
    mean: float = 0.5, 
    std: float = 0.5
) -> torch.Tensor:
    """Denormalize image tensor.
    
    Args:
        tensor: Normalized image tensor.
        mean: Mean used for normalization.
        std: Standard deviation used for normalization.
        
    Returns:
        Denormalized image tensor.
    """
    return tensor * std + mean


def normalize_image(
    tensor: torch.Tensor, 
    mean: float = 0.5, 
    std: float = 0.5
) -> torch.Tensor:
    """Normalize image tensor.
    
    Args:
        tensor: Image tensor.
        mean: Mean for normalization.
        std: Standard deviation for normalization.
        
    Returns:
        Normalized image tensor.
    """
    return (tensor - mean) / std


class EarlyStopping:
    """Early stopping utility to prevent overfitting.
    
    Attributes:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as an improvement.
        best_score: Best score seen so far.
        counter: Number of epochs without improvement.
        early_stop: Whether to stop training.
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score.
            
        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop
