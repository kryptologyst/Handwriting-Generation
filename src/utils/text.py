"""Text processing utilities for handwriting generation."""

import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    """Text encoder for converting text to embeddings.
    
    Attributes:
        tokenizer: Tokenizer for text processing.
        model: Pre-trained language model.
        embedding_dim: Dimension of output embeddings.
    """
    
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased",
        embedding_dim: int = 256,
        freeze_model: bool = True
    ):
        """Initialize text encoder.
        
        Args:
            model_name: Name of pre-trained model to use.
            embedding_dim: Dimension of output embeddings.
            freeze_model: Whether to freeze the pre-trained model.
        """
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Projection layer to match desired embedding dimension
        self.projection = nn.Linear(
            self.model.config.hidden_size, 
            embedding_dim
        )
    
    def forward(self, text: List[str]) -> torch.Tensor:
        """Encode text to embeddings.
        
        Args:
            text: List of text strings.
            
        Returns:
            Text embeddings tensor of shape (batch_size, embedding_dim).
        """
        # Tokenize text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Move inputs to same device as model
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings from pre-trained model
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use [CLS] token representation
        embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Project to desired dimension
        embeddings = self.projection(embeddings)
        
        return embeddings


def clean_text(text: str) -> str:
    """Clean and normalize text for handwriting generation.
    
    Args:
        text: Input text string.
        
    Returns:
        Cleaned text string.
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might not render well
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
    
    # Limit length
    if len(text) > 200:
        text = text[:200] + "..."
    
    return text


def split_text_into_lines(text: str, max_chars_per_line: int = 50) -> List[str]:
    """Split text into lines for better handwriting generation.
    
    Args:
        text: Input text string.
        max_chars_per_line: Maximum characters per line.
        
    Returns:
        List of text lines.
    """
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line + " " + word) <= max_chars_per_line:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                lines.append(word)
    
    if current_line:
        lines.append(current_line)
    
    return lines


def create_text_prompts() -> List[str]:
    """Create a set of example text prompts for handwriting generation.
    
    Returns:
        List of example text prompts.
    """
    prompts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Handwriting generation is fascinating.",
        "Machine learning and AI are transforming our world.",
        "Beautiful handwritten text can be generated automatically.",
        "This is a test of the handwriting generation system.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "The future of AI is bright and full of possibilities.",
        "Handwriting synthesis using deep learning models.",
        "Generative models can create realistic handwritten text.",
    ]
    
    return prompts


def get_text_length_stats(texts: List[str]) -> dict:
    """Get statistics about text lengths.
    
    Args:
        texts: List of text strings.
        
    Returns:
        Dictionary containing text length statistics.
    """
    lengths = [len(text) for text in texts]
    
    return {
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": sum(lengths) / len(lengths),
        "median_length": sorted(lengths)[len(lengths) // 2],
        "total_texts": len(texts),
    }
