"""Streamlit demo application for handwriting generation."""

import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

from src.models.gan import HandwritingGAN
from src.models.vae import HandwritingVAE
from src.utils.core import get_device, set_seed
from src.utils.sampling import HandwritingSampler
from src.utils.text import clean_text


def load_model(config_path: str, checkpoint_path: str) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    from omegaconf import OmegaConf
    
    config = OmegaConf.load(config_path)
    
    # Determine model type
    model_type = config.model._target_.split(".")[-1].lower()
    
    if model_type == "handwritinggan":
        model = HandwritingGAN(config.model)
    elif model_type == "handwritingvae":
        model = HandwritingVAE(config.model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract model state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove "model." prefix if present
        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                model_state_dict[key[6:]] = value
            else:
                model_state_dict[key] = value
        model.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    return model


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    # Denormalize
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    
    numpy_array = tensor.cpu().numpy()
    numpy_array = (numpy_array * 255).astype(np.uint8)
    
    return Image.fromarray(numpy_array, mode='L')


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Handwriting Generation Demo",
        page_icon="✍️",
        layout="wide"
    )
    
    st.title("✍️ Handwriting Generation Demo")
    st.markdown("Generate realistic handwritten text using deep learning models")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["GAN", "VAE"],
        help="Choose the type of generative model to use"
    )
    
    # Check if model files exist
    config_path = f"configs/model/{model_type.lower()}.yaml"
    checkpoint_path = f"checkpoints/{model_type.lower()}_model.ckpt"
    
    if not os.path.exists(config_path):
        st.error(f"Configuration file not found: {config_path}")
        st.info("Please train a model first using the training script.")
        return
    
    if not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint file not found: {checkpoint_path}")
        st.info("Please train a model first using the training script.")
        return
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            device = get_device()
            model = load_model(config_path, checkpoint_path)
            model.to(device)
            model.eval()
            sampler = HandwritingSampler(model, device)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Configuration")
        
        # Text input
        text_input = st.text_area(
            "Enter text to generate handwriting for:",
            value="Hello, world!",
            height=100,
            help="Enter the text you want to generate handwriting for"
        )
        
        # Generation parameters
        st.subheader("Generation Parameters")
        
        num_samples = st.slider(
            "Number of samples",
            min_value=1,
            max_value=8,
            value=4,
            help="Number of handwriting samples to generate"
        )
        
        guidance_scale = st.slider(
            "Guidance scale",
            min_value=1.0,
            max_value=20.0,
            value=7.5,
            step=0.5,
            help="Controls the strength of text conditioning"
        )
        
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=1000000,
            value=42,
            help="Random seed for reproducible generation"
        )
        
        # Generate button
        generate_button = st.button("Generate Handwriting", type="primary")
    
    with col2:
        st.header("Generated Samples")
        
        if generate_button:
            if not text_input.strip():
                st.warning("Please enter some text to generate handwriting for.")
            else:
                try:
                    with st.spinner("Generating handwriting..."):
                        # Clean text
                        clean_text_input = clean_text(text_input)
                        
                        # Generate samples
                        samples = sampler.sample_with_guidance(
                            text=clean_text_input,
                            guidance_scale=guidance_scale,
                            num_samples=num_samples,
                            seed=seed
                        )
                    
                    # Display samples
                    st.success(f"Generated {num_samples} samples for: '{clean_text_input}'")
                    
                    # Create grid layout
                    cols = min(2, num_samples)
                    rows = (num_samples + cols - 1) // cols
                    
                    for i in range(num_samples):
                        row = i // cols
                        col = i % cols
                        
                        with st.container():
                            # Convert tensor to image
                            sample_image = tensor_to_image(samples[i])
                            
                            # Display image
                            st.image(
                                sample_image,
                                caption=f"Sample {i+1}",
                                use_column_width=True
                            )
                    
                    # Download options
                    st.subheader("Download Options")
                    
                    # Individual samples
                    for i, sample in enumerate(samples):
                        sample_image = tensor_to_image(sample)
                        
                        # Convert to bytes
                        img_buffer = io.BytesIO()
                        sample_image.save(img_buffer, format='PNG')
                        img_bytes = img_buffer.getvalue()
                        
                        st.download_button(
                            label=f"Download Sample {i+1}",
                            data=img_bytes,
                            file_name=f"handwriting_sample_{i+1}.png",
                            mime="image/png"
                        )
                    
                    # Grid image
                    fig = sampler.create_sample_grid(samples, clean_text_input)
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='PNG', dpi=150, bbox_inches='tight')
                    img_bytes = img_buffer.getvalue()
                    
                    st.download_button(
                        label="Download All Samples (Grid)",
                        data=img_bytes,
                        file_name="handwriting_grid.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating samples: {str(e)}")
                    st.info("Please check your input and try again.")
    
    # Additional features
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Model Information")
        st.info(f"""
        **Model Type:** {model_type}
        **Device:** {device}
        **Parameters:** {sum(p.numel() for p in model.parameters()):,}
        """)
    
    with col2:
        st.subheader("🎯 Tips for Better Results")
        st.info("""
        - Use clear, simple text
        - Avoid special characters
        - Try different guidance scales
        - Experiment with different seeds
        """)
    
    with col3:
        st.subheader("🔧 Advanced Features")
        if st.button("Generate Demo Samples"):
            with st.spinner("Generating demo samples..."):
                demo_results = sampler.generate_demo_samples(
                    num_samples=4,
                    save_dir="demo_samples"
                )
            st.success("Demo samples generated! Check the 'demo_samples' directory.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Handwriting Generation Demo** | "
        "Built with PyTorch, Streamlit, and Deep Learning"
    )


if __name__ == "__main__":
    main()
