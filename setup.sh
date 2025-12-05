#!/bin/bash
# Setup script for handwriting generation project

echo "Setting up Handwriting Generation Project..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10+ is required. Found: $python_version"
    exit 1
fi

echo "Python version check passed: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install in development mode
echo "Installing package in development mode..."
pip install -e .

# Create necessary directories
echo "Creating directories..."
mkdir -p data outputs checkpoints logs assets demo_samples

# Run tests
echo "Running tests..."
python -m pytest tests/ -v

echo ""
echo "Setup completed successfully!"
echo ""
echo "To get started:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the example script: python example.py"
echo "3. Start the interactive demo: streamlit run demo/app.py"
echo ""
echo "For more information, see the README.md file."
