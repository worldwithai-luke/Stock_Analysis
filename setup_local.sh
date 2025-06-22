#!/bin/bash

echo "Stock Analysis Dashboard - Linux/macOS Setup"
echo "============================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.11+ and try again"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.11"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)"; then
    echo "Error: Python 3.11+ required. Current version: $python_version"
    echo "Please upgrade Python and try again"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r local_requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "✅ Setup complete! To run the application:"
echo "1. Run: source venv/bin/activate"
echo "2. Run: streamlit run app.py"
echo ""
echo "Or simply run: python3 run_local.py"

# Make the script executable
chmod +x run_local.py