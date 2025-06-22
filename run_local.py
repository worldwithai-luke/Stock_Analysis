#!/usr/bin/env python3
"""
Local development runner for the Stock Analysis Dashboard
This script helps you run the application on your local computer
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.11 or higher"""
    if sys.version_info < (3, 11):
        print(f"❌ Python 3.11+ required. Current version: {sys.version}")
        print("Please upgrade Python and try again.")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "local_requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("Make sure you have an active internet connection and try again.")
        return False

def check_dependencies():
    """Check if all required packages are available"""
    required_packages = [
        'streamlit', 'yfinance', 'plotly', 'pandas', 'numpy', 'sklearn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are available")
    return True

def run_streamlit():
    """Run the Streamlit application"""
    print("🚀 Starting Stock Analysis Dashboard...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")

def main():
    """Main function to run the local setup"""
    print("🏠 Stock Analysis Dashboard - Local Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ app.py not found in current directory")
        print("Please navigate to the project folder and try again.")
        return
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check dependencies
    if not check_dependencies():
        print("\n🔧 Installing missing dependencies...")
        if not install_dependencies():
            return
    
    # Run the application
    print("\n" + "=" * 50)
    run_streamlit()

if __name__ == "__main__":
    main()