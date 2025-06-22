#!/usr/bin/env python3
"""
Quick start script for local development
Simplified version that just runs the app
"""

import subprocess
import sys
import webbrowser
import time
from threading import Timer

def open_browser():
    """Open browser after a short delay"""
    print("🌐 Opening browser...")
    webbrowser.open('http://localhost:8501')

def main():
    print("🚀 Starting Stock Analysis Dashboard...")
    
    # Start browser opening in 3 seconds
    timer = Timer(3.0, open_browser)
    timer.start()
    
    try:
        # Run streamlit with local settings
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.serverAddress", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install dependencies first:")
        print("   pip install -r local_requirements.txt")

if __name__ == "__main__":
    main()