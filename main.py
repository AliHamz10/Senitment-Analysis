"""
Application Entry Point
Run this file to start the AI Face Emotion & Persona Overlay application.

Created by Ali Hamza & Zarmeena Jawad
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main import main

if __name__ == '__main__':
    main()

