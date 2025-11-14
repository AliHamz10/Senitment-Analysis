"""
Import test - Legacy test file.
Created by Ali Hamza & Zarmeena Jawad
"""
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.camera import face_detector
    print('face_detector imported')
    from src.ui import overlay_utils
    print('overlay_utils imported')
    from src.models import emotion_model
    print('emotion_model imported')
    print('All imports OK')
except Exception:
    traceback.print_exc()
    print('Import test failed')
