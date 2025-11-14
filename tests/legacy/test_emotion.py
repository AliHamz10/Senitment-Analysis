"""
Test emotion model - Legacy test file.
Created by Ali Hamza & Zarmeena Jawad
"""
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.models.emotion_model import EmotionModel
    print('Imported emotion_model')
    em = EmotionModel(mode='image')
    print('Created instance')
    res = em.predict([], (480, 640))
    print('RESULT:', res)
except Exception as e:
    traceback.print_exc()
    print('ERROR:', e)
