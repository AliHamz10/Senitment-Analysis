"""
Legacy app runner - Use main.py instead.
Created by Ali Hamza & Zarmeena Jawad
"""
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.main import main
    print('Starting main()')
    main()
except Exception:
    traceback.print_exc()
    print('ERROR launching main')
