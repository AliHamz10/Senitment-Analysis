# How to Run the Application

## Quick Start

### 1. Navigate to Project Directory

```bash
cd "/Users/alihamza/CursorCode Projects/Senitment Analysis"
```

### 2. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 3. Run the Application

```bash
python main.py
```

## Complete Command Sequence

Copy and paste this entire sequence:

```bash
cd "/Users/alihamza/CursorCode Projects/Senitment Analysis" && source venv/bin/activate && python main.py
```

## Step-by-Step (If Virtual Environment Not Activated)

If you're starting fresh:

```bash
# 1. Navigate to project
cd "/Users/alihamza/CursorCode Projects/Senitment Analysis"

# 2. Activate virtual environment
source venv/bin/activate

# 3. Verify Python version (should be 3.12)
python --version

# 4. Run the application
python main.py
```

## Keyboard Controls (While Running)

- **ESC** - Exit application
- **S** - Save screenshot
- **+** or **=** - Increase smoothing
- **-** - Decrease smoothing
- **]** - Increase bounding box speed
- **[** - Decrease bounding box speed
- **d** - Toggle deep learning mode
- **m** - Switch DL backend

## Troubleshooting

If you get import errors:

```bash
# Make sure you're in the project root
pwd

# Verify virtual environment is activated
which python

# Should show: .../venv/bin/python
```

If camera doesn't open:

```bash
# Test camera separately
python tests/test_camera.py
```

## Stop the Application

Press **ESC** in the application window, or press **Ctrl+C** in the terminal.
