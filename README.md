# AI Face Emotion Detection System

A real-time computer vision application that analyzes facial expressions through webcam input and displays detected emotions with a modern cyberpunk-inspired interface overlay.

## Overview

This application combines MediaPipe face detection with advanced emotion recognition algorithms to provide real-time emotion analysis. The system processes video frames from your webcam, detects facial landmarks, and classifies emotions using both heuristic-based analysis and optional deep learning models.

## Key Features

### Real-Time Face Detection
- MediaPipe Face Mesh integration for accurate facial landmark detection
- Automatic face tracking and bounding box generation
- Support for multiple face detection in frame

### Advanced Emotion Recognition
- Multiple detection modes: fast heuristic analysis or deep learning inference
- Improved accuracy through enhanced feature extraction:
  - Eye Aspect Ratio (EAR) calculation using 6-point per-eye analysis
  - Mouth Aspect Ratio (MAR) for mouth opening detection
  - Eyebrow position analysis for emotional context
  - Facial symmetry detection for complex emotions
- Support for multiple emotion categories: happy, sad, angry, surprised, excited, confused, fear, disgust, neutral

### Smart Camera Management
- Automatic camera device selection and initialization
- Prioritizes built-in Mac camera over external devices
- Intelligent Continuity Camera handling (skips iPhone cameras by default)
- Robust error handling and camera warm-up sequences

### Modern User Interface
- Cyberpunk-inspired HUD overlay with neon effects
- Multi-layer glow effects and smooth animations
- Real-time confidence visualization with gradient bars
- Animated scanline effects and glitch text rendering
- Configurable color schemes and panel layouts

### Configuration System
- YAML-based configuration for all application settings
- Runtime adjustable parameters
- Separate configuration files for different environments
- Easy customization without code changes

## System Requirements

- Operating System: macOS (optimized), Windows, or Linux
- Python: 3.11 or higher (3.12 recommended)
- Camera: Built-in webcam or external USB camera supported by OpenCV
- RAM: 4GB minimum (8GB recommended for deep learning modes)
- Storage: 2GB free space for dependencies and models

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd sentiment-analysis
```

### Step 2: Create Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Note: First-time installation may take several minutes as it downloads large dependencies including PyTorch, MediaPipe, and other machine learning libraries.

### Step 4: Verify Installation

```bash
python -c "from src.camera import CameraManager; from src.models import EmotionModel; print('Installation successful')"
```

## Quick Start

### Running the Application

```bash
python main.py
```

On first launch:
1. Grant camera permissions when prompted by your operating system
2. The application will automatically detect and initialize your camera
3. Emotion detection models will load (may take a few seconds)
4. The webcam feed with emotion overlay will appear

### Basic Usage

- Position yourself in front of the camera
- The application will automatically detect your face
- Emotion labels will appear in real-time with confidence scores
- Press ESC to exit the application

## Configuration

All settings can be customized in `config/config.yaml`. Key configuration sections:

### Camera Settings

```yaml
camera:
  prefer_builtin: true      # Prefer built-in camera
  skip_continuity: true     # Skip iPhone Continuity Camera
  width: 1280               # Camera resolution width
  height: 720                # Camera resolution height
  fps: 30                   # Target frames per second
```

### Model Settings

```yaml
model:
  mode: image               # Detection mode: 'image', 'text', 'hybrid', or 'dl'
  dl_backend: onnx         # Deep learning backend: 'onnx' or 'deepface'
  recent_decay: 0.85        # Smoothing factor for emotion transitions
  confidence_threshold: 0.3  # Minimum confidence to display emotion
```

### UI Settings

```yaml
ui:
  colors:
    primary: [0, 255, 255]   # Cyan for primary elements
    secondary: [255, 0, 255] # Magenta for secondary elements
    accent: [0, 255, 0]     # Green for accent elements
  positions:
    emotion_label: [50, 100] # Position of emotion label
    fps_counter: [50, 50]    # Position of FPS counter
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| ESC | Exit application |
| S | Save screenshot to `screenshots/` directory |
| + or = | Increase label smoothing (less jittery) |
| - | Decrease label smoothing (more reactive) |
| ] | Increase bounding box follow speed |
| [ | Decrease bounding box follow speed |
| d | Toggle deep learning mode ON/OFF |
| m | Switch DL backend (ONNX ↔ DeepFace) |

## Project Structure

```
.
├── src/                    # Main application source code
│   ├── camera/            # Camera management and face detection
│   │   ├── camera_manager.py
│   │   └── face_detector.py
│   ├── models/            # Emotion detection models
│   │   └── emotion_model.py
│   ├── ui/                # UI rendering and overlays
│   │   └── overlay_utils.py
│   ├── utils/             # Utilities
│   │   ├── config.py      # Configuration management
│   │   └── logger.py      # Logging system
│   └── main.py            # Main application logic
├── config/                # Configuration files
│   └── config.yaml        # Application settings
├── ai_face_persona/       # Assets and legacy files
│   ├── assets/           # Fonts, images, model files
│   └── legacy/           # Legacy code (reference only)
├── tests/                 # Test utilities
│   └── test_camera.py    # Camera test script
├── main.py               # Application entry point
└── requirements.txt      # Python dependencies
```

## Architecture

### Camera System

The `CameraManager` class handles all camera operations:
- Device enumeration and selection
- Automatic camera initialization with fallback options
- Frame capture and validation
- Error recovery and troubleshooting

### Emotion Detection System

The `EmotionModel` class provides multiple detection strategies:

1. **Image Mode (Default)**: Fast heuristic-based analysis using facial landmarks
   - Calculates geometric features from MediaPipe landmarks
   - Uses thresholds and scoring algorithms
   - Low latency, high performance

2. **Text Mode**: Uses Hugging Face text classifier
   - Converts facial features to text descriptions
   - Classifies using transformer models
   - More accurate but slower

3. **Hybrid Mode**: Combines image and text analysis
   - Best of both approaches
   - Balanced accuracy and performance

4. **Deep Learning Mode**: Uses ONNX or DeepFace models
   - Highest accuracy
   - Requires model files
   - Higher computational cost

### UI Rendering System

The overlay system provides:
- Real-time HUD rendering with OpenCV
- Smooth animations and transitions
- Configurable visual effects
- Performance-optimized drawing operations

## Troubleshooting

### Camera Not Opening

1. Check system camera permissions:
   - macOS: System Settings > Privacy & Security > Camera
   - Windows: Settings > Privacy > Camera
   - Linux: Check v4l2 permissions

2. Ensure no other applications are using the camera:
   - Close Zoom, FaceTime, or other video applications
   - Check for background processes

3. Test camera access:
   ```bash
   python tests/test_camera.py
   ```

### Wrong Camera Selected

1. Check `config/config.yaml`:
   ```yaml
   camera:
     prefer_builtin: true
     skip_continuity: true
   ```

2. The application logs available cameras on startup
   - Check the console output for camera enumeration
   - Adjust camera index if needed

### Low Performance

1. Reduce camera resolution in `config/config.yaml`:
   ```yaml
   camera:
     width: 640
     height: 480
   ```

2. Disable deep learning mode (press `d` key)

3. Use `image` mode instead of `dl` mode

4. Close other resource-intensive applications

### Import Errors

1. Ensure virtual environment is activated:
   ```bash
   source venv/bin/activate
   ```

2. Verify dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. Check Python version:
   ```bash
   python --version  # Should be 3.11 or higher
   ```

### Model Download Issues

If ONNX or DeepFace models fail to download:
1. Check internet connection
2. Models are downloaded automatically on first use
3. Model files are cached in `ai_face_persona/assets/`
4. Delete cached models to force re-download

## Development

### Running Tests

```bash
# Test camera functionality
python tests/test_camera.py

# Verify imports
python -c "from src.camera import CameraManager; from src.models import EmotionModel; print('OK')"
```

### Code Style

The project follows PEP 8 style guidelines:
- Type hints for all function signatures
- Docstrings for all classes and functions
- Clear variable naming
- Modular architecture

### Adding New Features

1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement changes following existing code patterns
3. Update configuration if needed
4. Test thoroughly
5. Submit pull request

## Performance Optimization

### For Better FPS

1. Use `image` mode instead of `dl` mode
2. Reduce camera resolution
3. Disable unnecessary UI effects
4. Use ONNX instead of DeepFace (faster)

### For Better Accuracy

1. Use `hybrid` or `dl` mode
2. Increase camera resolution
3. Ensure good lighting conditions
4. Position face clearly in frame

## Technical Details

### Emotion Detection Algorithm

The heuristic-based emotion detection uses:

1. **Eye Aspect Ratio (EAR)**: Measures eye openness
   - Calculated using 6 landmark points per eye
   - Lower EAR indicates closed/squinting eyes
   - Higher EAR indicates wide-open eyes

2. **Mouth Aspect Ratio (MAR)**: Measures mouth opening
   - Vertical and horizontal mouth dimensions
   - Indicates smiling, talking, or surprise

3. **Eyebrow Position**: Relative to eye position
   - Raised eyebrows indicate surprise
   - Lowered eyebrows indicate anger or concentration

4. **Facial Asymmetry**: Left-right differences
   - Asymmetry can indicate confusion or complex emotions
   - Calculated from landmark positions

### Smoothing Algorithm

Emotion transitions are smoothed using exponential decay:
- Recent emotions weighted more heavily
- Prevents rapid label switching
- Configurable decay factor in config

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Authors

Ali Hamza & Zarmeena Jawad

## Acknowledgments

- MediaPipe team for face detection technology
- OpenCV community for computer vision tools
- Hugging Face for transformer models
- All contributors and users of this project

## Support

For issues, questions, or contributions:
- Check existing issues in the repository
- Review the configuration documentation
- Test with `tests/test_camera.py` for camera issues
- Check logs in `logs/app.log` for detailed error messages

---

For detailed contribution guidelines, see CONTRIBUTING.md
