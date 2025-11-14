# Contributing to AI Face Emotion & Persona Overlay

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/sentiment-analysis.git`
3. Create a virtual environment: `python3.12 -m venv venv`
4. Activate it: `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Create a branch: `git checkout -b feature/your-feature-name`

## Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Write docstrings for all functions and classes
- Keep functions focused and small
- Use meaningful variable names

### Project Structure

```
src/
├── camera/      # Camera management and face detection
├── models/      # Emotion detection models
├── ui/          # UI rendering and overlays
├── utils/       # Configuration and logging utilities
└── main.py      # Main application entry point
```

### Configuration

- All settings should be configurable via `config/config.yaml`
- Don't hardcode values that users might want to change
- Document new configuration options in README.md

### Testing

Before submitting a PR:

1. Test your changes thoroughly
2. Run the application: `python main.py`
3. Test with different camera configurations
4. Verify no linting errors: Check your IDE or use `pylint`

### Commit Messages

Use clear, descriptive commit messages:

- `feat: Add new emotion detection mode`
- `fix: Resolve camera initialization issue on macOS`
- `docs: Update README with new features`
- `refactor: Improve emotion model accuracy`
- `style: Format code according to PEP 8`

## Pull Request Process

1. Update README.md if needed
2. Ensure all tests pass
3. Submit PR with a clear description of changes
4. Respond to any feedback or requested changes

## Areas for Contribution

- **Emotion Detection**: Improve accuracy, add new emotions
- **UI/UX**: Enhance visual effects, improve layout
- **Performance**: Optimize frame processing, reduce latency
- **Documentation**: Improve docs, add examples
- **Testing**: Add unit tests, integration tests
- **Features**: New detection modes, export options, etc.

## Questions?

Open an issue for questions, bug reports, or feature requests.

Thank you for contributing!

