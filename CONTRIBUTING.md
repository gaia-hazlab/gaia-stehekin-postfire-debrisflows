# Contributing to GAIA Stehekin Post-Fire Debris Flows

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Run tests to ensure everything works
6. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/gaia-stehekin-postfire-debrisflows.git
cd gaia-stehekin-postfire-debrisflows

# Create conda environment
conda env create -f environment.yml
conda activate gaia-stehekin

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

## Code Style

This project follows standard Python coding conventions:

- **PEP 8** for code style
- **Black** for automatic code formatting (line length: 100)
- **isort** for import sorting
- **Type hints** where appropriate
- **Docstrings** for all public functions and classes (Google style)

### Formatting Your Code

Before submitting a pull request, please format your code:

```bash
# Format with black
black src/ tests/

# Sort imports
isort src/ tests/

# Check with flake8
flake8 src/ tests/
```

## Testing

All new features should include tests. We use pytest for testing.

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names
- Include docstrings explaining what is being tested

Example:

```python
def test_model_forward_pass():
    """Test that the model forward pass produces correct output shape."""
    model = SeismicCNN(input_channels=3, num_classes=2)
    x = torch.randn(4, 3, 3000)
    output = model(x)
    assert output.shape == (4, 2)
```

## Documentation

- Update the README.md if you add new features
- Add docstrings to all new functions and classes
- Use Google-style docstrings

Example:

```python
def my_function(param1, param2):
    """
    Brief description of the function.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
    
    Returns:
        type: Description of return value
    
    Raises:
        ValueError: Description of when this error is raised
    """
    pass
```

## Submitting Changes

1. **Commit messages**: Use clear, descriptive commit messages
   - First line: brief summary (50 chars or less)
   - Blank line
   - Detailed description if needed

2. **Pull requests**:
   - Provide a clear description of the changes
   - Reference any related issues
   - Ensure all tests pass
   - Update documentation as needed

3. **Code review**:
   - Be open to feedback
   - Respond to review comments promptly
   - Make requested changes

## Reporting Issues

When reporting issues, please include:

- Clear description of the problem
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python version, OS, and relevant package versions
- Error messages and stack traces if applicable

## Feature Requests

Feature requests are welcome! Please:

- Check if the feature has already been requested
- Clearly describe the feature and its use case
- Explain why it would be beneficial to the project

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Prioritize the project's goals

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Contact the maintainers

Thank you for contributing to GAIA Stehekin Post-Fire Debris Flows!
