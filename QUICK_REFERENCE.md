# Quick Reference Guide

## Project Structure

```
gaia-stehekin-postfire-debrisflows/
├── src/                    # Source code package
├── notebooks/              # Jupyter notebooks for analysis
├── data/                   # Data directory (git-ignored)
├── plots/                  # Output plots (git-ignored)
├── tests/                  # Unit tests
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment
├── setup.py                # Package setup
└── pyproject.toml          # Project configuration
```

## Common Commands

### Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate gaia-stehekin

# Install package in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

### Running Code

```bash
# Run Python script
python scripts/your_script.py

# Launch Jupyter notebook
jupyter notebook

# Run specific notebook
jupyter notebook notebooks/example_analysis.ipynb
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v tests/
```

### Code Formatting

```bash
# Format code with black
black src/ tests/

# Sort imports
isort src/ tests/

# Lint with flake8
flake8 src/ tests/
```

## Module Usage

### Models

```python
from src.models import SeismicCNN, save_model, load_model

# Create model
model = SeismicCNN(input_channels=3, num_classes=2)

# Save model
save_model(model, 'checkpoint.pth', optimizer=optimizer, epoch=10)

# Load model
info = load_model(model, 'checkpoint.pth', device='cpu')
```

### Data Processing

```python
from src.data import SeismicDataProcessor

# Create processor
processor = SeismicDataProcessor(
    sampling_rate=100.0,
    window_length=30.0,
    normalize=True
)

# Load data
stream = processor.load_seismic_data('data.mseed')

# Preprocess
stream = processor.preprocess_stream(stream, freqmin=1.0, freqmax=20.0)

# Convert to array
data = processor.stream_to_array(stream)

# Convert to torch tensor
tensor = processor.to_torch(data)
```

### Utilities

```python
from src.utils import get_device, plot_seismogram, set_random_seed

# Get device
device = get_device()  # Auto-detect GPU/CPU

# Set random seed
set_random_seed(42)

# Plot seismogram
fig, axes = plot_seismogram(stream, save_path='plot.png')
```

### Event Detection

```python
from src.detect import multi_class_detection, calculate_event_metrics

# Detect events
events = multi_class_detection(probabilities, threshold=0.5, min_distance=100)

# Calculate metrics
event = calculate_event_metrics(probabilities, events[0])
```

## Configuration

Edit `config.yaml` to change default settings:

```yaml
data:
  sampling_rate: 100.0
  window_length: 30.0
  freqmin: 1.0
  freqmax: 20.0

model:
  input_channels: 3
  num_classes: 2
  dropout: 0.5

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
```

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/my-feature

# Stage changes
git add .

# Commit changes
git commit -m "Add feature description"

# Push to remote
git push origin feature/my-feature

# Create pull request on GitHub
```

## Troubleshooting

### Import Errors

If you get import errors, make sure:
1. You've activated the conda environment
2. You've installed the package with `pip install -e .`
3. You're in the project root directory

### CUDA/GPU Issues

```python
# Check if CUDA is available
import torch
print(torch.cuda.is_available())

# Force CPU usage
device = get_device('cpu')
```

### ObsPy Data Loading

```python
# Check supported formats
from obspy import read
stream = read('path/to/data')  # Auto-detects format

# Specify format explicitly
stream = read('path/to/data', format='MSEED')
```

## Resources

- [ObsPy Documentation](https://docs.obspy.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [SeisBench Documentation](https://seisbench.readthedocs.io/)
