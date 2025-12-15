# GAIA Stehekin Post-Fire Debris Flows

A Python framework for analyzing post-fire debris flows in Stehekin using seismic data and deep learning methods.

## Features

- PyTorch-based neural network models for seismic data classification
- Integration with ObsPy for seismic data processing following seismology conventions
- SeisBench workflow for automated event detection
- Interactive visualization tools for detection quality control
- Jupyter notebook examples for complete workflows

## Repository Structure

```
gaia-stehekin-postfire-debrisflows/
├── src/                           # Source code
│   ├── __init__.py               # Package initialization
│   ├── models.py                 # PyTorch model definitions
│   ├── data.py                   # ObsPy-compliant data processing
│   ├── utils.py                  # Visualization and helper functions
│   └── detect.py                 # Event detection utilities
├── notebooks/                     # Jupyter notebooks
├── plots/                         # Generated plots and visualizations
├── data/                          # Seismic data
│   ├── raw/                      # Raw input data
│   └── processed/                # Processed datasets
├── tests/                         # Unit tests
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
├── setup.py                       # Package setup file
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/gaia-hazlab/gaia-stehekin-postfire-debrisflows.git
cd gaia-stehekin-postfire-debrisflows

# Create and activate conda environment
conda env create -f environment.yml
conda activate gaia-stehekin

# Install the package in development mode
pip install -e .
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/gaia-hazlab/gaia-stehekin-postfire-debrisflows.git
cd gaia-stehekin-postfire-debrisflows

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torch
from src.models import SeismicCNN, load_model, save_model
from src.data import SeismicDataProcessor
from src.utils import plot_seismogram, get_device

# Initialize device
device = get_device()

# Create a model
model = SeismicCNN(input_channels=3, num_classes=2)
model = model.to(device)

# Initialize data processor
processor = SeismicDataProcessor(
    sampling_rate=100.0,
    window_length=30.0,
    normalize=True
)

# Load and process seismic data
stream = processor.load_seismic_data('path/to/seismic_file.mseed')
stream = processor.preprocess_stream(stream, freqmin=1.0, freqmax=20.0)
data = processor.stream_to_array(stream)

# Make predictions
data_tensor = processor.to_torch(data).unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    output = model(data_tensor)
    probabilities = torch.softmax(output, dim=1)

print(f"Predictions: {probabilities}")
```

## Dependencies

Core dependencies:

- PyTorch (>=2.0.0): Deep learning framework
- ObsPy (>=1.4.0): Seismic data processing
- SeisBench (>=0.4.0): Pre-trained seismic models and benchmarks
- NumPy, SciPy, Pandas: Scientific computing and data analysis
- Matplotlib: Visualization

See `requirements.txt` or `environment.yml` for the complete list.

## Data Format

The framework follows ObsPy conventions for seismic data processing and supports various formats including MiniSEED (.mseed), SAC, SEG-Y, and ASDF.

### Data Processing Workflow

The `SeismicDataProcessor` follows the standard ObsPy preprocessing workflow:

1. Load data using ObsPy's `read()` function with automatic format detection
2. Detrend to remove linear trend and mean
3. Taper using cosine taper to avoid edge effects
4. Filter with bandpass filter
5. Resample to target sampling rate
6. Sort components (Z, N, E order)
7. Merge to handle gaps and overlaps

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See `CONTRIBUTING.md` for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project builds upon:
- [PyTorch](https://pytorch.org/)
- [ObsPy](https://obspy.org/)
- [SeisBench](https://github.com/seisbench/seisbench)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gaia_stehekin_2024,
  title={GAIA Stehekin Post-Fire Debris Flows Analysis},
  author={Gaia Hazlab},
  year={2024},
  url={https://github.com/gaia-hazlab/gaia-stehekin-postfire-debrisflows}
}
```
