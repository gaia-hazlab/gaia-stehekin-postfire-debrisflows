"""
GAIA Stehekin Post-Fire Debris Flows Analysis Package

This package provides tools for analyzing post-fire debris flows
using seismic data and deep learning methods.
"""

__version__ = "0.1.0"
__author__ = "Gaia Hazlab"

# Import main modules for easier access
from . import models
from . import data
from . import utils

__all__ = ["models", "data", "utils"]
