"""
Configuration module for the Flame Super-Resolution project.
This module defines the configuration settings for data, model, training, and visualization components of the project.
It uses dataclasses and provides utility methods for setting up the environment.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    """
    input_path: str  # Path to the directory containing input data files
    output_path: str  # Path to the directory where outputs will be saved
    batch_size: int = 32  # Number of samples per batch during training/validation
    num_workers: int = 0  # Number of worker processes for data loading (0 for single-threaded)
    train_csv: str = 'train.csv'  # Filename of the training data CSV file
    val_csv: str = 'val.csv'  # Filename of the validation data CSV file
    lr_resolution: tuple = (16, 16)  # Resolution of low-resolution input data (height, width)
    hr_resolution: tuple = (128, 128)  # Resolution of high-resolution target data (height, width)


@dataclass
class ModelConfig:
    """
    This class defines parameters for the neural network model, including Fourier modes, layer widths, number of layers, and dropout rates for different streams.
    """
    modes1: int = 8  # Number of Fourier modes in the first spatial dimension
    modes2: int = 8  # Number of Fourier modes in the second spatial dimension
    width_rho: int = 48  # Width for the density stream
    width_vel: int = 64  # Width for the velocity stream
    n_layers: int = 4  # Number of layers in the model
    dropout: float = 0.15  # Dropout probability for regularization
    dropout_rho_factor: float = 0.7  # Factor to reduce dropout for the density stream (relative to main dropout)


@dataclass
class TrainingConfig:
    """
    This class specifies hyperparameters for training, including learning rate, regularization, early stopping, and learning rate scheduling.
    """
    epochs: int = 300  # Maximum number of training epochs
    learning_rate: float = 1e-3  # Initial learning rate for optimization
    weight_decay: float = 1e-4  # L2 regularization strength
    grad_clip_norm: float = 1.0  # Maximum norm for gradient clipping
    patience: int = 50  # Number of epochs to wait for improvement before early stopping
    improvement_threshold: float = 0.995  # Minimum improvement ratio required (e.g., 0.5% improvement)
    checkpoint_frequency: int = 10  # Frequency (in epochs) to save model checkpoints

    # Learning rate scheduler configuration (Cosine Annealing with Warm Restarts)
    scheduler_T_0: int = 20  # Initial restart interval for the scheduler
    scheduler_T_mult: int = 2  # Multiplier for increasing restart intervals
    scheduler_eta_min: float = 1e-6  # Minimum learning rate after annealing


@dataclass
class VisualizationConfig:
    """
    This class holds settings for figure resolution, format, sample count, and LaTeX rendering options.
    """
    dpi: int = 300  # Dots per inch for figure resolution
    figure_format: str = 'pdf'  # File format for saving figures (e.g., 'pdf', 'png')
    num_samples: int = 3  # Number of sample visualizations to generate
    use_latex: bool = True  # Whether to use LaTeX for text rendering in plots


class Config:
    """
    This class initializes and manages the overall configuration for the project, including device setup, directory creation, random seed setting, and visualization configuration. It provides methods to set up the environment for experiments.
    """

    def __init__(self):
        # Initialize sub-configurations
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.visualization = VisualizationConfig()

        # Automatically detect and set the compute device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set a default random seed
        self.seed = 42

    def setup_matplotlib(self):
        """
        This method sets up matplotlib parameters.
        """

        # Enable LaTeX rendering
        if self.visualization.use_latex:
            plt.rcParams["text.usetex"] = True
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = ["Computer Modern"]

        # Set font sizes for various plot elements
        plt.rcParams['xtick.labelsize'] = 11  # X-axis tick labels
        plt.rcParams['ytick.labelsize'] = 11  # Y-axis tick labels
        plt.rcParams['axes.titlesize'] = 11  # Plot titles
        plt.rcParams['axes.labelsize'] = 11  # Axis labels
        plt.rcParams['legend.fontsize'] = 10  # Legend text

        # Set resolution and saving parameters
        plt.rcParams['figure.dpi'] = self.visualization.dpi  # Display DPI
        plt.rcParams['savefig.dpi'] = self.visualization.dpi  # Save DPI
        plt.rcParams['savefig.bbox'] = 'tight'  # Tight bounding box for saves
        plt.rcParams['axes.grid'] = False  # Disable grid by default

    def setup_directories(self):
        """
        Create necessary directories for the project.
        """
        from pathlib import Path

        # List of directories to create
        paths = [
            self.data.output_path,  # Main output directory
            f"{self.data.output_path}figures/",  # Directory for saving figures
            f"{self.data.output_path}logs/",  # Directory for logging files
            f"{self.data.output_path}checkpoints/",  # Directory for model checkpoints
        ]

        # if it doesn't exist create it
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)

    def set_random_seeds(self):
        """
        Set random seeds for reproducibility across libraries.
        """
        # Set seed for PyTorch CPU operations
        torch.manual_seed(self.seed)

        # Set seed for NumPy random number generation
        np.random.seed(self.seed)

        # Set seeds for CUDA operations if GPU is available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)  # For multi-GPU setups
            torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
            torch.backends.cudnn.benchmark = False  # Disable benchmark for reproducibility

    def print_config(self):
        """
        Prints a summary of the current configuration.
        """
        print("=" * 80)
        print("Configuration Summary")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Random Seed: {self.seed}")
        print()

        print("Data Configuration:")
        print(f"  Batch Size: {self.data.batch_size}")
        print(f"  Low-Resolution: {self.data.lr_resolution}")
        print(f"  High-Resolution: {self.data.hr_resolution}")
        print()

        print("Model Configuration:")
        print(f"  Fourier Modes: {self.model.modes1} x {self.model.modes2}")
        print(f"  Width for the Density flow-field: {self.model.width_rho}")
        print(f"  Width for the Velocity flow-field: {self.model.width_vel}")
        print(f"  Layers: {self.model.n_layers}")
        print(f"  Dropout: {self.model.dropout}")
        print()

        print("Training Configuration:")
        print(f"  Epochs: {self.training.epochs}")
        print(f"  Learning Rate: {self.training.learning_rate}")
        print(f"  Weight Decay: {self.training.weight_decay}")
        print(f"  Patience: {self.training.patience}")
        print("=" * 80)
        print()