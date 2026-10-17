"""
Configuration module for NASA C-MAPSS Transformer RUL Prediction.

Contains all hyperparameters, file paths, and model configuration.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Config:
    """Central configuration for the RUL prediction pipeline."""
    
    # ==========================================================================
    # Data Paths
    # ==========================================================================
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "Dataset")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")
    
    # Dataset files (FD001 by default)
    TRAIN_FILE: str = "train_FD001.txt"
    TEST_FILE: str = "test_FD001.txt"
    RUL_FILE: str = "RUL_FD001.txt"
    
    # ==========================================================================
    # Data Configuration
    # ==========================================================================
    SEQUENCE_LENGTH: int = 30          # Time window for input sequences
    MAX_RUL: int = 125                  # Piece-wise linear RUL cap
    VALIDATION_SPLIT: float = 0.2      # Train/validation split ratio
    RANDOM_SEED: int = 42
    
    # Column names
    ID_COL: str = "engine_id"
    CYCLE_COL: str = "cycle"
    OP_SETTINGS: List[str] = field(default_factory=lambda: [
        "op_setting_1", "op_setting_2", "op_setting_3"
    ])
    SENSOR_COLS: List[str] = field(default_factory=lambda: [
        f"sensor_{i}" for i in range(1, 22)
    ])
    
    # ==========================================================================
    # Model Architecture (Transformer Encoder)
    # ==========================================================================
    D_MODEL: int = 128                  # Transformer embedding dimension
    N_HEADS: int = 8                    # Number of attention heads
    N_ENCODER_LAYERS: int = 4           # Number of encoder layers
    D_FF: int = 256                     # Feed-forward hidden dimension
    DROPOUT: float = 0.1                # Dropout rate
    
    # ==========================================================================
    # Training Configuration
    # ==========================================================================
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 15
    
    # Loss function
    LAMBDA_OVER: float = 10.0           # Overestimation penalty weight
    
    # ==========================================================================
    # Sensor to Subsystem Mapping (for interpretability)
    # ==========================================================================
    SENSOR_TO_SUBSYSTEM: Dict[str, List[str]] = field(default_factory=lambda: {
        'sensor_1': ['Fan Inlet Temperature', 'Overall Inlet'],
        'sensor_2': ['Fan Inlet Pressure', 'Overall Inlet'],
        'sensor_3': ['HPC Outlet Pressure', 'High-Pressure Compressor (HPC)'],
        'sensor_4': ['HPT Outlet Temperature', 'High-Pressure Turbine (HPT)'],
        'sensor_5': ['LPC Outlet Temperature', 'Low-Pressure Compressor (LPC)'],
        'sensor_6': ['Fan Speed (RPM)', 'Fan'],
        'sensor_7': ['Fan Inlet Pressure', 'Overall Inlet'],
        'sensor_8': ['HPC Outlet Static Pressure', 'High-Pressure Compressor (HPC)'],
        'sensor_9': ['Corrected Fan Speed', 'Fan'],
        'sensor_10': ['Bypass Pressure Ratio', 'Fan'],
        'sensor_11': ['Fuel Flow', 'Combustion Chamber'],
        'sensor_12': ['LPC Outlet Temperature (Corrected)', 'Low-Pressure Compressor (LPC)'],
        'sensor_13': ['HPT Outlet Temperature (Corrected)', 'High-Pressure Turbine (HPT)'],
        'sensor_14': ['Physical Fan Speed', 'Fan'],
        'sensor_15': ['Physical Core Speed', 'HPC and HPT'],
        'sensor_16': ['Bleed Air', 'Overall Engine System'],
        'sensor_17': ['Fuel-Air Ratio', 'Combustion Chamber'],
        'sensor_18': ['TSFC', 'Overall Engine System'],
        'sensor_19': ['LPC Exit Temperature', 'Low-Pressure Compressor (LPC)'],
        'sensor_20': ['Engine Pressure Ratio', 'Overall Engine System'],
        'sensor_21': ['HPT Exit Temperature', 'High-Pressure Turbine (HPT)']
    })
    
    # ==========================================================================
    # Device Configuration
    # ==========================================================================
    @property
    def DEVICE(self) -> torch.device:
        """Get the best available device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    def TRAIN_PATH(self) -> str:
        return os.path.join(self.DATA_DIR, self.TRAIN_FILE)
    
    @property
    def TEST_PATH(self) -> str:
        return os.path.join(self.DATA_DIR, self.TEST_FILE)
    
    @property
    def RUL_PATH(self) -> str:
        return os.path.join(self.DATA_DIR, self.RUL_FILE)
    
    @property
    def FEATURE_COLS(self) -> List[str]:
        """All feature columns (op_settings + sensors)."""
        return self.OP_SETTINGS + self.SENSOR_COLS
    
    @property
    def INPUT_DIM(self) -> int:
        """Number of input features."""
        return len(self.FEATURE_COLS)
    
    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


# Global configuration instance
config = Config()
