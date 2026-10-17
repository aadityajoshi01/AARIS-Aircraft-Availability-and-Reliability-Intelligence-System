"""
Data preprocessing module for NASA C-MAPSS dataset.

Handles loading, cleaning, feature engineering, and scaling of turbofan
engine degradation data for RUL prediction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class CMAPSSPreprocessor:
    """
    Preprocessor for NASA C-MAPSS turbofan engine degradation dataset.
    
    Handles:
    - Data loading and parsing
    - RUL computation with piece-wise linear capping
    - Feature scaling (MinMax normalization)
    - Train/test data preparation
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration object. Uses default if None.
        """
        self.config = config or Config()
        self.scaler = MinMaxScaler()
        self._is_fitted = False
        
    def _get_column_names(self) -> list:
        """Generate column names for the dataset."""
        return (
            [self.config.ID_COL, self.config.CYCLE_COL] + 
            self.config.OP_SETTINGS + 
            self.config.SENSOR_COLS
        )
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and parse a C-MAPSS data file.
        
        Args:
            file_path: Path to the data file (space-separated).
            
        Returns:
            DataFrame with proper column names.
        """
        df = pd.read_csv(file_path, sep=' ', header=None)
        # Drop the last two NaN columns (artifact of space separator)
        df.drop(df.columns[-2:], axis=1, inplace=True)
        df.columns = self._get_column_names()
        return df
    
    def compute_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Remaining Useful Life for each record.
        
        Uses piece-wise linear RUL with capping at MAX_RUL to handle
        the "healthy" phase where degradation is not yet observable.
        
        Args:
            df: DataFrame with engine_id and cycle columns.
            
        Returns:
            DataFrame with added 'RUL' column.
        """
        df = df.copy()
        
        # Get maximum cycle for each engine
        max_cycles = df.groupby(self.config.ID_COL)[self.config.CYCLE_COL].max()
        max_cycles = max_cycles.reset_index()
        max_cycles.columns = [self.config.ID_COL, 'max_cycle']
        
        # Merge and compute RUL
        df = df.merge(max_cycles, on=self.config.ID_COL)
        df['RUL'] = df['max_cycle'] - df[self.config.CYCLE_COL]
        
        # Apply piece-wise linear cap
        df['RUL'] = df['RUL'].clip(upper=self.config.MAX_RUL)
        
        # Clean up
        df.drop('max_cycle', axis=1, inplace=True)
        
        return df
    
    def fit_scaler(self, df: pd.DataFrame) -> 'CMAPSSPreprocessor':
        """
        Fit the MinMax scaler on training data.
        
        Args:
            df: Training DataFrame.
            
        Returns:
            Self for method chaining.
        """
        self.scaler.fit(df[self.config.FEATURE_COLS])
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling to feature columns.
        
        Args:
            df: DataFrame to transform.
            
        Returns:
            DataFrame with scaled features.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit_scaler() first.")
        
        df = df.copy()
        df[self.config.FEATURE_COLS] = self.scaler.transform(
            df[self.config.FEATURE_COLS]
        )
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform in one step."""
        return self.fit_scaler(df).transform(df)
    
    def prepare_train_data(self) -> pd.DataFrame:
        """
        Load, process, and scale training data.
        
        Returns:
            Processed training DataFrame with RUL and scaled features.
        """
        # Load data
        train_df = self.load_data(self.config.TRAIN_PATH)
        
        # Compute RUL
        train_df = self.compute_rul(train_df)
        
        # Fit and transform features
        train_df = self.fit_transform(train_df)
        
        return train_df
    
    def prepare_test_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load and process test data.
        
        Returns:
            Tuple of (processed test DataFrame, true RUL values).
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Scaler not fitted. Call prepare_train_data() first."
            )
        
        # Load test data
        test_df = self.load_data(self.config.TEST_PATH)
        
        # Load ground truth RUL
        true_rul = pd.read_csv(self.config.RUL_PATH, header=None)
        true_rul = true_rul.values.flatten()
        
        # Transform features (no RUL for test set until evaluation)
        test_df = self.transform(test_df)
        
        return test_df, true_rul
    
    def get_engine_data(
        self, 
        df: pd.DataFrame, 
        engine_id: int
    ) -> pd.DataFrame:
        """
        Extract data for a specific engine.
        
        Args:
            df: Full DataFrame.
            engine_id: Engine ID to filter.
            
        Returns:
            DataFrame for the specified engine.
        """
        return df[df[self.config.ID_COL] == engine_id].reset_index(drop=True)
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """
        Compute dataset statistics for reporting.
        
        Args:
            df: DataFrame to analyze.
            
        Returns:
            Dictionary of statistics.
        """
        stats = {
            'num_engines': df[self.config.ID_COL].nunique(),
            'total_records': len(df),
            'avg_cycles_per_engine': df.groupby(
                self.config.ID_COL
            )[self.config.CYCLE_COL].max().mean(),
            'min_cycles': df.groupby(
                self.config.ID_COL
            )[self.config.CYCLE_COL].max().min(),
            'max_cycles': df.groupby(
                self.config.ID_COL
            )[self.config.CYCLE_COL].max().max(),
        }
        
        if 'RUL' in df.columns:
            stats['rul_range'] = (df['RUL'].min(), df['RUL'].max())
        
        return stats


def create_sequences(
    df: pd.DataFrame,
    config: Config,
    seq_length: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for training.
    
    Args:
        df: Preprocessed DataFrame with features and RUL.
        config: Configuration object.
        seq_length: Sequence length (uses config default if None).
        
    Returns:
        Tuple of (X, y) where:
        - X: [num_samples, seq_length, num_features]
        - y: [num_samples], RUL normalized to [0, 1] by MAX_RUL
    """
    seq_length = seq_length or config.SEQUENCE_LENGTH
    
    X, y = [], []
    
    for engine_id in df[config.ID_COL].unique():
        engine_df = df[df[config.ID_COL] == engine_id].reset_index(drop=True)
        
        # Create sequences using sliding window
        for i in range(len(engine_df) - seq_length + 1):
            # Input: seq_length consecutive cycles
            X.append(engine_df.iloc[i:i + seq_length][config.FEATURE_COLS].values)
            # Target: normalized RUL at the LAST cycle of the window, so that
            # training windows match test windows (which end on the final cycle)
            y.append(engine_df.iloc[i + seq_length - 1]['RUL'] / config.MAX_RUL)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def create_test_sequences(
    df: pd.DataFrame,
    config: Config,
    seq_length: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for test set (last seq_length cycles per engine).
    
    Args:
        df: Preprocessed test DataFrame.
        config: Configuration object.
        seq_length: Sequence length (uses config default if None).
        
    Returns:
        Tuple of (X, engine_ids).
    """
    seq_length = seq_length or config.SEQUENCE_LENGTH
    
    X, engine_ids = [], []
    
    for engine_id in df[config.ID_COL].unique():
        engine_df = df[df[config.ID_COL] == engine_id].reset_index(drop=True)
        
        if len(engine_df) >= seq_length:
            # Take last seq_length cycles
            X.append(engine_df.iloc[-seq_length:][config.FEATURE_COLS].values)
            engine_ids.append(engine_id)
    
    return np.array(X, dtype=np.float32), np.array(engine_ids)
