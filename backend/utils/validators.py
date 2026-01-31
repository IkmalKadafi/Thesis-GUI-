"""
Data validation utilities
"""
from pathlib import Path
from typing import Tuple
import pandas as pd
from config.settings import ALLOWED_EXTENSIONS, MAX_FILE_SIZE


class ValidationError(Exception):
    """Custom validation error"""
    pass


def validate_file_extension(filename: str) -> bool:
    """Validate file extension"""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def validate_file_size(file_size: int) -> bool:
    """Validate file size"""
    return file_size <= MAX_FILE_SIZE


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate uploaded dataframe
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if dataframe is empty
    if df.empty:
        return False, "File is empty"
    
    # Check if there are at least 2 columns (1 region + 1 numeric)
    if len(df.columns) < 2:
        return False, "File must have at least 2 columns (1 region column + 1 numeric column)"
    
    # Check if there's at least one numeric column
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        return False, "File must have at least one numeric column"
    
    # Check for excessive missing values (>50% in any column)
    for col in df.columns:
        missing_pct = df[col].isna().sum() / len(df) * 100
        if missing_pct > 50:
            return False, f"Column '{col}' has {missing_pct:.1f}% missing values (>50% threshold)"
    
    return True, "Validation successful"


def get_region_column(df: pd.DataFrame) -> str:
    """
    Identify the region column (first non-numeric column)
    
    Returns:
        Column name of the region column
    """
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            return col
    
    # If no object column found, use first column
    return df.columns[0]
