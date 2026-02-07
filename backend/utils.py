"""
Utility functions (Helpers & Validators)
"""
import re
from typing import Any, Dict, List, Tuple
from pathlib import Path
import pandas as pd
from backend.settings import ALLOWED_EXTENSIONS, MAX_FILE_SIZE

# ==========================================
# HELPERS
# ==========================================

def normalize_region_name(name: str) -> str:
    """
    Normalize region name for matching
    
    Examples:
        "Kab. Bandung" -> "kabupaten bandung"
        "Kota Jakarta" -> "kota jakarta"
        "DKI Jakarta" -> "dki jakarta"
    """
    if not isinstance(name, str):
        return str(name).lower().strip()
    
    # Convert to lowercase
    normalized = name.lower().strip()
    
    # Replace common abbreviations
    normalized = normalized.replace("kab.", "kabupaten")
    normalized = normalized.replace("kab ", "kabupaten ")
    
    # Remove extra spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with thousand separators"""
    return f"{value:,.{decimals}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def dict_to_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert dictionary to list of dictionaries"""
    return [{"key": k, "value": v} for k, v in data.items()]


# ==========================================
# VALIDATORS
# ==========================================

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
