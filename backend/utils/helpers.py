"""
Helper utility functions
"""
import re
from typing import Any, Dict, List


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
