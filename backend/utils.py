import re
from typing import Any, Dict, List, Tuple
from pathlib import Path
import pandas as pd
from backend.settings import ALLOWED_EXTENSIONS, MAX_FILE_SIZE


class ValidationError(Exception):
    pass


def normalize_region_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name).lower().strip()
    normalized = name.lower().strip().replace("kab.", "kabupaten").replace("kab ", "kabupaten ")
    return re.sub(r'\s+', ' ', normalized)


def format_number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def dict_to_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"key": k, "value": v} for k, v in data.items()]


def validate_file_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def validate_file_size(file_size: int) -> bool:
    return file_size <= MAX_FILE_SIZE


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    if df.empty:
        return False, "File is empty"
    if len(df.columns) < 2:
        return False, "File must have at least 2 columns (1 region column + 1 numeric column)"
    if len(df.select_dtypes(include=['number']).columns) == 0:
        return False, "File must have at least one numeric column"
    for col in df.columns:
        missing_pct = df[col].isna().sum() / len(df) * 100
        if missing_pct > 50:
            return False, f"Column '{col}' has {missing_pct:.1f}% missing values (>50% threshold)"
    return True, "Validation successful"


def get_region_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if df[col].dtype in ('object', 'category') or df[col].dtype.name == 'category':
            return col
    return df.columns[0]
