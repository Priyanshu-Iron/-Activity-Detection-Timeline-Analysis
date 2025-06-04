# src/utils/helpers.py
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import json

def save_to_json(data: Any, filename: str) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filename: Output filename
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)

def load_from_json(filename: str) -> Any:
    """
    Load data from JSON file
    
    Args:
        filename: Input filename
    
    Returns:
        Loaded data
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def convert_to_dataframe(data: List[Dict]) -> pd.DataFrame:
    """
    Convert list of dictionaries to pandas DataFrame
    
    Args:
        data: List of dictionaries
    
    Returns:
        DataFrame
    """
    return pd.DataFrame(data)

def timestamp_to_string(timestamp: datetime) -> str:
    """
    Convert timestamp to formatted string
    
    Args:
        timestamp: Datetime object
    
    Returns:
        Formatted string
    """
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')

def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate if DataFrame has required columns
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names
    
    Returns:
        Boolean indicating if all required columns are present
    """
    return all(col in df.columns for col in required_columns)