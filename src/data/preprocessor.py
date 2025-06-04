# src/data/preprocessor.py
import pandas as pd
import re
from typing import Dict, List, Optional
from datetime import datetime

class DataPreprocessor:
    def __init__(self):
        """Initialize the DataPreprocessor"""
        self.timezone_mappings = {
            "EST": ["new york", "eastern", "est", "edt"],
            "PST": ["california", "pacific", "pst", "pdt"],
            "GMT": ["london", "uk", "gmt", "utc"],
            "CET": ["berlin", "paris", "cet", "cest"],
            "JST": ["japan", "tokyo", "jst"],
            "IST": ["india", "mumbai", "delhi", "ist"]
        }
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a DataFrame containing text and timestamp data
        
        Args:
            df: Input DataFrame with 'text' and 'timestamp' columns
        
        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df
        
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure required columns
        required_columns = ['text', 'timestamp']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['datetime'] = df['timestamp']
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.day_name()
        
        # Clean text
        df['text'] = df['text'].apply(self._clean_text)
        
        # Extract timezone if available
        df['timezone'] = df['text'].apply(self._extract_timezone)
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing unwanted elements"""
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Remove URLs and emails
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        text = re.sub(r'http\S+|www\S+', '[URL]', text)
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.:!?@#$%&*()+=\-\[\]{};\'",<>/|\\`~_^]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_timezone(self, text: str) -> Optional[str]:
        """Extract timezone from text based on keywords"""
        if not isinstance(text, str):
            return None
        
        text = text.lower()
        for timezone, keywords in self.timezone_mappings.items():
            if any(keyword in text for keyword in keywords):
                return timezone
        return None
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data for consistent processing
        
        Args:
            df: Input DataFrame
        
        Returns:
            Normalized DataFrame
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Handle missing values
        df['text'] = df['text'].fillna("")
        df['timestamp'] = df['timestamp'].fillna(datetime.now())
        
        # Convert types
        df['text'] = df['text'].astype(str)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        return df