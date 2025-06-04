import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime, timedelta
from src.api.huggingface_client import HuggingFaceClient
from config.settings import ACTIVITY_LABELS, CONFIDENCE_THRESHOLD 

class ActivityClassifier:
    def __init__(self):
        self.hf_client = HuggingFaceClient()
        self.activity_labels = ACTIVITY_LABELS
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
    def classify_single_text(self, text: str, category: str = "general_activities") -> Dict:
        """Classify a single text into activity categories"""
        labels = self.activity_labels.get(category, self.activity_labels["general_activities"])
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        if not cleaned_text.strip():
            return {"error": "Empty text after preprocessing"}
        
        # Get classification from Hugging Face
        result = self.hf_client.classify_activity(cleaned_text, labels)
        
        if not result:
            return {"error": "Failed to get classification from API"}
        
        # Process and enhance results
        processed_result = self._process_classification_result(result, text)
        
        return processed_result
    
    def classify_batch(self, texts: List[str], category: str = "general_activities") -> List[Dict]:
        """Classify multiple texts at once"""
        results = []
        labels = self.activity_labels.get(category, self.activity_labels["general_activities"])
        
        for text in texts:
            cleaned_text = self._preprocess_text(text)
            if cleaned_text.strip():
                result = self.hf_client.classify_activity(cleaned_text, labels)
                if result:
                    processed_result = self._process_classification_result(result, text)
                    results.append(processed_result)
                else:
                    results.append({"error": "API classification failed", "original_text": text})
            else:
                results.append({"error": "Empty text", "original_text": text})
        
        return results
    
    def classify_with_context(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify activities with temporal and contextual information"""
        if data.empty:
            return data
        
        # Ensure required columns exist
        required_cols = ['text', 'timestamp']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Convert timestamp to datetime
        data['datetime'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['datetime'].dt.hour
        data['day_of_week'] = data['datetime'].dt.day_name()
        
        # Classify each text
        classifications = []
        for idx, row in data.iterrows():
            # Determine activity category based on time context
            category = self._determine_category_by_time(row['hour'])
            
            # Add temporal context to text
            contextual_text = self._add_temporal_context(row['text'], row['hour'], row['day_of_week'])
            
            # Classify
            result = self.classify_single_text(contextual_text, category)
            result['original_text'] = row['text']
            result['hour'] = row['hour']
            result['day_of_week'] = row['day_of_week']
            
            classifications.append(result)
        
        # Add classification results to dataframe
        classification_df = pd.DataFrame(classifications)
        result_df = pd.concat([data.reset_index(drop=True), classification_df], axis=1)
        
        return result_df
    
    def detect_routine_activities(self, data: pd.DataFrame) -> Dict[str, List]:
        """Detect recurring routine activities"""
        if 'predicted_activity' not in data.columns:
            data = self.classify_with_context(data)
        
        routines = {
            'morning_routine': [],
            'work_routine': [],
            'evening_routine': [],
            'weekend_routine': []
        }
        
        # Group by time periods
        morning_data = data[data['hour'].between(6, 11)]
        work_data = data[data['hour'].between(9, 17)]
        evening_data = data[data['hour'].between(18, 23)]
        weekend_data = data[data['day_of_week'].isin(['Saturday', 'Sunday'])]
        
        # Find common activities in each period
        routines['morning_routine'] = self._find_common_activities(morning_data)
        routines['work_routine'] = self._find_common_activities(work_data)
        routines['evening_routine'] = self._find_common_activities(evening_data)
        routines['weekend_routine'] = self._find_common_activities(weekend_data)
        
        return routines
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for classification"""
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove email addresses and URLs
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        text = re.sub(r'http\S+|www\S+', '[URL]', text)
        
        # Remove special characters but keep emoticons
        text = re.sub(r'[^\w\s.:!?@#$%&*()+=\-\[\]{};\'",<>/|\\`~_^]', '', text)
        
        # Limit length
        if len(text) > 512:
            text = text[:512]
        
        return text
    
    def _process_classification_result(self, result: Dict, original_text: str) -> Dict:
        """Process and enhance classification results"""
        if 'labels' not in result or 'scores' not in result:
            return {"error": "Invalid API response format"}
        
        processed = {
            'predicted_activity': result['labels'][0],
            'confidence': result['scores'][0],
            'all_predictions': dict(zip(result['labels'], result['scores'])),
            'original_text': original_text,
            'high_confidence': result['scores'][0] > self.confidence_threshold
        }
        
        # Add activity insights
        processed['insights'] = self._generate_activity_insights(
            result['labels'][0], 
            result['scores'][0], 
            original_text
        )
        
        return processed
    
    def _determine_category_by_time(self, hour: int) -> str:
        """Determine activity category based on time of day"""
        if 6 <= hour < 9:
            return "daily_routine"  # Morning routine
        elif 9 <= hour < 17:
            return "general_activities"  # Work hours
        elif 17 <= hour < 22:
            return "general_activities"  # Evening activities
        else:
            return "daily_routine"  # Late night/early morning
    
    def _add_temporal_context(self, text: str, hour: int, day_of_week: str) -> str:
        """Add temporal context to text for better classification"""
        time_context = ""
        
        if 6 <= hour < 12:
            time_context = "In the morning: "
        elif 12 <= hour < 17:
            time_context = "In the afternoon: "
        elif 17 <= hour < 22:
            time_context = "In the evening: "
        else:
            time_context = "Late at night: "
        
        if day_of_week in ['Saturday', 'Sunday']:
            time_context += f"On {day_of_week} "
        
        return time_context + text
    
    def _find_common_activities(self, data: pd.DataFrame, min_frequency: int = 2) -> List[Dict]:
        """Find common activities in a dataset"""
        if data.empty or 'predicted_activity' not in data.columns:
            return []
        
        activity_counts = data['predicted_activity'].value_counts()
        common_activities = activity_counts[activity_counts >= min_frequency]
        
        results = []
        for activity, count in common_activities.items():
            avg_confidence = data[data['predicted_activity'] == activity]['confidence'].mean()
            
            results.append({
                'activity': activity,
                'frequency': count,
                'average_confidence': avg_confidence,
                'percentage': (count / len(data)) * 100
            })
        
        return sorted(results, key=lambda x: x['frequency'], reverse=True)
    
    def _generate_activity_insights(self, activity: str, confidence: float, text: str) -> Dict:
        """Generate insights about the classified activity"""
        insights = {
            'confidence_level': 'high' if confidence > 0.8 else 'medium' if confidence > 0.5 else 'low',
            'activity_type': self._categorize_activity_type(activity),
            'keywords_found': self._extract_keywords(text, activity)
        }
        
        return insights
    
    def _categorize_activity_type(self, activity: str) -> str:
        """Categorize activity into broader types"""
        work_activities = ['Work', 'Studying', 'Meeting']
        leisure_activities = ['Entertainment', 'Socializing', 'Shopping']
        health_activities = ['Exercise', 'Eating', 'Sleeping']
        
        if activity in work_activities:
            return 'productive'
        elif activity in leisure_activities:
            return 'leisure'
        elif activity in health_activities:
            return 'health_wellness'
        else:
            return 'other'
    
    def _extract_keywords(self, text: str, activity: str) -> List[str]:
        """Extract relevant keywords that led to the classification"""
        # Simple keyword extraction based on activity type
        activity_keywords = {
            'Work': ['work', 'office', 'meeting', 'project', 'deadline', 'email', 'call'],
            'Exercise': ['gym', 'run', 'workout', 'fitness', 'training', 'sports'],
            'Travel': ['flight', 'airport', 'hotel', 'vacation', 'trip', 'destination'],
            'Eating': ['lunch', 'dinner', 'breakfast', 'food', 'restaurant', 'cooking'],
            'Shopping': ['buy', 'purchase', 'store', 'mall', 'shopping', 'amazon'],
            'Socializing': ['friends', 'party', 'hangout', 'social', 'meet', 'chat'],
            'Entertainment': ['movie', 'show', 'music', 'game', 'watch', 'play']
        }
        
        keywords = activity_keywords.get(activity, [])
        found_keywords = [kw for kw in keywords if kw.lower() in text.lower()]
        
        return found_keywords