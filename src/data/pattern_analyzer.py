import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import Counter

class PatternAnalyzer:
    def __init__(self):
        self.daily_patterns = {}
        self.weekly_patterns = {}
        self.life_events = []
    
    def analyze_daily_routine(self, data: pd.DataFrame) -> Dict:
        """Analyze daily routine patterns"""
        if data.empty or 'datetime' not in data.columns:
            return {}
        
        # Debug information
        print(f"Input data shape: {data.shape}")
        print(f"Input data columns: {data.columns.tolist()}")
        
        # Work with a copy to avoid modifying original
        data = data.copy()
        
        # Fix duplicate columns by keeping only the first occurrence
        data = data.loc[:, ~data.columns.duplicated()]
        print(f"After removing duplicates: {data.columns.tolist()}")
        
        # Ensure datetime column is properly formatted
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['hour'] = data['datetime'].dt.hour
        data['date'] = data['datetime'].dt.date
        
        patterns = {}
        
        # Analyze wake-up time (first activity of the day)
        daily_first_activity = data.groupby('date')['hour'].min()
        patterns['wake_up_time'] = {
            'average': daily_first_activity.mean(),
            'std': daily_first_activity.std(),
            'most_common': daily_first_activity.mode().iloc[0] if not daily_first_activity.empty else None
        }
        
        # Analyze sleep time (last activity of the day) 
        daily_last_activity = data.groupby('date')['hour'].max()
        patterns['sleep_time'] = {
            'average': daily_last_activity.mean(),
            'std': daily_last_activity.std(),
            'most_common': daily_last_activity.mode().iloc[0] if not daily_last_activity.empty else None
        }
        
        # Activity frequency by hour
        hourly_activity = data.groupby('hour').size()
        patterns['hourly_activity'] = hourly_activity.to_dict()
        
        # Most active hours
        patterns['peak_hours'] = hourly_activity.nlargest(3).index.tolist()
        patterns['quiet_hours'] = hourly_activity.nsmallest(3).index.tolist()
        
        return patterns

    def detect_life_events(self, data: pd.DataFrame) -> List[Dict]:
        """Detect significant life events from activity patterns"""
        events = []
        
        if data.empty or len(data) < 10:
            return events
        
        # Ensure datetime column exists
        if 'datetime' not in data.columns and 'timestamp' in data.columns:
            data['datetime'] = pd.to_datetime(data['timestamp'])
        
        data['date'] = data['datetime'].dt.date
        
        # Detect activity volume changes
        daily_counts = data.groupby('date').size()
        rolling_avg = daily_counts.rolling(window=7, center=True).mean()
        
        for date, count in daily_counts.items():
            expected = rolling_avg.get(date, count)
            if pd.notna(expected) and abs(count - expected) > 2 * daily_counts.std():
                event_type = "high_activity" if count > expected else "low_activity"
                
                events.append({
                    'date': date,
                    'type': event_type,
                    'description': f"Unusual {event_type.replace('_', ' ')} detected",
                    'confidence': min(abs(count - expected) / daily_counts.std() / 3, 1.0),
                    'activity_count': count,
                    'expected_count': expected
                })
        
        # Detect travel patterns if travel activity exists
        if 'predicted_activity' in data.columns:
            travel_events = self._detect_travel_patterns(data)
            events.extend(travel_events)
        
        return sorted(events, key=lambda x: x['date'])
    
    def generate_insights(self, patterns: Dict, life_events: List[Dict]) -> Dict:
        """Generate insights from patterns and events"""
        insights = {}
        
        # Daily routine insights
        if 'wake_up_time' in patterns:
            wake_up = patterns['wake_up_time']['average']
            if wake_up < 7:
                insights['sleep_pattern'] = "You're an early riser! Most activity starts before 7 AM."
            elif wake_up > 9:
                insights['sleep_pattern'] = "You tend to start your day later, with activity beginning after 9 AM."
            else:
                insights['sleep_pattern'] = "You have a regular morning routine, starting around 7-9 AM."
        
        # Activity level insights
        if 'hourly_activity' in patterns:
            total_hours = len([h for h in patterns['hourly_activity'].values() if h > 0])
            if total_hours > 14:
                insights['activity_level'] = "You maintain high activity levels throughout most of the day."
            elif total_hours < 8:
                insights['activity_level'] = "Your activity is concentrated in fewer hours of the day."
            else:
                insights['activity_level'] = "You have moderate activity spread across the day."
        
        # Life events insights
        if life_events:
            high_activity_events = [e for e in life_events if e['type'] == 'high_activity']
            if high_activity_events:
                insights['life_events'] = f"Detected {len(high_activity_events)} periods of unusually high activity."
        
        return insights
    
    def _detect_travel_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """Detect travel patterns from activity data"""
        events = []
        
        travel_data = data[data['predicted_activity'] == 'Travel']
        if travel_data.empty:
            return events
        
        travel_data = travel_data.sort_values('datetime')
        travel_data['date'] = travel_data['datetime'].dt.date
        
        # Group consecutive travel days
        dates = sorted(travel_data['date'].unique())
        travel_periods = []
        current_period = []
        
        for i, date in enumerate(dates):
            if not current_period or (date - current_period[-1]).days <= 1:
                current_period.append(date)
            else:
                if len(current_period) >= 1:
                    travel_periods.append(current_period)
                current_period = [date]
        
        if current_period:
            travel_periods.append(current_period)
        
        # Create events for multi-day travel
        for period in travel_periods:
            if len(period) >= 2:
                events.append({
                    'date': period[0],
                    'type': 'travel_period', 
                    'description': f"{len(period)}-day travel period",
                    'confidence': 0.8,
                    'duration_days': len(period)
                })
        
        return events