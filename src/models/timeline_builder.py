import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class TimelineBuilder:
    def __init__(self):
        self.timeline_data = []
        self.major_events = []
        
    def build_activity_timeline(self, data: pd.DataFrame) -> Dict:
        """Build comprehensive activity timeline"""
        
        # Fix duplicate columns issue
        data = data.loc[:, ~data.columns.duplicated()]
        
        if data.empty:
            return {}
        
        # Rest of your existing method...
        timeline = {
            'daily_timeline': self._create_daily_timeline(data),
            # ... other timeline components
        }
        
        return timeline
    
    def _create_daily_timeline(self, data: pd.DataFrame) -> Dict:
        """Create daily timeline breakdown"""
        
        # Ensure data doesn't have duplicate columns
        data = data.loc[:, ~data.columns.duplicated()]
        
        # Ensure datetime column exists and is properly formatted
        if 'datetime' not in data.columns:
            if 'timestamp' in data.columns:
                data['datetime'] = pd.to_datetime(data['timestamp'])
            else:
                return {}
        
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['date'] = data['datetime'].dt.date
        data['hour'] = data['datetime'].dt.hour
        
        daily_timeline = {}
        
        for date in sorted(data['date'].unique()):
            day_data = data[data['date'] == date]
            
            daily_timeline[str(date)] = {
                'total_activities': len(day_data),
                'active_hours': day_data['hour'].nunique(),  # Use nunique() instead of len(unique())
                'first_activity': day_data['hour'].min(),
                'last_activity': day_data['hour'].max(),
                'activities': day_data['predicted_activity'].value_counts().to_dict() if 'predicted_activity' in day_data.columns else {},
                'hourly_breakdown': day_data.groupby('hour')['predicted_activity'].apply(list).to_dict() if 'predicted_activity' in day_data.columns else {}
            }
        
        return daily_timeline

    def _create_weekly_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze weekly activity patterns"""
        if data.empty:
            return {}
        
        data['day_of_week'] = data['datetime'].dt.day_name()
        data['week'] = data['datetime'].dt.isocalendar().week
        data['year'] = data['datetime'].dt.year
        
        weekly_patterns = {}
        
        # Activity distribution by day of week
        day_activity = data.groupby(['day_of_week', 'predicted_activity']).size().unstack(fill_value=0)
        weekly_patterns['activity_by_day'] = day_activity.to_dict()
        
        # Weekly activity volume
        weekly_volume = data.groupby(['year', 'week']).size()
        weekly_patterns['weekly_volume'] = {
            'average': weekly_volume.mean(),
            'std': weekly_volume.std(),
            'trend': self._calculate_trend(weekly_volume.values)
        }
        
        # Weekend vs Weekday patterns
        weekends = data[data['day_of_week'].isin(['Saturday', 'Sunday'])]
        weekdays = data[~data['day_of_week'].isin(['Saturday', 'Sunday'])]
        
        weekly_patterns['weekend_vs_weekday'] = {
            'weekend_activities': weekends['predicted_activity'].value_counts().to_dict() if not weekends.empty else {},
            'weekday_activities': weekdays['predicted_activity'].value_counts().to_dict() if not weekdays.empty else {},
            'weekend_avg_per_day': len(weekends) / max(len(weekends['date'].unique()), 1),
            'weekday_avg_per_day': len(weekdays) / max(len(weekdays['date'].unique()), 1)
        }
        
        return weekly_patterns
    
    def _create_monthly_overview(self, data: pd.DataFrame) -> Dict:
        """Create monthly activity overview"""
        if data.empty:
            return {}
        
        data['month'] = data['datetime'].dt.to_period('M')
        
        monthly_overview = {}
        
        for month in data['month'].unique():
            month_data = data[data['month'] == month]
            
            overview = {
                'total_activities': len(month_data),
                'unique_days': len(month_data['date'].unique()),
                'avg_activities_per_day': len(month_data) / len(month_data['date'].unique()),
                'top_activities': month_data['predicted_activity'].value_counts().head(5).to_dict(),
                'activity_distribution': self._calculate_activity_distribution(month_data),
                'busiest_day': month_data.groupby('date').size().idxmax(),
                'quietest_day': month_data.groupby('date').size().idxmin()
            }
            
            monthly_overview[str(month)] = overview
        
        return monthly_overview
    
    def _identify_major_events(self, data: pd.DataFrame) -> List[Dict]:
        """Identify major life events from activity patterns"""
        events = []
        
        if data.empty or len(data) < 10:
            return events
        
        # Detect significant activity changes
        data['date'] = data['datetime'].dt.date
        daily_counts = data.groupby('date').size()
        
        # Calculate rolling average
        rolling_avg = daily_counts.rolling(window=7, center=True).mean()
        
        # Find anomalies
        for date, count in daily_counts.items():
            expected = rolling_avg.get(date, count)
            if abs(count - expected) > 2 * daily_counts.std():
                event_type = "high_activity" if count > expected else "low_activity"
                
                events.append({
                    'date': date,
                    'type': event_type,
                    'severity': abs(count - expected) / daily_counts.std(),
                    'description': f"Unusual {event_type.replace('_', ' ')} detected",
                    'activity_count': count,
                    'expected_count': expected
                })
        
        # Detect activity type changes
        events.extend(self._detect_activity_type_changes(data))
        
        # Detect travel patterns
        events.extend(self._detect_travel_from_timeline(data))
        
        return sorted(events, key=lambda x: x['date'])
    
    def _analyze_activity_trends(self, data: pd.DataFrame) -> Dict:
        """Analyze long-term activity trends"""
        if data.empty:
            return {}
        
        trends = {}
        
        # Overall activity trend
        data['week'] = data['datetime'].dt.to_period('W')
        weekly_counts = data.groupby('week').size()
        
        if len(weekly_counts) > 2:
            trends['overall_trend'] = self._calculate_trend(weekly_counts.values)
            trends['trend_direction'] = 'increasing' if trends['overall_trend'] > 0 else 'decreasing' if trends['overall_trend'] < 0 else 'stable'
        
        # Activity type trends
        activity_trends = {}
        for activity in data['predicted_activity'].unique():
            activity_data = data[data['predicted_activity'] == activity]
            activity_weekly = activity_data.groupby('week').size()
            if len(activity_weekly) > 2:
                activity_trends[activity] = self._calculate_trend(activity_weekly.values)
        
        trends['activity_specific_trends'] = activity_trends
        
        # Time-based trends
        hourly_trends = data.groupby([data['datetime'].dt.hour]).size()
        trends['hourly_distribution'] = hourly_trends.to_dict()
        
        return trends
    
    def create_visualization(self, timeline_data: Dict) -> Dict[str, go.Figure]:
        """Create visualizations for the timeline"""
        visualizations = {}
        
        # Daily activity heatmap
        if 'weekly_patterns' in timeline_data and timeline_data['weekly_patterns']:
            visualizations['weekly_heatmap'] = self._create_weekly_heatmap(timeline_data['weekly_patterns'])
        
        # Activity trends over time
        if 'activity_trends' in timeline_data and timeline_data['activity_trends']:
            visualizations['trend_chart'] = self._create_trend_chart(timeline_data['activity_trends'])
        
        # Major events timeline
        if 'major_events' in timeline_data and timeline_data['major_events']:
            visualizations['events_timeline'] = self._create_events_timeline(timeline_data['major_events'])
        
        return visualizations
    
    def _add_activity_gaps(self, timeline: List[Dict]) -> List[Dict]:
        """Add gaps between activities to show periods of inactivity"""
        if len(timeline) < 2:
            return timeline
        
        enhanced_timeline = []
        
        for i, activity in enumerate(timeline):
            enhanced_timeline.append(activity)
            
            # Check for gaps between activities
            if i < len(timeline) - 1:
                current_time = datetime.strptime(activity['time'], '%H:%M')
                next_time = datetime.strptime(timeline[i + 1]['time'], '%H:%M')
                
                gap_duration = (next_time - current_time).total_seconds() / 3600  # hours
                
                if gap_duration > 2:  # Gap of more than 2 hours
                    enhanced_timeline.append({
                        'time': 'gap',
                        'activity': 'Inactive Period',
                        'confidence': 0,
                        'text_preview': f"No activity detected for {gap_duration:.1f} hours",
                        'is_gap': True
                    })
        
        return enhanced_timeline
    
    def _calculate_trend(self, values: np.ndarray) -> float:
        """Calculate trend using linear regression slope"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        coefficients = np.polyfit(x, values, 1)
        return coefficients[0]  # Slope
    
    def _calculate_activity_distribution(self, data: pd.DataFrame) -> Dict:
        """Calculate activity distribution statistics"""
        if 'predicted_activity' not in data.columns:
            return {}
        
        activity_counts = data['predicted_activity'].value_counts()
        total = len(data)
        
        distribution = {}
        for activity, count in activity_counts.items():
            distribution[activity] = {
                'count': count,
                'percentage': (count / total) * 100,
                'avg_confidence': data[data['predicted_activity'] == activity]['confidence'].mean()
            }
        
        return distribution
    
    def _detect_activity_type_changes(self, data: pd.DataFrame) -> List[Dict]:
        """Detect significant changes in activity types"""
        events = []
        
        if 'predicted_activity' not in data.columns:
            return events
        
        # Group by week and analyze activity type distribution
        data['week'] = data['datetime'].dt.to_period('W')
        
        weekly_activity_dist = {}
        for week in data['week'].unique():
            week_data = data[data['week'] == week]
            weekly_activity_dist[week] = week_data['predicted_activity'].value_counts(normalize=True)
        
        weeks = sorted(weekly_activity_dist.keys())
        
        for i in range(1, len(weeks)):
            current_week = weeks[i]
            previous_week = weeks[i-1]
            
            current_dist = weekly_activity_dist[current_week]
            previous_dist = weekly_activity_dist[previous_week]
            
            # Calculate distribution change
            all_activities = set(current_dist.index) | set(previous_dist.index)
            
            total_change = 0
            for activity in all_activities:
                current_pct = current_dist.get(activity, 0)
                previous_pct = previous_dist.get(activity, 0)
                total_change += abs(current_pct - previous_pct)
            
            if total_change > 0.5:  # 50% change in distribution
                events.append({
                    'date': current_week.start_time.date(),
                    'type': 'activity_pattern_change',
                    'severity': total_change,
                    'description': f"Significant change in activity patterns detected"
                })
        
        return events
    
    def _detect_travel_from_timeline(self, data: pd.DataFrame) -> List[Dict]:
        """Detect travel events from timeline patterns"""
        events = []
        
        if 'predicted_activity' not in data.columns:
            return events
        
        # Look for travel-related activities
        travel_data = data[data['predicted_activity'] == 'Travel']
        
        if travel_data.empty:
            return events
        
        # Group consecutive travel activities
        travel_data = travel_data.sort_values('datetime')
        travel_data['date'] = travel_data['datetime'].dt.date
        
        dates = travel_data['date'].unique()
        travel_periods = []
        current_period = []
        
        for i, date in enumerate(sorted(dates)):
            if not current_period or (date - current_period[-1]).days <= 1:
                current_period.append(date)
            else:
                if len(current_period) >= 1:
                    travel_periods.append(current_period)
                current_period = [date]
        
        if current_period:
            travel_periods.append(current_period)
        
        for period in travel_periods:
            if len(period) >= 2:  # Multi-day travel
                events.append({
                    'date': period[0],
                    'type': 'travel_period',
                    'severity': len(period),
                    'description': f"{len(period)}-day travel period detected",
                    'end_date': period[-1]
                })
        
        return events
    
    def _create_weekly_heatmap(self, weekly_data: Dict) -> go.Figure:
        """Create a weekly activity heatmap"""
        if not weekly_data or 'activity_by_day' not in weekly_data:
            return go.Figure()
        
        activity_by_day = weekly_data['activity_by_day']
        
        # Convert to matrix format
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        activities = list(set().union(*[day_data.keys() for day_data in activity_by_day.values()]))
        
        z_data = []
        for activity in activities:
            row = [activity_by_day.get(day, {}).get(activity, 0) for day in days]
            z_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=days,
            y=activities,
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Weekly Activity Patterns",
            xaxis_title="Day of Week",
            yaxis_title="Activity Type"
        )
        
        return fig
    
    def _create_trend_chart(self, trend_data: Dict) -> go.Figure:
        """Create activity trends chart"""
        fig = go.Figure()
        
        if 'hourly_distribution' in trend_data:
            hours = list(trend_data['hourly_distribution'].keys())
            counts = list(trend_data['hourly_distribution'].values())
            
            fig.add_trace(go.Scatter(
                x=hours,
                y=counts,
                mode='lines+markers',
                name='Hourly Activity',
                line=dict(color='blue', width=2)
            ))
        
        fig.update_layout(
            title="Activity Trends Over Time",
            xaxis_title="Hour of Day",
            yaxis_title="Activity Count",
            showlegend=True
        )
        
        return fig
    
    def _create_events_timeline(self, events: List[Dict]) -> go.Figure:
        """Create timeline visualization for major events"""
        if not events:
            return go.Figure()
        
        dates = [event['date'] for event in events]
        event_types = [event['type'] for event in events]
        descriptions = [event['description'] for event in events]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=event_types,
            mode='markers+text',
            text=descriptions,
            textposition="top center",
            marker=dict(size=12, color='red'),
            name='Major Events'
        ))
        
        fig.update_layout(
            title="Major Life Events Timeline",
            xaxis_title="Date",
            yaxis_title="Event Type",
            showlegend=True
        )
        
        return fig