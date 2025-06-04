# src/utils/visualizer.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List

def create_activity_pie_chart(data: pd.DataFrame, column: str = 'predicted_activity') -> go.Figure:
    """
    Create a pie chart for activity distribution
    
    Args:
        data: DataFrame with activity data
        column: Column name containing activities
    
    Returns:
        Plotly Figure
    """
    activity_counts = data[column].value_counts()
    fig = px.pie(
        values=activity_counts.values,
        names=activity_counts.index,
        title="Activity Distribution"
    )
    return fig

def create_confidence_histogram(data: pd.DataFrame, column: str = 'confidence') -> go.Figure:
    """
    Create a histogram for confidence scores
    
    Args:
        data: DataFrame with confidence scores
        column: Column name containing confidence scores
    
    Returns:
        Plotly Figure
    """
    fig = px.histogram(
        data,
        x=column,
        nbins=20,
        title="Confidence Score Distribution",
        labels={column: "Confidence Score"}
    )
    return fig

def create_timeline_chart(data: pd.DataFrame) -> go.Figure:
    """
    Create a timeline chart for activities
    
    Args:
        data: DataFrame with datetime and activity columns
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['datetime'],
        y=data['predicted_activity'],
        mode='markers+text',
        text=data['predicted_activity'],
        textposition="top center",
        marker=dict(size=10)
    ))
    fig.update_layout(
        title="Activity Timeline",
        xaxis_title="Date/Time",
        yaxis_title="Activity",
        showlegend=False
    )
    return fig