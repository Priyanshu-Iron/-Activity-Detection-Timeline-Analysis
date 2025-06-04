# streamlit_app/components/widgets.py
import streamlit as st
import pandas as pd

def display_metrics(metrics: dict[str, float], title: str = "Metrics") -> None:
    """
    Display metrics in a formatted layout
    
    Args:
        metrics: Dictionary of metric names and values
        title: Title for the metrics section
    """
    st.subheader(title)
    cols = st.columns(len(metrics), gap="medium")
    for i, (metric_name, value) in enumerate(metrics.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-container">
                <h2>{value:.2f}</h2>
                <p>{metric_name}</p>
            </div>
            """, unsafe_allow_html=True)

def display_data_preview(df: pd.DataFrame, title: str = "Data Preview") -> None:
    """
    Display a preview of the DataFrame
    
    Args:
        df: Input DataFrame
        title: Title for the preview section
    """
    st.subheader(title)
    st.dataframe(df.head(), use_container_width=True, height=300)