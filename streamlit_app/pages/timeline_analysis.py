import streamlit as st
import pandas as pd
from datetime import datetime
from src.models.activity_classifier import ActivityClassifier
from src.models.timeline_builder import TimelineBuilder
from src.data.pattern_analyzer import PatternAnalyzer
from src.utils.visualizer import create_activity_pie_chart, create_confidence_histogram, create_timeline_chart

def show_timeline_analysis():
    st.title("📅 Timeline Analysis")
    st.markdown("Explore patterns and visualize timelines from your activity data.")
    
    # Initialize components
    if 'classifier' not in st.session_state:
        st.session_state.classifier = ActivityClassifier()
    if 'timeline_builder' not in st.session_state:
        st.session_state.timeline_builder = TimelineBuilder()
    if 'pattern_analyzer' not in st.session_state:
        st.session_state.pattern_analyzer = PatternAnalyzer()
    
    # Check for uploaded data
    if 'uploaded_data' not in st.session_state or st.session_state.uploaded_data.empty:
        st.warning("⚠️ Please upload data on the Data Upload page first.")
        return
    
    df = st.session_state.uploaded_data
    
    # Classify activities
    with st.spinner("🤖 Analyzing activities..."):
        classified_df = st.session_state.classifier.classify_with_context(df)
    
    # Analyze patterns
    with st.spinner("🔍 Detecting patterns..."):
        patterns = st.session_state.pattern_analyzer.analyze_daily_routine(classified_df)
        life_events = st.session_state.pattern_analyzer.detect_life_events(classified_df)
        insights = st.session_state.pattern_analyzer.generate_insights(patterns, life_events)
    
    # Build timeline
    with st.spinner("📅 Generating timeline..."):
        timeline = st.session_state.timeline_builder.build_activity_timeline(classified_df)
        visualizations = st.session_state.timeline_builder.create_visualization(timeline)
    
    # Display results
    st.subheader("📊 Analysis Results", anchor=False)
    
    # Daily patterns
    st.markdown("### 🕒 Daily Patterns")
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("**Wake-up Time**")
        wake_up = patterns.get('wake_up_time', {})
        avg_wake = wake_up.get('average', 0)
        common_wake = wake_up.get('most_common', 0)
        st.markdown(f"**Average**: {avg_wake:.1f} hours" if avg_wake else "**Average**: N/A")
        st.markdown(f"**Most Common**: {common_wake} hours" if common_wake else "**Most Common**: N/A")
    
    with col2:
        st.markdown("**Sleep Time**")
        sleep_time = patterns.get('sleep_time', {})
        avg_sleep = sleep_time.get('average', 0)
        common_sleep = sleep_time.get('most_common', 0)
        st.markdown(f"**Average**: {avg_sleep:.1f} hours" if avg_sleep else "**Average**: N/A")
        st.markdown(f"**Most Common**: {common_sleep} hours" if common_sleep else "**Most Common**: N/A")
    
    # Insights
    st.markdown("### 💡 Insights")
    for key, insight in insights.items():
        st.info(f"**{key.replace('_', ' ').title()}**: {insight}")
    
    # Life events
    st.markdown("### 🎉 Life Events")
    if life_events:
        events_df = pd.DataFrame(life_events)
        st.dataframe(events_df[['date', 'type', 'description', 'confidence']], use_container_width=True, height=300)
    else:
        st.info("No significant life events detected.")
    
    # Visualizations
    st.markdown("### 📈 Visualizations")
    for viz_name, fig in visualizations.items():
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional charts
    st.markdown("### 📊 Additional Charts")
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.plotly_chart(create_activity_pie_chart(classified_df), use_container_width=True)
    with col2:
        st.plotly_chart(create_confidence_histogram(classified_df), use_container_width=True)
    
    # Download results
    st.markdown("### 📥 Download Results")
    csv = classified_df.to_csv(index=False)
    st.download_button(
        label="Download Classified Data",
        data=csv,
        file_name=f"timeline_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        type="primary"
    )