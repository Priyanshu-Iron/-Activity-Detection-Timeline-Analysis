import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import json

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.activity_classifier import ActivityClassifier
from config.settings import ACTIVITY_LABELS

def show_activity_detection():
    st.title("📧📱 Real-Time Activity Detection")
    st.markdown("Enter text to classify activities using AI models via Hugging Face APIs.")
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = ActivityClassifier()
    
    # Initialize history
    if 'classification_history' not in st.session_state:
        st.session_state.classification_history = []
    
    # Sidebar options
    st.sidebar.subheader("⚙️ Settings")
    
    # Activity category selection
    category = st.sidebar.selectbox(
        "Choose activity category:",
        list(ACTIVITY_LABELS.keys()),
        index=2  # Default to general_activities
    )
    
    # Show available labels for selected category
    with st.sidebar.expander("📋 Available Activity Labels"):
        labels = ACTIVITY_LABELS[category]
        for label in labels:
            st.write(f"• {label}")
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Minimum confidence level for high-confidence predictions"
    )
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Single Text", "📝 Batch Processing", "📊 Analysis History"])
    
    with tab1:
        show_single_text_classification(category, confidence_threshold)
    
    with tab2:
        show_batch_processing(category, confidence_threshold)
    
    with tab3:
        show_analysis_history()

def show_single_text_classification(category, confidence_threshold):
    st.subheader("🎯 Single Text Classification")
    
    # Sample texts for quick testing
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_area(
            "Enter your text here:",
            height=100,
            placeholder="Example: Just finished my morning workout at the gym. Time for a protein shake!"
        )
    
    with col2:
        st.markdown("**Quick Examples:**")
        example_texts = [
            "Meeting with the team at 2 PM",
            "Going for a run in the park",
            "Cooking dinner for the family",
            "Watching Netflix after work",
            "Shopping for groceries",
            "Video call with friends"
        ]
        
        for i, example in enumerate(example_texts):
            if st.button(f"📝 Example {i+1}", key=f"example_{i}"):
                st.session_state.example_text = example
    
    # Use example text if selected
    if 'example_text' in st.session_state:
        user_input = st.session_state.example_text
        del st.session_state.example_text
    
    # Classification button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col2:
        classify_button = st.button("🚀 Classify Activity", type="primary")
    
    if classify_button and user_input.strip():
        with st.spinner("🤖 Analyzing with AI..."):
            try:
                result = st.session_state.classifier.classify_single_text(user_input, category)
                
                if 'error' not in result:
                    # Add to history
                    history_entry = {
                        'timestamp': datetime.now(),
                        'text': user_input,
                        'category': category,
                        'result': result
                    }
                    st.session_state.classification_history.append(history_entry)
                    
                    # Display results
                    display_classification_result(result, confidence_threshold)
                else:
                    st.error(f"❌ Classification failed: {result['error']}")
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    
    elif classify_button and not user_input.strip():
        st.warning("⚠️ Please enter some text to classify.")

def show_batch_processing(category, confidence_threshold):
    st.subheader("📝 Batch Processing")
    
    st.markdown("Process multiple texts at once for bulk activity classification.")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["📝 Text Area", "📁 File Upload"],
        horizontal=True
    )
    
    texts_to_process = []
    
    if input_method == "📝 Text Area":
        batch_input = st.text_area(
            "Enter multiple texts (one per line):",
            height=200,
            placeholder="Line 1: Had coffee with Sarah at Starbucks\nLine 2: Attended team meeting via Zoom\nLine 3: Went grocery shopping at Whole Foods"
        )
        
        if batch_input.strip():
            texts_to_process = [line.strip() for line in batch_input.split('\n') if line.strip()]
    
    else:  # File Upload
        uploaded_file = st.file_uploader(
            "Upload a text file (.txt) or CSV file:",
            type=['txt', 'csv'],
            help="For CSV files, make sure there's a 'text' column"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.txt'):
                    content = str(uploaded_file.read(), "utf-8")
                    texts_to_process = [line.strip() for line in content.split('\n') if line.strip()]
                
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    if 'text' in df.columns:
                        texts_to_process = df['text'].dropna().tolist()
                    else:
                        st.error("❌ CSV file must contain a 'text' column")
                        return
                        
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                return
    
    # Display preview
    if texts_to_process:
        st.info(f"📊 Found {len(texts_to_process)} texts to process")
        
        with st.expander("👀 Preview texts"):
            for i, text in enumerate(texts_to_process[:5]):  # Show first 5
                st.write(f"{i+1}. {text}")
            if len(texts_to_process) > 5:
                st.write(f"... and {len(texts_to_process) - 5} more")
    
    # Process button
    if texts_to_process:
        if st.button("🚀 Process All Texts", type="primary"):
            with st.spinner(f"🤖 Processing {len(texts_to_process)} texts..."):
                try:
                    results = st.session_state.classifier.classify_batch(texts_to_process, category)
                    
                    # Add to history
                    for i, result in enumerate(results):
                        if 'error' not in result:
                            history_entry = {
                                'timestamp': datetime.now(),
                                'text': texts_to_process[i],
                                'category': category,
                                'result': result
                            }
                            st.session_state.classification_history.append(history_entry)
                    
                    # Display batch results
                    display_batch_results(results, texts_to_process, confidence_threshold)
                    
                except Exception as e:
                    st.error(f"❌ Batch processing failed: {str(e)}")

def show_analysis_history():
    st.subheader("📊 Analysis History")
    
    if not st.session_state.classification_history:
        st.info("📝 No classifications yet. Start by classifying some text!")
        return
    
    # History controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🗑️ Clear History"):
            st.session_state.classification_history = []
            st.success("✅ History cleared!")
            st.rerun()
    
    with col2:
        if st.button("📥 Download History"):
            download_history()
    
    # Statistics
    st.subheader("📈 Statistics")
    
    history_df = pd.DataFrame([
        {
            'timestamp': entry['timestamp'],
            'text': entry['text'][:50] + "..." if len(entry['text']) > 50 else entry['text'],
            'category': entry['category'],
            'predicted_activity': entry['result'].get('predicted_activity', 'Unknown'),
            'confidence': entry['result'].get('confidence', 0)
        }
        for entry in st.session_state.classification_history
        if 'error' not in entry['result']
    ])
    
    if not history_df.empty:
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Classifications", len(history_df))
        
        with col2:
            avg_confidence = history_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        with col3:
            high_conf_count = len(history_df[history_df['confidence'] > 0.8])
            st.metric("High Confidence", high_conf_count)
        
        with col4:
            unique_activities = history_df['predicted_activity'].nunique()
            st.metric("Unique Activities", unique_activities)
        
        # Activity distribution chart
        st.subheader("🎯 Activity Distribution")
        activity_counts = history_df['predicted_activity'].value_counts()
        
        fig = px.pie(
            values=activity_counts.values,
            names=activity_counts.index,
            title="Distribution of Predicted Activities"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence distribution
        st.subheader("📊 Confidence Distribution")
        fig = px.histogram(
            history_df,
            x='confidence',
            nbins=20,
            title="Distribution of Classification Confidence Scores"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent classifications table
        st.subheader("🕐 Recent Classifications")
        recent_df = history_df.sort_values('timestamp', ascending=False).head(10)
        st.dataframe(recent_df, use_container_width=True)

def display_classification_result(result, confidence_threshold):
    """Display the results of a single classification"""
    st.success("✅ Classification Complete!")
    
    # Main result
    col1, col2 = st.columns([2, 1])
    
    with col1:
        confidence = result['confidence']
        activity = result['predicted_activity']
        
        # Confidence indicator
        if confidence > 0.8:
            confidence_color = "🟢"
            confidence_text = "High"
        elif confidence > 0.5:
            confidence_color = "🟡"
            confidence_text = "Medium"
        else:
            confidence_color = "🔴"
            confidence_text = "Low"
        
        st.markdown(f"""
        ### 🎯 **{activity}**
        {confidence_color} **Confidence:** {confidence:.2%} ({confidence_text})
        """)
    
    with col2:
        # Confidence gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence_threshold * 100
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    # All predictions
    with st.expander("📊 All Predictions"):
        all_preds = result.get('all_predictions', {})
        pred_df = pd.DataFrame([
            {'Activity': activity, 'Confidence': conf}
            for activity, conf in all_preds.items()
        ]).sort_values('Confidence', ascending=False)
        
        fig = px.bar(
            pred_df,
            x='Confidence',
            y='Activity',
            orientation='h',
            title="All Activity Predictions",
            color='Confidence',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    if 'insights' in result:
        with st.expander("💡 Insights"):
            insights = result['insights']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Confidence Level:** {insights.get('confidence_level', 'Unknown')}")
                st.write(f"**Activity Type:** {insights.get('activity_type', 'Unknown')}")
            
            with col2:
                keywords = insights.get('keywords_found', [])
                if keywords:
                    st.write("**Keywords Found:**")
                    for keyword in keywords:
                        st.write(f"• {keyword}")
                else:
                    st.write("**Keywords Found:** None")

def display_batch_results(results, texts, confidence_threshold):
    """Display results from batch processing"""
    st.success(f"✅ Processed {len(results)} texts!")
    
    # Summary statistics
    successful_results = [r for r in results if 'error' not in r]
    error_count = len(results) - len(successful_results)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Processed", len(results))
    
    with col2:
        st.metric("Successful", len(successful_results))
    
    with col3:
        st.metric("Errors", error_count)
    
    with col4:
        if successful_results:
            avg_conf = sum(r['confidence'] for r in successful_results) / len(successful_results)
            st.metric("Avg Confidence", f"{avg_conf:.2%}")
    
    if successful_results:
        # Create results dataframe
        results_df = pd.DataFrame([
            {
                'Text': texts[i][:50] + "..." if len(texts[i]) > 50 else texts[i],
                'Full_Text': texts[i],
                'Predicted_Activity': result['predicted_activity'],
                'Confidence': result['confidence'],
                'High_Confidence': result['confidence'] > confidence_threshold
            }
            for i, result in enumerate(results)
            if 'error' not in result
        ])
        
        # Activity distribution
        st.subheader("📊 Activity Distribution")
        activity_dist = results_df['Predicted_Activity'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=activity_dist.values,
                names=activity_dist.index,
                title="Activity Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                results_df,
                x='Confidence',
                nbins=15,
                title="Confidence Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Results table
        st.subheader("📋 Detailed Results")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            show_only_high_conf = st.checkbox("Show only high confidence results")
        
        with col2:
            selected_activities = st.multiselect(
                "Filter by activity:",
                options=results_df['Predicted_Activity'].unique(),
                default=results_df['Predicted_Activity'].unique()
            )
        
        # Apply filters
        filtered_df = results_df.copy()
        
        if show_only_high_conf:
            filtered_df = filtered_df[filtered_df['High_Confidence']]
        
        if selected_activities:
            filtered_df = filtered_df[filtered_df['Predicted_Activity'].isin(selected_activities)]
        
        # Display table
        st.dataframe(
            filtered_df[['Text', 'Predicted_Activity', 'Confidence']],
            use_container_width=True
        )
        
        # Download option
        if st.button("📥 Download Results as CSV"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"activity_classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Show errors if any
    if error_count > 0:
        with st.expander(f"❌ Errors ({error_count})"):
            for i, result in enumerate(results):
                if 'error' in result:
                    st.error(f"Text {i+1}: {result['error']}")

def download_history():
    """Prepare history data for download"""
    if not st.session_state.classification_history:
        st.warning("No history to download")
        return
    
    # Prepare data for download
    download_data = []
    for entry in st.session_state.classification_history:
        if 'error' not in entry['result']:
            download_data.append({
                'timestamp': entry['timestamp'].isoformat(),
                'text': entry['text'],
                'category': entry['category'],
                'predicted_activity': entry['result']['predicted_activity'],
                'confidence': entry['result']['confidence'],
                'high_confidence': entry['result']['high_confidence']
            })
    
    if download_data:
        df = pd.DataFrame(download_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download History CSV",
            data=csv,
            file_name=f"activity_detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No successful classifications to download")