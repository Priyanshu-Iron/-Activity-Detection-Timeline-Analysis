import streamlit as st
import pandas as pd
from datetime import datetime
import os
from src.data.preprocessor import DataPreprocessor
from src.utils.helpers import validate_data

def show_data_upload():
    st.title("📁 Data Upload")
    st.markdown("Upload your email or social media data for in-depth analysis.")
    
    # Initialize preprocessor
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = DataPreprocessor()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV or JSON file",
        type=['csv', 'json'],
        help="Upload a file containing text and timestamp data"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            
            # Validate data
            required_columns = ['text', 'timestamp']
            if not validate_data(df, required_columns):
                st.error("❌ File must contain 'text' and 'timestamp' columns")
                return
            
            # Preprocess data
            with st.spinner("🔄 Processing data..."):
                processed_df = st.session_state.preprocessor.preprocess_dataframe(df)
            
            # Display preview
            st.success("✅ Data uploaded and processed successfully!")
            st.session_state.uploaded_data = processed_df
            
            st.subheader("📊 Data Preview")
            st.dataframe(processed_df.head(), use_container_width=True, height=300)
            
            # Download processed data
            csv = processed_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Processed Data",
                data=csv,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    # Display instructions
    with st.expander("ℹ️ How to Prepare Your Data"):
        st.markdown("""
        #### Data Format Requirements
        
        - **Supported Formats**: CSV or JSON
        - **Required Columns**:
          - `text`: The content to analyze (e.g., email body, social media post)
          - `timestamp`: Date and time of the activity (ISO format preferred)
        - **Optional Columns**: Any additional metadata like sender, subject, etc.
        
        #### Example CSV Format:
        ```
        text,timestamp
        "Had a great meeting with the team today","2024-01-15 09:30:00"
        "Going for a run in the park","2024-01-15 18:00:00"
        "Dinner with family at home","2024-01-15 19:30:00"
        ```
        
        #### Example JSON Format:
        ```json
        [
            {
                "text": "Had a great meeting with the team today",
                "timestamp": "2024-01-15T09:30:00"
            },
            {
                "text": "Going for a run in the park", 
                "timestamp": "2024-01-15T18:00:00"
            }
        ]
        ```
        
        #### Tips:
        - Use standard timestamp formats (ISO 8601 recommended)
        - Ensure text is descriptive for accurate classification
        - Remove sensitive information before uploading
        """)