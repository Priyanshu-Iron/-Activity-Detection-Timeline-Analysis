import streamlit as st
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from streamlit_app.pages.activity_detection import show_activity_detection
from streamlit_app.pages.timeline_analysis import show_timeline_analysis
from streamlit_app.pages.data_upload import show_data_upload

# Page configuration
st.set_page_config(
    page_title="Activity Detection & Timeline Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .feature-card {
        background: var(--background-color, rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        border-color: rgba(103, 126, 234, 0.3);
    }
    
    .feature-card h3 {
        color: var(--text-color);
        margin-bottom: 0.5rem;
    }
    
    .feature-card p {
        color: var(--text-color);
        opacity: 0.8;
        line-height: 1.4;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: scale(1.05);
    }
    
    /* Dark theme specific adjustments */
    [data-testid="stAppViewContainer"] {
        --background-color: rgba(255, 255, 255, 0.05);
        --text-color: inherit;
    }
    
    /* Light theme fallback */
    @media (prefers-color-scheme: light) {
        .feature-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(0, 0, 0, 0.1);
        }
    }
    
    /* Additional dark mode detection */
    .stApp[data-theme="dark"] .feature-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stApp[data-theme="light"] .feature-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📧 Activity Detection", "📅 Timeline Analysis", "📁 Data Upload"]
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "📧 Activity Detection":
        show_activity_detection()
    elif page == "📅 Timeline Analysis":
        show_timeline_analysis()
    elif page == "📁 Data Upload":
        show_data_upload()

def show_home():
    # Main header
    st.markdown('<h1 class="main-header">📧📱 Activity Detection & Timeline Analysis</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    Welcome to the **Activity Detection System**! This application uses advanced machine learning 
    to analyze your daily activities from email and social media data, helping you understand 
    your patterns and detect major life events.
    """)
    
    # Features overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Activity Classification</h3>
            <p>Automatically classify your activities from text using state-of-the-art NLP models via Hugging Face API.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Pattern Recognition</h3>
            <p>Discover your daily routines, sleep patterns, and activity trends over time.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🎉 Life Events Detection</h3>
            <p>Automatically detect major life events like job changes, travel, and significant activity shifts.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works
    st.subheader("🚀 How It Works")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 1. Data Input
        - Upload your email exports or social media data
        - Or manually input text for real-time classification
        - Supports CSV, JSON, and text formats
        
        ### 2. AI Analysis
        - Uses Facebook's BART model for activity classification
        - Analyzes temporal patterns and contexts
        - Detects anomalies and significant changes
        """)
    
    with col2:
        st.markdown("""
        ### 3. Insights Generation
        - Creates detailed activity timelines
        - Identifies daily routines and patterns
        - Detects major life events automatically
        
        ### 4. Visualization
        - Interactive charts and graphs
        - Timeline visualizations
        - Pattern analysis dashboards
        """)
    
    # Sample use cases
    st.subheader("💡 Use Cases")
    
    use_cases = [
        {
            "title": "📊 Personal Analytics",
            "description": "Track your daily habits and productivity patterns to optimize your routine."
        },
        {
            "title": "🔄 Life Event Detection", 
            "description": "Automatically identify when you changed jobs, went on vacation, or moved locations."
        },
        {
            "title": "⏰ Sleep Pattern Analysis",
            "description": "Understand your sleep schedule and activity patterns throughout the day."
        },
        {
            "title": "🌍 Travel Detection",
            "description": "Detect travel periods and timezone changes from your activity data."
        }
    ]
    
    cols = st.columns(2)
    for i, use_case in enumerate(use_cases):
        with cols[i % 2]:
            st.info(f"**{use_case['title']}**\n\n{use_case['description']}")
    
    # Getting started
    st.subheader("🏁 Getting Started")
    
    st.markdown("""
    1. **Start Simple**: Go to the "Activity Detection" page to test single text classification
    2. **Upload Data**: Use the "Data Upload" page to analyze your email or social media exports
    3. **Explore Timeline**: Check the "Timeline Analysis" page for comprehensive pattern analysis
    4. **Interpret Results**: Look for insights about your daily routines and life events
    """)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        ### Models Used
        - **Primary Classification**: Facebook BART-large-MNLI
        - **Sentiment Analysis**: Cardiff Twitter-RoBERTa
        - **Text Summarization**: Facebook BART-large-CNN
        
        ### Features
        - ✅ Real-time API-based classification (no local models required)
        - ✅ Batch processing for large datasets
        - ✅ Temporal pattern analysis
        - ✅ Anomaly detection for life events
        - ✅ Interactive visualizations
        - ✅ Export capabilities for results
        
        ### Privacy & Security
        - 🔒 Your data is processed securely through Hugging Face APIs
        - 🔒 No data is stored permanently on our servers
        - 🔒 All processing happens in real-time
        """)
    
    # Quick stats (mock data for demo)
    st.subheader("📊 Quick Demo Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h2>95%</h2>
            <p>Classification Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h2>12</h2>
            <p>Activity Categories</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h2>24/7</h2>
            <p>Timeline Coverage</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <h2>API</h2>
            <p>Cloud-Based</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()