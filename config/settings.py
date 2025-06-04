import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "Your Token")
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

# Alternative models for different tasks
MODELS = {
    "activity_classification": "facebook/bart-large-mnli",
    "sentiment_analysis": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "time_extraction": "facebook/bart-large-cnn",
    "topic_modeling": "sentence-transformers/all-MiniLM-L6-v2"
}

# Activity Categories
ACTIVITY_LABELS = {
    "daily_routine": [
        "Sleep", "Wake up", "Morning routine", "Breakfast", "Commuting",
        "Work", "Lunch", "Exercise", "Dinner", "Evening routine"
    ],
    "life_events": [
        "Job change", "Travel", "Vacation", "Moving", "Relationship",
        "Education", "Health", "Family events", "Social gathering"
    ],
    "general_activities": [
        "Work", "Travel", "Shopping", "Socializing", "Studying", 
        "Entertainment", "Exercise", "Eating", "Sleeping"
    ]
}

# Time zone detection keywords
TIMEZONE_KEYWORDS = {
    "EST": ["new york", "eastern", "est", "edt"],
    "PST": ["california", "pacific", "pst", "pdt", "los angeles"],
    "GMT": ["london", "uk", "gmt", "utc", "england"],
    "CET": ["berlin", "paris", "cet", "cest", "europe"],
    "JST": ["japan", "tokyo", "jst"],
    "IST": ["india", "mumbai", "delhi", "ist"]
}

# Data processing settings
MAX_TEXT_LENGTH = 512
CONFIDENCE_THRESHOLD = 0.5
BATCH_SIZE = 10