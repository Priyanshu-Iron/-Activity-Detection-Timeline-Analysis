
# 🕵️ Activity Detection & Timeline Analysis

A smart NLP-driven system that extracts, classifies, and visualizes user activity timelines from emails or raw text using machine learning and Hugging Face's powerful inference APIs.

---

## 🚀 Features

- 📥 **Email Extraction** using IMAP
- ✂️ **Text Preprocessing** with NLTK & TextBlob
- 🧠 **Activity Classification** via Hugging Face (zero-shot)
- 💬 **Sentiment Analysis** of activities
- 🧩 **Pattern Recognition** for behavior clustering
- 📆 **Timeline Construction** based on extracted events
- 📊 **Interactive Dashboard** using Streamlit & Plotly

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML/NLP**: Hugging Face Transformers, Scikit-learn, TextBlob, NLTK
- **Visualization**: Plotly
- **Data Handling**: Pandas, NumPy
- **Email Handling**: imaplib2, email-validator
- **Config**: dotenv-based configuration

---

## 📂 Project Structure

```
├── main.py                   # Streamlit entry point
├── activity_classifier.py    # Classify activities using ML/API
├── pattern_analyzer.py       # Behavioral patterns and trends
├── timeline_builder.py       # Build timelines from events
├── preprocessor.py           # Clean and normalize text
├── visualizer.py             # Plotly visual components
├── widgets.py                # Custom UI widgets for Streamlit
├── helpers.py                # Common helper functions
├── config/
│   └── settings.py           # API tokens, model names, and constants
├── requirements.txt          # Dependencies list
```

---

## 🤗 Hugging Face Integration

This project uses Hugging Face's Inference API for the following:

- **Zero-shot Activity Classification**
- **Sentiment Analysis**
- **Summarization / Time Extraction**

### 🔐 Authentication

Store your API key in `config/settings.py`:

```python
HUGGINGFACE_API_TOKEN = "your_token_here"
```

Set model names in the same file:

```python
MODELS = {
    "activity_classification": "facebook/bart-large-mnli",
    "sentiment_analysis": "distilbert-base-uncased-finetuned-sst-2-english",
    "time_extraction": "facebook/bart-large-cnn"
}
```

### ✅ Sample Usage

```python
from src.huggingface_client import HuggingFaceClient

client = HuggingFaceClient()
activity = client.classify_activity("Went for a run", ["work", "exercise", "social"])
sentiment = client.analyze_sentiment("I enjoyed the meeting")
summary = client.extract_summary("This is a long message...")
```

Built-in retry logic and rate-limiting ensure smooth API calls even if the model is still loading.

---

## 📸 Sample Visuals

- Timeline charts of daily activity
- Pie charts for activity types
- Trend lines and clustering graphs

*(Add visuals in the repo for better illustration)*

---

## ⚙️ How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Priyanshu-Iron/-Activity-Detection-Timeline-Analysis.git
   cd -Activity-Detection-Timeline-Analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   streamlit run main.py
   ```

---

## 🧪 Use Cases

- 📨 Analyzing personal or team email activity
- 👨‍💼 Understanding work patterns and productivity
- 🧘‍♀️ Behavioral analysis for well-being
- 🔍 Detecting anomalies or unexpected trends

---

## 🙌 Author

- **Priyanshu Kumar Singh** — [@Priyanshu-Iron](https://github.com/Priyanshu-Iron)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 💡 Future Enhancements

- Real-time Gmail/Outlook sync
- Custom activity tags & training
- Exportable timeline reports
- Role-based dashboards for teams