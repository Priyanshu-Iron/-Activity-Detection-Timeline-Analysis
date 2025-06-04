import requests
import time
from typing import Dict, List, Any, Optional
from config.settings import HUGGINGFACE_API_TOKEN, MODELS

class HuggingFaceClient:
    def __init__(self):
        self.api_token = HUGGINGFACE_API_TOKEN
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
        self.base_url = "https://api-inference.huggingface.co/models/"
    
    def query_model(self, model_name: str, payload: Dict[str, Any], max_retries: int = 3) -> Optional[Dict]:
        """Query a Hugging Face model with retry logic"""
        url = f"{self.base_url}{model_name}"
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=self.headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    time.sleep(10)
                    continue
                else:
                    print(f"API Error {response.status_code}: {response.text}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        
        return None
    
    def classify_activity(self, text: str, candidate_labels: List[str]) -> Optional[Dict]:
        """Classify text into activity categories"""
        payload = {
            "inputs": text,
            "parameters": {"candidate_labels": candidate_labels}
        }
        return self.query_model(MODELS["activity_classification"], payload)
    
    def analyze_sentiment(self, text: str) -> Optional[Dict]:
        """Analyze sentiment of text"""
        payload = {"inputs": text}
        return self.query_model(MODELS["sentiment_analysis"], payload)
    
    def extract_summary(self, text: str, max_length: int = 150) -> Optional[Dict]:
        """Extract summary from long text"""
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": max_length,
                "min_length": 30,
                "do_sample": False
            }
        }
        return self.query_model(MODELS["time_extraction"], payload)
    
    def batch_classify(self, texts: List[str], candidate_labels: List[str]) -> List[Optional[Dict]]:
        """Batch classify multiple texts"""
        results = []
        for text in texts:
            result = self.classify_activity(text, candidate_labels)
            results.append(result)
            time.sleep(0.5)  # Rate limiting
        return results