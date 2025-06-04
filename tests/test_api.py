# tests/test_api.py
import unittest
from src.api.huggingface_client import HuggingFaceClient
from src.api.email_client import EmailClient
from src.api.social_media_client import SocialMediaClient
from config.settings import ACTIVITY_LABELS

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.hf_client = HuggingFaceClient()
    
    def test_huggingface_classification(self):
        """Test HuggingFace activity classification"""
        test_text = "Meeting with team at 2 PM"
        labels = ACTIVITY_LABELS["general_activities"]
        result = self.hf_client.classify_activity(test_text, labels)
        
        self.assertIsNotNone(result)
        self.assertIn('labels', result)
        self.assertIn('scores', result)
        self.assertEqual(len(result['labels']), len(labels))
    
    def test_huggingface_sentiment(self):
        """Test HuggingFace sentiment analysis"""
        test_text = "Had an awesome day at work!"
        result = self.hf_client.analyze_sentiment(test_text)
        
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, dict))
    
    def test_email_client(self):
        """Test email client (mock)"""
        # Note: Requires valid credentials for real testing
        email_client = EmailClient(
            email_address="test@example.com",
            password="password",
            imap_server="imap.example.com"
        )
        # Mock test - actual testing would require a test email account
        self.assertEqual(email_client.email_address, "test@example.com")
    
    def test_social_media_client(self):
        """Test social media client (mock)"""
        # Note: Requires valid API credentials for real testing
        social_client = SocialMediaClient(
            api_key="key",
            api_secret="secret",
            access_token="token",
            access_token_secret="token_secret"
        )
        # Mock test - actual testing would require valid API credentials
        self.assertEqual(social_client.api_key, "key")

if __name__ == '__main__':
    unittest.main()