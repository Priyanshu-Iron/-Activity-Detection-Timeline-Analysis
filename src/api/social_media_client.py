# src/api/social_media_client.py
import tweepy
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime

class SocialMediaClient:
    def __init__(self, api_key: str, api_secret: str, access_token: str, access_token_secret: str):
        """
        Initialize the SocialMediaClient for accessing social media APIs (Twitter/X focus)
        
        Args:
            api_key: API key for authentication
            api_secret: API secret key
            access_token: Access token for authentication
            access_token_secret: Access token secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.api = None
    
    def connect(self) -> None:
        """Establish connection to the Twitter/X API"""
        try:
            auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
            auth.set_access_token(self.access_token, self.access_token_secret)
            self.api = tweepy.API(auth, wait_on_rate_limit=True)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to social media API: {str(e)}")
    
    def fetch_user_posts(self, username: str, max_posts: int = 100) -> List[Dict]:
        """
        Fetch posts from a specific user
        
        Args:
            username: Social media username
            max_posts: Maximum number of posts to fetch
        
        Returns:
            List of dictionaries containing post data
        """
        if not self.api:
            self.connect()
        
        try:
            posts = []
            for tweet in tweepy.Cursor(self.api.user_timeline, screen_name=username, tweet_mode="extended").items(max_posts):
                posts.append({
                    "timestamp": tweet.created_at,
                    "text": tweet.full_text,
                    "source": "Twitter",
                    "post_id": tweet.id_str
                })
            return posts
        except Exception as e:
            print(f"Error fetching posts: {str(e)}")
            return []
    
    def fetch_posts_to_dataframe(self, username: str, max_posts: int = 100) -> pd.DataFrame:
        """
        Fetch posts and return as a pandas DataFrame
        
        Args:
            username: Social media username
            max_posts: Maximum number of posts to fetch
        
        Returns:
            DataFrame with post data
        """
        posts = self.fetch_user_posts(username, max_posts)
        return pd.DataFrame(posts)