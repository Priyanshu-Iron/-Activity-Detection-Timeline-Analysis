# src/api/email_client.py
import imaplib
import email
from email.header import decode_header
from typing import List, Dict, Optional
import re
from datetime import datetime
import pandas as pd

class EmailClient:
    def __init__(self, email_address: str, password: str, imap_server: str):
        """
        Initialize the EmailClient for accessing email accounts via IMAP
        
        Args:
            email_address: User's email address
            password: Email account password
            imap_server: IMAP server address (e.g., 'imap.gmail.com')
        """
        self.email_address = email_address
        self.password = password
        self.imap_server = imap_server
        self.imap = None
    
    def connect(self) -> None:
        """Establish connection to the IMAP server"""
        try:
            self.imap = imaplib.IMAP4_SSL(self.imap_server)
            self.imap.login(self.email_address, self.password)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to IMAP server: {str(e)}")
    
    def disconnect(self) -> None:
        """Close the IMAP connection"""
        if self.imap:
            try:
                self.imap.logout()
            except:
                pass
            self.imap = None
    
    def fetch_emails(self, folder: str = "INBOX", max_emails: int = 100) -> List[Dict]:
        """
        Fetch emails from specified folder
        
        Args:
            folder: Email folder to fetch from (default: INBOX)
            max_emails: Maximum number of emails to fetch
        
        Returns:
            List of dictionaries containing email data
        """
        if not self.imap:
            self.connect()
        
        try:
            self.imap.select(folder)
            _, message_numbers = self.imap.search(None, "ALL")
            
            emails = []
            for num in message_numbers[0].split()[:max_emails]:
                _, msg_data = self.imap.fetch(num, "(RFC822)")
                email_body = msg_data[0][1]
                msg = email.message_from_bytes(email_body)
                
                # Decode subject
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or "utf-8")
                
                # Get email body
                body = self._get_email_body(msg)
                
                # Extract timestamp
                date_str = msg["Date"]
                timestamp = self._parse_email_date(date_str)
                
                emails.append({
                    "timestamp": timestamp,
                    "subject": subject,
                    "from": msg["From"],
                    "text": body
                })
            
            return emails
            
        except Exception as e:
            print(f"Error fetching emails: {str(e)}")
            return []
            
        finally:
            self.disconnect()
    
    def _get_email_body(self, msg: email.message.Message) -> str:
        """Extract plain text body from email"""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode()
        else:
            return msg.get_payload(decode=True).decode()
        return ""
    
    def _parse_email_date(self, date_str: str) -> Optional[datetime]:
        """Parse email date string to datetime object"""
        try:
            parsed_date = email.utils.parsedate_to_datetime(date_str)
            return parsed_date
        except:
            return datetime.now()
    
    def fetch_emails_to_dataframe(self, folder: str = "INBOX", max_emails: int = 100) -> pd.DataFrame:
        """
        Fetch emails and return as a pandas DataFrame
        
        Args:
            folder: Email folder to fetch from
            max_emails: Maximum number of emails to fetch
        
        Returns:
            DataFrame with email data
        """
        emails = self.fetch_emails(folder, max_emails)
        return pd.DataFrame(emails)