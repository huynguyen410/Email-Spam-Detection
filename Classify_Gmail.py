import os
import base64
import re
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from google.auth.transport.requests import Request
import joblib
from transformers import pipeline

# Load your trained model and TF-IDF Vectorizer
model = joblib.load('spam_classifier_model.joblib')
feature_extraction = joblib.load('tfidf_vectorizer.joblib')

# Set up Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def authenticate_gmail():
    # Perform authentication with OAuth
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def list_emails(service, max_results=10):
    # Fetch a list of emails
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])
    return messages

def get_email_content_and_subject(service, msg_id):
    # Retrieve email content and subject
    message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
    
    # Extract subject
    headers = message['payload']['headers']
    subject = next((header['value'] for header in headers if header['name'] == 'Subject'), "No Subject")
    
    # Extract the email body content
    for part in message['payload']['parts']:
        if part['mimeType'] == 'text/plain':
            data = part['body']['data']
            text = base64.urlsafe_b64decode(data).decode('utf-8')
            return subject, text
    return subject, ""  # Return subject even if the body is empty

def is_gibberish(text):
    # Detect text with high symbol-to-alphanumeric ratio or lack of common words
    if len(text) == 0:
        return False
    symbol_ratio = len(re.findall(r'\W', text)) / len(text)
    # Flag as spam if more than 30% symbols or less than 2 recognizable words
    if symbol_ratio > 0.3 or len(re.findall(r'\b\w+\b', text)) < 2:
        return True
    return False

def classify_email(content):
    # Predict spam or ham
    if is_gibberish(content):
        return "Spam"
    else:
        content_features = feature_extraction.transform([content])
        prediction = model.predict(content_features)
        return "Ham" if prediction[0] == 1 else "Spam"

def summarize_text(text):
    # Limit input to the model's max token length
    max_input_length = 512
    text = text[:max_input_length]
    
    # Set max_length dynamically based on input length
    input_length = len(text.split())  # Estimate token count by word count
    max_length = min(80, max(20, input_length // 2))  # Set max_length as half of input length, with bounds
    
    try:
        summary = summarizer(text, max_length=max_length, min_length=10, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return text  

def main():
    service = authenticate_gmail()
    emails = list_emails(service)

    for email in emails:
        msg_id = email['id']
        subject, content = get_email_content_and_subject(service, msg_id)
        summarized_content = summarize_text(content)
        if content:
            classification = classify_email(summarized_content)
            print(f"Subject: {subject}\nSummarized Content: {summarized_content}\nClassification: {classification}\n")

if __name__ == '__main__':
    main()
