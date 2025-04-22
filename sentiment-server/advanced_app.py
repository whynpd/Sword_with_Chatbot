import numpy as np
import requests
import json
from collections import deque
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "")  # Add your Hugging Face token here
ROBERTA_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HISTORY_SIZE = 100  # Store the last 100 messages for baseline calculation

# Label mapping for sentiment
IDX2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.api_url = f"https://api-inference.huggingface.co/models/{ROBERTA_MODEL_ID}"
        self.headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        print("Advanced sentiment analyzer initialized")
    
    def query_huggingface(self, text):
        """Query the Hugging Face API for sentiment analysis"""
        payload = {"inputs": text}
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            return response.json()
        except Exception as e:
            print(f"Error querying Hugging Face API: {e}")
            return None

    def predict(self, text):
        """Predict sentiment using RoBERTa model via Hugging Face API"""
        # First try Hugging Face API
        api_result = self.query_huggingface(text)
        
        if api_result and not isinstance(api_result, dict):  # Check if result is a list as expected
            # If successful, parse results
            # Format is: [{'label': 'LABEL_X', 'score': 0.XX}, ...]
            try:
                # Sort by score in descending order
                sorted_results = sorted(api_result, key=lambda x: x['score'], reverse=True)
                
                # Get the label and confidence
                best_result = sorted_results[0]
                label = best_result['label']
                
                # Extract numerical label from "LABEL_X" format
                label_idx = int(label.split('_')[1])
                sentiment = IDX2LABEL[label_idx]
                confidence = best_result['score']
                
                # Calculate probabilities for all classes
                probabilities = [0.0, 0.0, 0.0]  # [negative, neutral, positive]
                for result in api_result:
                    idx = int(result['label'].split('_')[1])
                    probabilities[idx] = result['score']
                
                return sentiment, confidence, probabilities
            except (KeyError, IndexError, ValueError) as e:
                print(f"Error parsing API response: {e}, response: {api_result}")
                # Fall back to word-based approach
        
        # Fallback to word-based approach if API fails
        return self.fallback_predict(text)
    
    def fallback_predict(self, text):
        """Fallback to word-based sentiment analysis if API fails"""
        text = text.lower()
        words = text.split()
        
        # Define sentiment words
        positive_words = [
            "good", "great", "excellent", "amazing", "awesome", "nice", "wonderful", 
            "fantastic", "terrific", "perfect", "happy", "glad", "pleased", "love", 
            "beautiful", "brilliant", "outstanding", "superb", "delightful", "success", 
            "successful", "well", "positive", "win", "winning", "won", "improve", 
            "improvement", "improved", "better", "best", "solved", "fix", "fixed"
        ]
        
        negative_words = [
            "bad", "terrible", "horrible", "awful", "disappointing", "poor", "sad", 
            "unhappy", "angry", "upset", "annoyed", "frustrated", "disappointing", 
            "worst", "hate", "hates", "hated", "hating", "dislike", "failure", "fail", "failed", 
            "problem", "issue", "bug", "error", "crash", "slow", "difficult", "hard", 
            "confusing", "confused", "negative", "worse", "worst", "broken", "damage", 
            "damaged", "stupid", "idiot", "dumb", "useless", "garbage", "trash",
            "evil", "nasty", "disgusting", "terrible", "awful", "mean"
        ]
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in words)
        negative_count = sum(1 for word in negative_words if word in words)
                
        # If we explicitly find "hate", increase negative count significantly
        if "hate" in words:
            negative_count += 3
        
        # Calculate probabilities
        total = positive_count + negative_count
        if total == 0:  # If no sentiment words found
            return "neutral", 1.0, [0.0, 1.0, 0.0]
            
        positive_prob = positive_count / (total + 1)
        negative_prob = negative_count / (total + 1)
        neutral_prob = 1 - positive_prob - negative_prob
        
        # Determine sentiment
        if positive_prob > negative_prob and positive_prob > neutral_prob:
            sentiment = "positive"
            confidence = positive_prob
        elif negative_prob > positive_prob and negative_prob > neutral_prob:
            sentiment = "negative"
            confidence = negative_prob
        else:
            sentiment = "neutral"
            confidence = neutral_prob
        
        return sentiment, confidence, [negative_prob, neutral_prob, positive_prob]

# Initialize the model
print("Loading advanced sentiment analyzer...")
model = AdvancedSentimentAnalyzer()
print("Model loaded successfully!")

# Create a message history container (deque for efficient fixed-size queue)
message_history = deque(maxlen=HISTORY_SIZE)

# Function to get sentiment of a message
def get_message_sentiment(message):
    sentiment, confidence, probs = model.predict(message)
    return sentiment, confidence, probs

# Function to calculate baseline sentiment from historical messages
def get_baseline_sentiment(message_history):
    if not message_history:
        return 0.0  # Default neutral if no history
    
    # Map sentiment labels to numerical values
    sentiment_values = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    
    sentiments = []
    for message in message_history:
        sentiment_label, confidence, _ = get_message_sentiment(message)
        # Weight the sentiment by confidence
        sentiment_value = sentiment_values[sentiment_label] * confidence
        sentiments.append(sentiment_value)
    
    # Calculate the baseline sentiment (mean sentiment of past messages)
    baseline_sentiment = np.mean(sentiments)
    return baseline_sentiment

# Function to check if the current sentiment is abnormal
def check_if_abnormal(current_sentiment, baseline_sentiment, threshold=0.3):
    sentiment_values = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    current_value = sentiment_values[current_sentiment]
    
    # Calculate the difference between current and baseline sentiment
    sentiment_diff = current_value - baseline_sentiment
    
    if sentiment_diff < -threshold:
        return f"The sentiment appears more negative than the baseline. Consider investigating potential issues."
    elif sentiment_diff > threshold:
        return f"The sentiment appears more positive than the baseline. This is a positive trend!"
    else:
        return f"The sentiment is consistent with the baseline. No significant deviation detected."

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    new_message = data.get('message', '')
    
    if not new_message:
        return jsonify({'error': 'No message provided'}), 400
        
    # Get sentiment of the current message
    current_sentiment, confidence, probs = get_message_sentiment(new_message)
    
    result = {
        'message': new_message,
        'sentiment': current_sentiment,
        'confidence': float(confidence),
        'probabilities': {
            'negative': float(probs[0]),
            'neutral': float(probs[1]),
            'positive': float(probs[2])
        }
    }
    
    # If we have historical messages, analyze them too
    if len(message_history) > 0:
        # Get baseline sentiment from past messages
        baseline_sentiment = get_baseline_sentiment(message_history)
        
        # Check if current sentiment is abnormal compared to baseline
        recommendation = check_if_abnormal(current_sentiment, baseline_sentiment)
        
        result['baseline_sentiment'] = float(baseline_sentiment)
        result['recommendation'] = recommendation
    else:
        result['recommendation'] = "No historical messages available for baseline comparison."
    
    # Add the new message to history
    message_history.append(new_message)
    
    return jsonify(result)

@app.route('/analyze-batch', methods=['POST'])
def analyze_batch():
    data = request.json
    messages = data.get('messages', [])
    
    if not messages:
        return jsonify({'error': 'No messages provided'}), 400
    
    # Clear current history and add the batch
    message_history.clear()
    for msg in messages[:-1]:  # Add all except the last message to history
        message_history.append(msg)
    
    # Analyze the last message
    last_message = messages[-1]
    
    # Get sentiment of the current message
    current_sentiment, confidence, probs = get_message_sentiment(last_message)
    
    result = {
        'message': last_message,
        'sentiment': current_sentiment,
        'confidence': float(confidence),
        'probabilities': {
            'negative': float(probs[0]),
            'neutral': float(probs[1]),
            'positive': float(probs[2])
        }
    }
    
    # Get baseline sentiment from historical messages
    if len(message_history) > 0:
        baseline_sentiment = get_baseline_sentiment(message_history)
        recommendation = check_if_abnormal(current_sentiment, baseline_sentiment)
        
        result['baseline_sentiment'] = float(baseline_sentiment)
        result['recommendation'] = recommendation
    else:
        result['recommendation'] = "No historical messages available for baseline comparison."
    
    # Add the last message to history
    message_history.append(last_message)
    
    return jsonify(result)

if __name__ == '__main__':
    if not HUGGINGFACE_API_TOKEN:
        print("WARNING: HUGGINGFACE_API_TOKEN not set. Using fallback method for all requests.")
    app.run(host='0.0.0.0', port=5000, debug=True) 