import numpy as np
import requests
import json
from collections import deque
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
from werkzeug.exceptions import BadRequest

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
API_KEY = os.environ.get("HUGGINGFACE_API_TOKEN", "")
ROBERTA_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HISTORY_SIZE = 100  # Store the last 100 messages for baseline calculation

# Label to index mapping for sentiment (based on cardiffnlp's model)
LABEL2IDX = {"negative": 0, "neutral": 1, "positive": 2}
IDX2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# Simple context memory to track recent sentiment
context_memory = deque(maxlen=5)  # Track the last 5 sentiment results

# Update context memory function
def update_context_memory(sentiment, confidence):
    """Update the context memory with a new sentiment analysis result"""
    context_memory.append({"sentiment": sentiment, "confidence": confidence})

# Function to analyze sentiment trend from context memory
def analyze_sentiment_trend():
    """Analyze trend from context memory"""
    if len(context_memory) < 2:
        return "Not enough context for trend analysis"
    
    # Convert sentiments to numerical values
    sentiment_values = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    
    # Calculate weighted sentiment values (more recent = higher weight)
    weighted_values = []
    for i, item in enumerate(context_memory):
        # Weight increases with recency
        weight = (i + 1) / sum(range(1, len(context_memory) + 1))
        sentiment_value = sentiment_values[item["sentiment"]] * item["confidence"] * weight
        weighted_values.append(sentiment_value)
    
    # Calculate trend as average change between consecutive sentiments
    trend_value = sum(weighted_values) / len(weighted_values)
    
    # Classify trend
    if trend_value > 0.2:
        return "improving"
    elif trend_value < -0.2:
        return "deteriorating"
    else:
        return "stable"

class LSTMModel:
    """Simple LSTM-based sentiment classifier using memory of past messages"""
    def __init__(self):
        # Initialize sentiment word lists
        self.positive_words = [
            "good", "great", "excellent", "amazing", "awesome", "nice", "wonderful", 
            "fantastic", "terrific", "perfect", "happy", "glad", "pleased", "love", 
            "beautiful", "brilliant", "outstanding", "superb", "delightful", "success", 
            "successful", "well", "positive", "win", "winning", "won", "improve", 
            "improvement", "improved", "better", "best", "solved", "fix", "fixed"
        ]
        
        self.negative_words = [
            "bad", "terrible", "horrible", "awful", "disappointing", "poor", "sad", 
            "unhappy", "angry", "upset", "annoyed", "frustrated", "disappointing", 
            "worst", "hate", "hates", "hated", "hating", "dislike", "failure", "fail", "failed", 
            "problem", "issue", "bug", "error", "crash", "slow", "difficult", "hard", 
            "confusing", "confused", "negative", "worse", "worst", "broken", "damage", 
            "damaged", "stupid", "idiot", "dumb", "useless", "garbage", "trash",
            "evil", "nasty", "disgusting", "terrible", "awful", "mean"
        ]
        
        # Initialize memory for LSTM-like context
        self.context_memory = []
        self.memory_size = 5  # Remember last 5 messages for context
        
        print("LSTM model initialized")
    
    def predict(self, message):
        """Predict sentiment with context awareness like an LSTM"""
        # Add message to context memory
        self.context_memory.append(message.lower())
        # Keep only recent messages
        if len(self.context_memory) > self.memory_size:
            self.context_memory = self.context_memory[-self.memory_size:]
        
        # Analyze current message
        current_score = self._analyze_text(message.lower())
        
        # If we have context, use it to refine prediction
        if len(self.context_memory) > 1:
            # Calculate sentiment trend (simple LSTM-like recurrent behavior)
            context_scores = [self._analyze_text(msg) for msg in self.context_memory[:-1]]
            avg_context = sum(context_scores) / len(context_scores)
            
            # Weight recent messages more heavily (recency bias in LSTM)
            weighted_scores = []
            for i, score in enumerate(context_scores):
                # More recent messages get higher weights
                weight = (i + 1) / len(context_scores)
                weighted_scores.append(score * weight)
            
            weighted_context = sum(weighted_scores) / sum(range(1, len(context_scores) + 1))
            
            # Blend context with current text (like LSTM memory gates)
            # Higher weight to current message (0.7) vs context (0.3)
            blended_score = 0.7 * current_score + 0.3 * weighted_context
            
            # Determine sentiment from blended score
            sentiment, confidence, probs = self._score_to_sentiment(blended_score)
        else:
            sentiment, confidence, probs = self._score_to_sentiment(current_score)
        
        return sentiment, confidence, probs
    
    def _analyze_text(self, text):
        """Analyze text and return a sentiment score from -1 to 1"""
        words = text.split()
        
        # Count positive and negative words
        positive_count = sum(1 for word in self.positive_words if word in words)
        negative_count = sum(1 for word in self.negative_words if word in words)
        
        # Apply additional weight for strong sentiment words
        if "hate" in words or "terrible" in words:
            negative_count += 2
        if "love" in words or "amazing" in words:
            positive_count += 2
            
        # Calculate sentiment score from -1 to 1
        total = positive_count + negative_count
        if total == 0:
            return 0  # Neutral
            
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _score_to_sentiment(self, score):
        """Convert a score to sentiment label, confidence, and probabilities"""
        # Map score to probabilities
        if score > 0.3:
            sentiment = "positive"
            confidence = min(0.5 + abs(score) / 2, 0.99)
            probs = [0.1, 0.2, 0.7]  # Mapping to [neg, neut, pos]
        elif score < -0.3:
            sentiment = "negative"
            confidence = min(0.5 + abs(score) / 2, 0.99)
            probs = [0.7, 0.2, 0.1]  # Mapping to [neg, neut, pos]
        else:
            sentiment = "neutral"
            confidence = max(0.5, 1 - abs(score) * 2)
            probs = [0.1, 0.8, 0.1]  # Mapping to [neg, neut, pos]
            
        # Adjust probabilities based on confidence
        if sentiment == "positive":
            probs = [0.1 * (1-confidence), 0.3 * (1-confidence), confidence]
        elif sentiment == "negative":
            probs = [confidence, 0.3 * (1-confidence), 0.1 * (1-confidence)]
        else:
            probs = [(1-confidence)/2, confidence, (1-confidence)/2]
            
        return sentiment, confidence, probs

class RobertaModel:
    """RoBERTa model through API"""
    def __init__(self, api_key="", model_id=ROBERTA_MODEL_ID):
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        print("RoBERTa model initialized")
    
    def predict(self, text):
        """Predict sentiment using RoBERTa model via API"""
        # Try to use API
        api_result = self._query_api(text)
        
        if api_result and not isinstance(api_result, dict):
            try:
                # Parse API response
                sorted_results = sorted(api_result, key=lambda x: x['score'], reverse=True)
                
                best_result = sorted_results[0]
                label = best_result['label']
                
                # Extract label index
                label_idx = int(label.split('_')[1])
                sentiment = IDX2LABEL[label_idx]
                confidence = best_result['score']
                
                # Get probabilities for all classes
                probabilities = [0.0, 0.0, 0.0]  # [negative, neutral, positive]
                for result in api_result:
                    idx = int(result['label'].split('_')[1])
                    probabilities[idx] = result['score']
                
                return sentiment, confidence, probabilities
            except (KeyError, IndexError, ValueError) as e:
                print(f"Error parsing API response: {e}")
                # Fall back to LSTM model
                return None
        return None
    
    def _query_api(self, text):
        """Send request to Hugging Face API"""
        payload = {"inputs": text}
        try:
            if not self.headers:
                # No API key provided, can't use API
                return None
                
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # The model is loading, wait and retry once
                time.sleep(2)
                response = requests.post(self.api_url, headers=self.headers, json=payload)
                if response.status_code == 200:
                    return response.json()
            return None
        except Exception as e:
            print(f"API request error: {e}")
            return None

class HybridSentimentAnalyzer:
    """Hybrid sentiment analyzer using both RoBERTa and LSTM"""
    def __init__(self, api_key=""):
        self.roberta = RobertaModel(api_key)
        self.lstm = LSTMModel()
        print("Hybrid sentiment analyzer initialized")
    
    def predict(self, text):
        """Predict sentiment using both models with preference for RoBERTa"""
        # First try RoBERTa
        roberta_result = self.roberta.predict(text)
        if roberta_result:
            # Successfully got RoBERTa prediction
            return roberta_result
        
        # Fall back to LSTM if RoBERTa fails
        return self.lstm.predict(text)

# Initialize the model
print("Loading hybrid sentiment analyzer...")
model = HybridSentimentAnalyzer(API_KEY)
print("Model loaded successfully!")

# Create a message history container (deque for efficient fixed-size queue)
message_history = deque(maxlen=HISTORY_SIZE)

# Function to get sentiment of a message
def get_message_sentiment(message):
    """Get sentiment of a message using the hybrid model"""
    sentiment, confidence, probs = model.predict(message)
    return sentiment, confidence, probs

# Function to calculate baseline sentiment from historical messages
def get_baseline_sentiment(message_history):
    """Calculate baseline sentiment from message history"""
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
    """Check if the current sentiment is abnormal compared to baseline"""
    sentiment_values = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    current_value = sentiment_values[current_sentiment]
    
    # Calculate the difference between current and baseline sentiment
    sentiment_diff = current_value - baseline_sentiment
    
    if sentiment_diff < -threshold:
        return f"Recommendation: The sentiment appears more negative than the baseline. Consider investigating potential issues."
    elif sentiment_diff > threshold:
        return f"Recommendation: The sentiment appears more positive than the baseline. This is a positive trend!"
    else:
        return f"Recommendation: The sentiment is consistent with the baseline. No significant deviation detected."

# Debug function to analyze a new message similar to the original code
def analyze_sentiment(new_message):
    """Analyze sentiment of a new message (debug/demo function)"""
    print("\n--- Sentiment Analysis ---")
    print(f"Analyzing message: \"{new_message}\"")
    
    # Get sentiment of the current message
    current_sentiment, confidence, probs = get_message_sentiment(new_message)
    
    # Print detailed sentiment analysis
    print(f"\nCurrent sentiment: {current_sentiment} (confidence: {confidence:.4f})")
    print(f"Sentiment probabilities: Negative: {probs[0]:.4f}, Neutral: {probs[1]:.4f}, Positive: {probs[2]:.4f}")
    
    # If we have historical messages, analyze them too
    if len(message_history) > 0:
        print(f"\nAnalyzing baseline from {len(message_history)} historical messages...")
        
        # Get baseline sentiment from past messages
        baseline_sentiment = get_baseline_sentiment(message_history)
        print(f"Baseline sentiment score: {baseline_sentiment:.4f}")
        
        # Check if current sentiment is abnormal compared to baseline
        recommendation = check_if_abnormal(current_sentiment, baseline_sentiment)
        print(f"\n{recommendation}")
    else:
        print("\nNo historical messages available for baseline comparison.")
    
    # Add the new message to history
    message_history.append(new_message)
    
    return current_sentiment, confidence

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to analyze a single message"""
    data = request.json
    
    # Check for required field with better error handling
    if 'message' not in data or not data.get('message'):
        raise BadRequest("Missing required 'message' field")
        
    new_message = data.get('message')
    
    # Get sentiment of the current message
    current_sentiment, confidence, probs = get_message_sentiment(new_message)
    
    # Update context memory with the new result
    update_context_memory(current_sentiment, confidence)
    
    # Get sentiment trend if we have enough context
    sentiment_trend = analyze_sentiment_trend() if len(context_memory) >= 2 else None
    
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
    
    # Add trend information if available
    if sentiment_trend:
        result['trend'] = sentiment_trend
    
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
    """API endpoint to analyze a batch of messages"""
    data = request.json
    
    # Check for required field with better error handling
    if 'messages' not in data or not data.get('messages'):
        raise BadRequest("Missing required 'messages' field")
    
    messages = data.get('messages', [])
    
    # Clear current history and add the batch
    message_history.clear()
    context_memory.clear()  # Also clear context memory
    
    # Process all messages but the last for history
    for msg in messages[:-1]:
        message_history.append(msg)
        # Also update context memory for trend analysis
        sentiment, confidence, _ = get_message_sentiment(msg)
        update_context_memory(sentiment, confidence)
    
    # Analyze the last message
    last_message = messages[-1]
    
    # Get sentiment of the current message
    current_sentiment, confidence, probs = get_message_sentiment(last_message)
    
    # Update context memory with the last message
    update_context_memory(current_sentiment, confidence)
    
    # Get sentiment trend
    sentiment_trend = analyze_sentiment_trend() if len(context_memory) >= 2 else None
    
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
    
    # Add trend information if available
    if sentiment_trend:
        result['trend'] = sentiment_trend
    
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

@app.route('/simulate', methods=['POST'])
def simulate_analysis():
    """API endpoint to simulate message stream analysis (for demo/debugging)"""
    data = request.json
    
    # Check for required field with better error handling
    if 'messages' not in data or not data.get('messages'):
        raise BadRequest("Missing required 'messages' field")
    
    messages = data.get('messages', [])
    
    # Clear message history and context memory
    message_history.clear()
    context_memory.clear()
    
    results = []
    for msg in messages:
        sentiment, confidence = analyze_sentiment(msg)
        # Update context memory
        update_context_memory(sentiment, confidence)
        
        # Include trend information if available
        result_item = {
            'message': msg,
            'sentiment': sentiment,
            'confidence': float(confidence)
        }
        
        if len(context_memory) >= 2:
            result_item['trend'] = analyze_sentiment_trend()
        
        results.append(result_item)
    
    # Calculate summary statistics
    sentiment_counts = {
        "negative": sum(1 for r in results if r['sentiment'] == "negative"),
        "neutral": sum(1 for r in results if r['sentiment'] == "neutral"),
        "positive": sum(1 for r in results if r['sentiment'] == "positive")
    }
    
    return jsonify({
        'results': results,
        'summary': {
            'total': len(results),
            'sentiment_counts': sentiment_counts,
            'percentages': {
                'positive': sentiment_counts['positive']/len(results)*100,
                'neutral': sentiment_counts['neutral']/len(results)*100,
                'negative': sentiment_counts['negative']/len(results)*100
            }
        }
    })

if __name__ == '__main__':
    if not API_KEY:
        print("WARNING: HUGGINGFACE_API_TOKEN not set. Using LSTM model for all requests.")
    app.run(host='0.0.0.0', port=5000, debug=True) 