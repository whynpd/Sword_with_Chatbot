# Sentiment Analysis Server for Discord Clone

This server provides sentiment analysis for messages in the Discord clone application. It includes two implementations:

1. **Simple Word-Based Model** (`app.py`): A lightweight rule-based sentiment analyzer that works without heavy dependencies
2. **Advanced RoBERTa Model** (`advanced_app.py`): Uses Hugging Face API to access state-of-the-art models

## Features

- Analyzes sentiment of individual messages
- Processes batches of messages to provide context-aware analysis
- Calculates baseline sentiment from message history
- Provides recommendations based on sentiment trends

## Setup

### Simple Word-Based Model

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the server:
   ```
   python app.py
   ```

### Advanced RoBERTa Model (Recommended)

1. Install the required packages:
   ```
   pip install -r requirements_advanced.txt
   ```

2. Get a Hugging Face API token:
   - Sign up at [huggingface.co](https://huggingface.co)
   - Go to Settings → Access Tokens
   - Create a new token with read permissions

3. Set your Hugging Face API token:
   ```
   # On Windows
   set HUGGINGFACE_API_TOKEN=your_token_here
   
   # On Linux/Mac
   export HUGGINGFACE_API_TOKEN=your_token_here
   ```

4. Run the advanced server:
   ```
   python advanced_app.py
   ```

The server will start on http://localhost:5000 by default.

## API Endpoints

Both implementations provide the same API endpoints:

### 1. Analyze a single message

**Endpoint:** `/analyze`

**Method:** POST

**Body:**
```json
{
  "message": "Your message text here"
}
```

**Response:**
```json
{
  "message": "Your message text here",
  "sentiment": "positive",
  "confidence": 0.92,
  "probabilities": {
    "negative": 0.05,
    "neutral": 0.03,
    "positive": 0.92
  },
  "recommendation": "The sentiment appears more positive than the baseline. This is a positive trend!"
}
```

### 2. Analyze a batch of messages

**Endpoint:** `/analyze-batch`

**Method:** POST

**Body:**
```json
{
  "messages": [
    "First message",
    "Second message",
    "Most recent message"
  ]
}
```

**Response:**
Same format as the single message analysis, but uses the earlier messages as context for the final message.

## Advanced Model Details

The advanced model implementation uses:

1. **RoBERTa** from cardiffnlp for state-of-the-art sentiment analysis
2. **API-based approach** to avoid heavy local dependencies
3. **Fallback mechanism** to the word-based model if the API is unavailable

## Integration with Discord Clone

To enable sentiment analysis in the Discord clone:

1. Make sure this server is running
2. Set the `NEXT_PUBLIC_SENTIMENT_API_URL` environment variable to point to this server
3. Use the sentiment analysis button in the message composer 