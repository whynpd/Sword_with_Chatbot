# Hybrid Sentiment Analysis Model

This server provides sentiment analysis using a hybrid approach that combines a RoBERTa model (via Hugging Face API) and a fallback LSTM-like model.

## Setup

1. Install requirements:
```
pip install -r requirements_hybrid.txt
```

2. Set Hugging Face API token (optional):
```
export HF_API_TOKEN=your_hugging_face_token
```

3. Run the server:
```
python hybrid_model_app.py
```

## API Endpoints

- `POST /analyze`: Analyze a single message
  - Body: `{"message": "text to analyze"}`
  
- `POST /analyze_batch`: Analyze multiple messages
  - Body: `{"messages": ["message1", "message2", ...]}`
  
- `POST /simulate_stream`: Simulate processing a stream of messages
  - Body: `{"messages": ["message1", "message2", ...]}`

## Features

- Uses RoBERTa model via Hugging Face API for accurate sentiment analysis
- Falls back to LSTM-like model if API is unavailable
- Maintains context of recent messages for trend analysis
- Provides sentiment probabilities and confidence scores
- Detects abnormal sentiment shifts from baseline 