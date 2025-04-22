// Sentiment analysis service
const SENTIMENT_API_URL = process.env.NEXT_PUBLIC_SENTIMENT_API_URL || 'http://localhost:5000';

export type SentimentResponse = {
  message: string;
  sentiment: 'negative' | 'neutral' | 'positive';
  confidence: number;
  probabilities: {
    negative: number;
    neutral: number;
    positive: number;
  };
  baseline_sentiment?: number;
  recommendation?: string;
};

/**
 * Analyzes the sentiment of a single message
 */
export const analyzeSentiment = async (message: string): Promise<SentimentResponse> => {
  try {
    const response = await fetch(`${SENTIMENT_API_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error analyzing sentiment:', error);
    throw error;
  }
};

/**
 * Analyzes a batch of messages, using all but the last as history
 * Returns sentiment analysis for the last message
 */
export const analyzeBatchSentiment = async (messages: string[]): Promise<SentimentResponse> => {
  try {
    const response = await fetch(`${SENTIMENT_API_URL}/analyze-batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ messages }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error analyzing batch sentiment:', error);
    throw error;
  }
}; 