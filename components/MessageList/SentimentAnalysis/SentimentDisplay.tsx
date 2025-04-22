'use client';

import { SentimentResponse } from '@/services/sentimentService';
import { useState } from 'react';

interface SentimentDisplayProps {
  analysis: SentimentResponse;
  isLoading?: boolean;
}

const SentimentDisplay = ({ analysis, isLoading = false }: SentimentDisplayProps) => {
  const [showDetails, setShowDetails] = useState(false);

  if (isLoading) {
    return (
      <div className="mt-2 bg-gray-100 p-4 rounded-md">
        <p className="text-gray-600">Analyzing sentiment...</p>
      </div>
    );
  }

  if (!analysis) {
    return null;
  }

  // Color based on sentiment
  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return 'text-green-600';
      case 'negative':
        return 'text-red-600';
      default:
        return 'text-blue-600';
    }
  };

  // Progress bar width for probabilities
  const getProgressWidth = (value: number) => {
    return `${Math.round(value * 100)}%`;
  };

  return (
    <div className="mt-2 bg-gray-100 p-4 rounded-md">
      <div className="flex justify-between items-center">
        <h3 className="font-semibold text-gray-700">
          Sentiment Analysis:
          <span className={`ml-2 ${getSentimentColor(analysis.sentiment)}`}>
            {analysis.sentiment.charAt(0).toUpperCase() + analysis.sentiment.slice(1)}
          </span>
        </h3>
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-xs text-blue-500 hover:text-blue-700"
        >
          {showDetails ? 'Hide Details' : 'Show Details'}
        </button>
      </div>

      {analysis.recommendation && (
        <div className="mt-2 text-sm text-gray-700">
          <p><strong>Recommendation:</strong> {analysis.recommendation}</p>
        </div>
      )}

      {showDetails && (
        <div className="mt-3 border-t pt-3">
          <h4 className="text-sm font-semibold text-gray-600 mb-2">Sentiment Probabilities:</h4>
          <div className="space-y-2">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span>Positive</span>
                <span>{(analysis.probabilities.positive * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-green-500 h-2 rounded-full" 
                  style={{ width: getProgressWidth(analysis.probabilities.positive) }}
                ></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span>Neutral</span>
                <span>{(analysis.probabilities.neutral * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full" 
                  style={{ width: getProgressWidth(analysis.probabilities.neutral) }}
                ></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span>Negative</span>
                <span>{(analysis.probabilities.negative * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-red-500 h-2 rounded-full" 
                  style={{ width: getProgressWidth(analysis.probabilities.negative) }}
                ></div>
              </div>
            </div>
          </div>
          
          {analysis.baseline_sentiment !== undefined && (
            <div className="mt-3">
              <p className="text-xs text-gray-600">
                <strong>Baseline Sentiment:</strong> {analysis.baseline_sentiment.toFixed(2)}
                <span className="ml-2 text-gray-500">
                  ({analysis.baseline_sentiment > 0 
                    ? 'Positive trend' 
                    : analysis.baseline_sentiment < 0 
                      ? 'Negative trend' 
                      : 'Neutral'})
                </span>
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SentimentDisplay; 