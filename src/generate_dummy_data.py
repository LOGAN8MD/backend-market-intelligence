import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

def generate_dummy_data(base_dir):
    TWEETS_FILE = base_dir / 'tweets' / 'market_tweets.parquet'
    SIGNALS_FILE = base_dir / 'signals' / 'trading_signals.parquet'
    ANALYSIS_FILE = base_dir / 'analysis' / 'market_analysis.parquet'

    # Ensure directories exist
    TWEETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SIGNALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # --- Dummy Tweets Data ---
    tweet_contents = [
        "Nifty looking bullish today!", "Sensex at record highs.", "Bank Nifty might dip tomorrow.",
        "FII buying supports the rally.", "Global markets positive sentiment.",
        "Watch out for earnings season.", "Technical breakout in Nifty50.",
        "Consolidation phase ongoing.", "Volume spike noticed in PSU banks.",
        "Market sentiment remains cautious."
    ]

    tweets_data = []
    start_time = datetime.now() - timedelta(hours=5)

    for i in range(1, 51):
        tweets_data.append({
            'id': i,
            'username': f'user{i%10 + 1}',
            'content': random.choice(tweet_contents),
            'timestamp': start_time + timedelta(minutes=i * 3),
            'collection_timestamp': datetime.now()
        })

    df_tweets = pd.DataFrame(tweets_data)
    df_tweets.to_parquet(TWEETS_FILE)

    # --- Dummy Signals Data ---
    signals_data = []
    signal_start_time = datetime.now() - timedelta(hours=23)

    for i in range(24):
        signals_data.append({
            'hour': (signal_start_time + timedelta(hours=i)).strftime('%Y-%m-%d %H:00'),
            'avg_sentiment': round(random.uniform(-0.3, 0.3), 2),
            'signal_strength': random.choice([-1, 0, 1]),
            'confidence_interval': round(random.uniform(0.85, 0.99), 2),
            'signal_quality': round(random.uniform(70, 95), 2),
            'tweet_count': random.randint(5, 20),
            'total_engagement': random.randint(50, 200)
        })

    df_signals = pd.DataFrame(signals_data)
    df_signals.to_parquet(SIGNALS_FILE)

    # --- Dummy Market Analysis Data ---
    analysis_indicators = [
        "Bullish Sentiment", "Bearish Sentiment", "Volume Surge", "Volatility Spike",
        "Moving Average Crossover", "RSI Overbought", "RSI Oversold", "MACD Divergence",
        "Support Zone Test", "Resistance Breakout"
    ]

    analysis_data = []
    for i in range(10):
        analysis_data.append({
            'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
            'indicator': analysis_indicators[i],
            'value': random.randint(30, 80),
            'notes': f"Observation on {analysis_indicators[i]}"
        })

    df_analysis = pd.DataFrame(analysis_data)
    df_analysis.to_parquet(ANALYSIS_FILE)

    print("Dummy data generated successfully!")