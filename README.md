# Market Intelligence System - Backend API

Professional Flask-based REST API for real-time market intelligence analysis from Indian stock market social media sentiment.

## 🚀 Overview

This backend system provides:
- **Twitter Data Collection**: Scrapes 2000+ tweets from Indian stock market discussions
- **Advanced Signal Processing**: Converts text to quantitative trading signals
- **Market Analysis**: Real-time sentiment analysis and trend detection
- **REST API**: Professional endpoints for frontend integration
- **Production-Ready**: Robust error handling, logging, and monitoring

## 📋 Requirements

- Python 3.8+
- 512MB RAM minimum (1GB recommended)
- Internet connection for data collection
- 100MB storage space

## ⚡ Quick Start

### 1. Clone and Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

2. Run Development Server

# Method 1: Direct execution
python app.py

# Method 2: Flask CLI
export FLASK_APP=app.py  # Linux/Mac
set FLASK_APP=app.py     # Windows
flask run

# Method 3: WSGI (production-like)
python wsgi.py

3. Verify Installation

# Test health endpoint
curl http://localhost:9999/api/health

# Or open in browser
http://localhost:9999/api/health

🏗️ Project Structure

backend/
├── app.py                  # Main Flask application
├── wsgi.py                # WSGI entry point for deployment
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── Procfile              # Render deployment config
├── src/                  # Core source code
│   ├── __init__.py
│   ├── data_collector.py  # Twitter scraping engine
│   ├── signal_processor.py # Signal generation
│   ├── data_processor.py  # Data cleaning and processing
│   ├── analyzer.py        # Market analysis
│   └── utils.py          # Utility functions
├── data/                 # Data storage
│   ├── tweets/          # Raw tweet data
│   ├── signals/         # Generated signals
│   └── analysis/        # Analysis results
└── logs/                # Application logs
    └── app.log

🔌 API Endpoints
Health & Status

    GET /api/health - System health check
    GET /api/status - Detailed system status
    GET /api/summary - Dashboard summary

Data & Analysis

    GET /api/signals - Latest trading signals
    GET /api/analysis - Market analysis results
    POST /api/collect - Trigger data collection

Example Responses

Health Check:

{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "version": "1.0.0",
  "data_files": {
    "tweets": true,
    "signals": true,
    "analysis": true
  }
}

Trading Signals:

{
  "signals": [
    {
      "hour": "2024-01-15T10:00:00",
      "avg_sentiment": 0.0234,
      "signal_strength": 1,
      "confidence_interval": 0.0456,
      "signal_quality": 0.78,
      "tweet_count": 45
    }
  ]
}

⚙️ Configuration
Environment Variables

Create .env file in backend directory:

# Flask Configuration
SECRET_KEY=your-secret-key-here
FLASK_ENV=development

# Data Collection Settings
MAX_TWEETS=2000
MAX_WORKERS=2
MIN_DELAY=1.0
MAX_DELAY=3.0
MAX_RETRIES=3

# Analysis Settings
SENTIMENT_THRESHOLD=0.1
CONFIDENCE_LEVEL=0.95

# Logging
LOG_LEVEL=INFO

Configuration Options

# config.py settings
HASHTAGS = ["#nifty50", "#sensex", "#intraday", "#banknifty", "#stockmarket"]
MAX_TWEETS = 2000
TWEETS_PER_HASHTAG = 400
MAX_WORKERS = 2
BATCH_SIZE = 500

🔄 Data Collection Process
Automated Collection

    Runs every 4 hours automatically
    Collects 2000+ tweets from last 24 hours
    Processes and generates trading signals
    Updates analysis and visualizations

Manual Collection

# Trigger via API
curl -X POST http://localhost:5000/api/collect

# Or through admin interface
python -c "from src.data_collector import MarketDataCollector; from config import Config; collector = MarketDataCollector(Config().__dict__); collector.collect_market_tweets()"

📊 Signal Generation
Sentiment Analysis

    TextBlob: Polarity and subjectivity scoring
    Market Keywords: Industry-specific sentiment
    Emoji Analysis: Social media context
    Confidence Scoring: Statistical reliability

Trading Signals

    Buy Signal (1): Positive sentiment above confidence threshold
    Sell Signal (-1): Negative sentiment below confidence threshold
    Hold Signal (0): Sentiment within confidence bounds

Quality Metrics

    Signal Strength: Magnitude of sentiment deviation
    Confidence Level: Statistical significance (95% default)
    Quality Score: Engagement and content quality weighting

🚀 Production Deployment
Using Gunicorn

# Install gunicorn
pip install gunicorn

# Run production server
gunicorn wsgi:app --bind 0.0.0.0:5000 --workers 1 --timeout 120