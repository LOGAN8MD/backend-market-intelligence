import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Professional configuration class for market intelligence system"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'market-intelligence-professional-2024'
    
    # CORS settings for React frontend
    CORS_ORIGINS = ["http://localhost:3000", "https://your-frontend-domain.com"]
    
    # Data collection settings
    HASHTAGS = ["#nifty50", "#sensex", "#intraday", "#banknifty", "#stockmarket"]
    MAX_TWEETS = int(os.environ.get('MAX_TWEETS', 50))
    TWEETS_PER_HASHTAG = MAX_TWEETS // len(HASHTAGS)
    
    # Performance settings
    MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 2))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 500))
    MAX_MEMORY_MB = int(os.environ.get('MAX_MEMORY_MB', 512))
    
    # Rate limiting for professional scraping
    MIN_DELAY = float(os.environ.get('MIN_DELAY', 1.0))
    MAX_DELAY = float(os.environ.get('MAX_DELAY', 3.0))
    MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 3))
    
    # Analysis settings
    SENTIMENT_THRESHOLD = float(os.environ.get('SENTIMENT_THRESHOLD', 0.1))
    CONFIDENCE_LEVEL = float(os.environ.get('CONFIDENCE_LEVEL', 0.95))
    
    # Professional file structure
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    LOGS_DIR = BASE_DIR / 'logs'
    
    # Ensure professional directory structure
    for directory in [DATA_DIR, LOGS_DIR, DATA_DIR / 'tweets', DATA_DIR / 'signals', DATA_DIR / 'analysis']:
        directory.mkdir(exist_ok=True, parents=True)
    
    # Data file paths
    TWEETS_FILE = DATA_DIR / 'tweets' / 'market_tweets.parquet'
    SIGNALS_FILE = DATA_DIR / 'signals' / 'trading_signals.parquet'
    ANALYSIS_FILE = DATA_DIR / 'analysis' / 'market_analysis.parquet'
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = LOGS_DIR / 'app.log'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    ENV = 'development'

class ProductionConfig(Config):
    """Production configuration for professional deployment"""
    DEBUG = False
    ENV = 'production'
    MAX_WORKERS = 1  # Optimized for cloud deployment

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}