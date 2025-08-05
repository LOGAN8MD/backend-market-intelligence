
import re
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import unicodedata
import hashlib
from datetime import datetime

def setup_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def clean_tweet_content(text: str) -> str:
    
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Remove URLs (HTTP, HTTPS, WWW)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Step 2: Remove mentions (@username) but keep the content context
    text = re.sub(r'@\w+', '', text)
    
    # Step 3: Clean hashtags (remove # but keep the word for analysis)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Step 4: Handle Unicode normalization (important for Hindi content)
    text = unicodedata.normalize('NFKD', text)
    
    # Step 5: Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    
    # Step 6: Remove excessive punctuation but preserve important market symbols
    # Preserve: ₹, $, %, numbers, decimal points
    text = re.sub(r'[^\w\s₹$%.,!?()-]', ' ', text, flags=re.UNICODE)
    
    # Step 7: Remove repeated characters (spammy content)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)  # Max 2 repeated chars
    
    # Step 8: Clean extra spaces
    text = ' '.join(text.split())
    
    # Step 9: Remove very short words (likely noise) but preserve important ones
    words = text.split()
    important_short_words = {'₹', '$', '%', 'UP', 'GO', 'BUY', 'SELL', 'NSE', 'BSE'}
    cleaned_words = [word for word in words if len(word) > 2 or word.upper() in important_short_words]
    text = ' '.join(cleaned_words)
    
    return text.strip()

def extract_market_indicators(text: str) -> Dict[str, Any]:
    
    text_lower = text.lower()
    
    # Price targets and levels
    price_pattern = r'₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:₹|rs|rupees?)?'
    prices = re.findall(price_pattern, text)
    
    # Percentage movements
    percentage_pattern = r'(\d+(?:\.\d+)?)\s*%'
    percentages = re.findall(percentage_pattern, text)
    
    # Market actions
    buy_signals = len(re.findall(r'\b(buy|long|bullish|target|breakout)\b', text_lower))
    sell_signals = len(re.findall(r'\b(sell|short|bearish|dump|crash|fall)\b', text_lower))
    
    # Time indicators
    time_urgency = len(re.findall(r'\b(now|today|immediate|urgent|alert)\b', text_lower))
    
    return {
        'price_mentions': [float(p.replace(',', '')) for p in prices if p],
        'percentage_mentions': [float(p) for p in percentages if p],
        'buy_signal_strength': buy_signals,
        'sell_signal_strength': sell_signals,
        'urgency_level': time_urgency,
        'has_financial_data': bool(prices or percentages)
    }

def calculate_content_hash(content: str, algorithm: str = 'md5') -> str:
    
    if not content:
        return ""
    
    # Normalize content for consistent hashing
    normalized = content.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    
    if algorithm == 'md5':
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

def validate_tweet_data(tweet_data: Dict) -> bool:
    
    required_fields = ['content', 'username', 'timestamp']
    
    # Check required fields
    for field in required_fields:
        if field not in tweet_data or not tweet_data[field]:
            return False
    
    # Content quality checks
    content = tweet_data['content']
    
    # Minimum content length
    if len(content.strip()) < 10:
        return False
    
    # Maximum content length (spam detection)
    if len(content) > 500:
        return False
    
    # Check for excessive repetition (spam)
    words = content.split()
    if len(set(words)) < len(words) * 0.3:  # Less than 30% unique words
        return False
    
    # Check engagement metrics are non-negative
    engagement_fields = ['likes', 'retweets', 'replies']
    for field in engagement_fields:
        if field in tweet_data and tweet_data[field] < 0:
            return False
    
    return True

def format_number(number: float, format_type: str = 'compact') -> str:
    
    if format_type == 'percentage':
        return f"{number:.2%}"
    elif format_type == 'full':
        return f"{number:,.0f}"
    elif format_type == 'compact':
        if abs(number) >= 1e9:
            return f"{number/1e9:.1f}B"
        elif abs(number) >= 1e6:
            return f"{number/1e6:.1f}M"
        elif abs(number) >= 1e3:
            return f"{number/1e3:.1f}K"
        else:
            return f"{number:.0f}"
    else:
        return str(number)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    
    if denominator == 0:
        return default
    return numerator / denominator

def get_market_session(timestamp: datetime) -> str:
   
    hour = timestamp.hour
    
    if 9 <= hour < 15:
        return "market_hours"
    elif 15 <= hour < 18:
        return "post_market"
    elif 6 <= hour < 9:
        return "pre_market"
    else:
        return "after_hours"

def normalize_hashtags(hashtags: List[str]) -> List[str]:
    
    if not hashtags:
        return []
    
    normalized = []
    for tag in hashtags:
        if isinstance(tag, str):
            # Remove # symbol, convert to lowercase, remove special characters
            clean_tag = re.sub(r'[^a-zA-Z0-9]', '', tag.lower().replace('#', ''))
            if clean_tag and len(clean_tag) > 1:  # Ignore single character tags
                normalized.append(clean_tag)
    
    return list(set(normalized))  # Remove duplicates

def calculate_engagement_score(likes: int, retweets: int, replies: int, quotes: int = 0) -> float:
    
    # Professional weighting based on engagement value
    score = (
        likes * 1.0 +      # Base engagement
        retweets * 2.0 +   # Higher value (amplification)
        replies * 3.0 +    # Highest value (conversation)
        quotes * 1.5       # Medium value (sharing with comment)
    )
    
    return float(score)

def detect_language_mix(text: str) -> Dict[str, float]:
    
    if not text:
        return {'english': 0.0, 'hindi': 0.0, 'mixed': 0.0}
    
    # Simple heuristic for Indian market context
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))  # Devanagari range
    total_chars = len(text.replace(' ', ''))
    
    if total_chars == 0:
        return {'english': 0.0, 'hindi': 0.0, 'mixed': 0.0}
    
    english_ratio = english_chars / total_chars
    hindi_ratio = hindi_chars / total_chars
    
    return {
        'english': english_ratio,
        'hindi': hindi_ratio,
        'mixed': 1.0 if (english_ratio > 0.1 and hindi_ratio > 0.1) else 0.0
    }

# Performance monitoring utilities
class PerformanceMonitor:
    
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}
    
    def start(self):
        """Start timing"""
        self.start_time = datetime.now()
        return self
    
    def checkpoint(self, name: str):
        """Add checkpoint"""
        if self.start_time:
            self.checkpoints[name] = (datetime.now() - self.start_time).total_seconds()
    
    def get_duration(self) -> float:
        """Get total duration in seconds"""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0
    
    def get_report(self) -> Dict[str, float]:
        """Get performance report"""
        report = {
            'total_duration': self.get_duration(),
            'checkpoints': self.checkpoints.copy()
        }
        return report

# Export main functions
__all__ = [
    'setup_logger',
    'clean_tweet_content',
    'extract_market_indicators',
    'calculate_content_hash',
    'validate_tweet_data',
    'format_number',
    'safe_divide',
    'get_market_session',
    'normalize_hashtags',
    'calculate_engagement_score',
    'detect_language_mix',
    'PerformanceMonitor'
]