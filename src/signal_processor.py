import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Tuple, Optional
from .utils import setup_logger

class SignalProcessor:
    
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.get('LOG_FILE'))
        
        # Initialize ML components
        self.vectorizer = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        self.scaler = StandardScaler()
        
        # Market-specific sentiment lexicon
        self.market_sentiment_words = {
            'bullish': ['bullish', 'rally', 'breakout', 'uptrend', 'buy', 'long', 'target', 'moon', 'rocket'],
            'bearish': ['bearish', 'crash', 'breakdown', 'downtrend', 'sell', 'short', 'dump', 'fall', 'drop'],
            'neutral': ['hold', 'sideways', 'range', 'consolidation', 'wait', 'watch']
        }
        
        # Signal processing cache
        self.signals_cache = {}
        self.processing_stats = {}
    
    def enhanced_sentiment_analysis(self, text_data: pd.Series) -> pd.DataFrame:
        
        sentiments = []
        
        self.logger.info(f"Processing sentiment for {len(text_data)} tweets")
        
        for idx, text in enumerate(text_data):
            try:
                # TextBlob analysis
                blob = TextBlob(str(text))
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Market-specific sentiment adjustment
                market_sentiment = self.calculate_market_sentiment(text)
                
                # Emoji sentiment
                emoji_sentiment = self.analyze_emoji_sentiment(text)
                
                # Compound sentiment with market weighting
                compound_sentiment = (
                    polarity * 0.4 +  # TextBlob base sentiment
                    market_sentiment * 0.4 +  # Market-specific keywords
                    emoji_sentiment * 0.2  # Emoji sentiment
                )
                
                # Confidence score based on subjectivity and market keywords
                confidence = self.calculate_sentiment_confidence(text, subjectivity)
                
                sentiments.append({
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'market_sentiment': market_sentiment,
                    'emoji_sentiment': emoji_sentiment,
                    'compound_sentiment': compound_sentiment,
                    'confidence': confidence
                })
                
                if idx % 500 == 0 and idx > 0:
                    self.logger.debug(f"Processed sentiment for {idx} tweets")
                    
            except Exception as e:
                self.logger.warning(f"Sentiment analysis failed for tweet {idx}: {e}")
                sentiments.append({
                    'polarity': 0, 'subjectivity': 0, 'market_sentiment': 0,
                    'emoji_sentiment': 0, 'compound_sentiment': 0, 'confidence': 0
                })
        
        return pd.DataFrame(sentiments)
    
    def calculate_market_sentiment(self, text: str) -> float:
        
        text_lower = text.lower()
        
        bullish_score = sum(1 for word in self.market_sentiment_words['bullish'] if word in text_lower)
        bearish_score = sum(1 for word in self.market_sentiment_words['bearish'] if word in text_lower)
        neutral_score = sum(1 for word in self.market_sentiment_words['neutral'] if word in text_lower)
        
        total_words = bullish_score + bearish_score + neutral_score
        
        if total_words == 0:
            return 0.0
        
        # Normalize to [-1, 1] range
        market_sentiment = (bullish_score - bearish_score) / total_words
        return max(-1.0, min(1.0, market_sentiment))
    
    def analyze_emoji_sentiment(self, text: str) -> float:
        
        positive_emojis = ['🚀', '📈', '💰', '🤑', '👍', '💪', '🔥', '⭐', '🎯']
        negative_emojis = ['📉', '💸', '😢', '😭', '👎', '⚠️', '🔻', '💔', '😰']
        
        positive_count = sum(text.count(emoji) for emoji in positive_emojis)
        negative_count = sum(text.count(emoji) for emoji in negative_emojis)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def calculate_sentiment_confidence(self, text: str, subjectivity: float) -> float:
        
        # Higher confidence for:
        # - More objective statements (lower subjectivity)
        # - Presence of market keywords
        # - Longer, more detailed content
        
        market_keywords_count = len([word for word_list in self.market_sentiment_words.values() 
                                   for word in word_list if word in text.lower()])
        
        length_factor = min(len(text) / 100, 1.0)  # Normalize to max 1.0
        objectivity_factor = 1.0 - subjectivity  # Lower subjectivity = higher confidence
        keyword_factor = min(market_keywords_count / 3, 1.0)  # Normalize to max 1.0
        
        confidence = (objectivity_factor * 0.4 + keyword_factor * 0.4 + length_factor * 0.2)
        return max(0.0, min(1.0, confidence))
    
    def volume_weighted_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Calculate comprehensive engagement score
        df['total_engagement'] = (
            df['likes'] * 1.0 +
            df['retweets'] * 2.0 +
            df['replies'] * 3.0 +
            df.get('quotes', 0) * 1.5
        )
        
        # User influence score (verified users, follower count)
        df['user_influence'] = np.log1p(df.get('user_followers', 0)) * (1 + df.get('user_verified', 0) * 0.5)
        
        # Time decay factor (more recent tweets have higher weight)
        current_time = datetime.now()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hours_old'] = (current_time - df['timestamp']).dt.total_seconds() / 3600
        df['time_decay'] = np.exp(-df['hours_old'] / 12)  # Decay over 12 hours
        
        # Combined volume weight
        df['volume_weight'] = (
            (df['total_engagement'] / df['total_engagement'].max()) * 0.4 +
            (df['user_influence'] / df['user_influence'].max()) * 0.3 +
            df['time_decay'] * 0.3
        )
        
        # Volume-weighted sentiment
        df['weighted_sentiment'] = df['compound_sentiment'] * df['volume_weight'] * df['confidence']
        
        return df
    
    def process_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Time-based aggregation (hourly signals)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('H')
        
        # Aggregate signals by hour
        hourly_signals = df.groupby('hour').agg({
            'weighted_sentiment': ['mean', 'std', 'count'],
            'compound_sentiment': 'mean',
            'confidence': 'mean',
            'total_engagement': 'sum',
            'volume_weight': 'mean',
            'urgency_score': 'mean'
        }).reset_index()
        
        # Flatten column names
        hourly_signals.columns = [
            'hour', 'avg_sentiment', 'sentiment_std', 'tweet_count',
            'raw_sentiment', 'avg_confidence', 'total_engagement',
            'avg_volume_weight', 'avg_urgency'
        ]
        
        # Calculate statistical confidence intervals
        hourly_signals['confidence_interval'] = (
            1.96 * (hourly_signals['sentiment_std'] / np.sqrt(hourly_signals['tweet_count']))
        ).fillna(0)
        
        # Generate trading signals with professional thresholds
        threshold = self.config.get('SENTIMENT_THRESHOLD', 0.1)
        
        conditions = [
            hourly_signals['avg_sentiment'] > (threshold + hourly_signals['confidence_interval']),
            hourly_signals['avg_sentiment'] < -(threshold + hourly_signals['confidence_interval'])
        ]
        choices = [1, -1]  # Buy, Sell
        hourly_signals['signal_strength'] = np.select(conditions, choices, default=0)  # Hold
        
        # Signal quality score (0-100)
        hourly_signals['signal_quality'] = (
            hourly_signals['tweet_count'] * 0.3 +  # Volume factor
            hourly_signals['avg_confidence'] * 50 +  # Confidence factor
            hourly_signals['total_engagement'] / 1000 * 0.2  # Engagement factor
        ).clip(0, 100)
        
        # Market regime classification
        hourly_signals['market_regime'] = np.select([
            hourly_signals['avg_sentiment'] > 0.2,
            hourly_signals['avg_sentiment'] < -0.2
        ], ['Bullish', 'Bearish'], default='Neutral')
        
        # Risk score (higher = more risky)
        hourly_signals['risk_score'] = (
            hourly_signals['sentiment_std'] * 50 +  # Volatility
            (1 - hourly_signals['avg_confidence']) * 30 +  # Uncertainty
            hourly_signals['avg_urgency'] * 20  # Urgency factor
        ).clip(0, 100)
        
        return hourly_signals
    
    def calculate_performance_metrics(self, signals_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics for signal quality"""
        if signals_df.empty:
            return {}
        
        metrics = {
            'signal_distribution': {
                'buy_signals': int((signals_df['signal_strength'] > 0).sum()),
                'sell_signals': int((signals_df['signal_strength'] < 0).sum()),
                'hold_signals': int((signals_df['signal_strength'] == 0).sum())
            },
            'quality_metrics': {
                'avg_signal_quality': float(signals_df['signal_quality'].mean()),
                'avg_confidence': float(signals_df['avg_confidence'].mean()),
                'avg_risk_score': float(signals_df['risk_score'].mean()),
                'high_quality_signals': int((signals_df['signal_quality'] > 70).sum())
            },
            'temporal_patterns': {
                'most_active_hour': int(signals_df.loc[signals_df['tweet_count'].idxmax(), 'hour'].hour),
                'peak_sentiment_hour': int(signals_df.loc[signals_df['avg_sentiment'].abs().idxmax(), 'hour'].hour),
                'total_tweets_processed': int(signals_df['tweet_count'].sum())
            },
            'market_sentiment': {
                'overall_sentiment': float(signals_df['avg_sentiment'].mean()),
                'sentiment_volatility': float(signals_df['avg_sentiment'].std()),
                'bullish_periods': int((signals_df['market_regime'] == 'Bullish').sum()),
                'bearish_periods': int((signals_df['market_regime'] == 'Bearish').sum())
            }
        }
        
        return metrics
    
    def save_signals_with_metadata(self, signals_df: pd.DataFrame, performance_metrics: Dict) -> bool:
        
        try:
            if signals_df.empty:
                self.logger.warning("No signals to save")
                return False
            
            # Optimize data types
            signals_df['tweet_count'] = signals_df['tweet_count'].astype('int32')
            signals_df['signal_strength'] = signals_df['signal_strength'].astype('int8')
            signals_df['signal_quality'] = signals_df['signal_quality'].astype('float32')
            signals_df['risk_score'] = signals_df['risk_score'].astype('float32')
            
            # Save signals
            signals_df.to_parquet(
                self.config['SIGNALS_FILE'],
                index=False,
                compression='snappy'
            )
            
            # Save metadata
            metadata = {
                'generation_info': {
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'algorithm': 'TF-IDF + Sentiment Analysis + Volume Weighting'
                },
                'signal_statistics': {
                    'total_signals': len(signals_df),
                    'time_range': {
                        'start': signals_df['hour'].min().isoformat(),
                        'end': signals_df['hour'].max().isoformat()
                    }
                },
                'performance_metrics': performance_metrics,
                'configuration': {
                    'sentiment_threshold': self.config.get('SENTIMENT_THRESHOLD', 0.1),
                    'confidence_level': self.config.get('CONFIDENCE_LEVEL', 0.95)
                }
            }
            
            metadata_file = self.config['DATA_DIR'] / 'signals' / 'signals_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Successfully saved {len(signals_df)} signals with metadata")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save signals: {e}")
            return False
    
    def generate_trading_signals(self) -> int:
        
        try:
            self.logger.info("=== Starting Professional Signal Generation ===")
            
            # Load tweet data
            if not self.config['TWEETS_FILE'].exists():
                self.logger.error("No tweet data found. Run data collection first.")
                return 0
            
            df = pd.read_parquet(self.config['TWEETS_FILE'])
            
            if df.empty:
                self.logger.warning("No tweets found in data file")
                return 0
            
            self.logger.info(f"Processing {len(df)} tweets for signal generation")
            
            # Step 1: Enhanced sentiment analysis
            sentiment_df = self.enhanced_sentiment_analysis(df['content'])
            df = pd.concat([df, sentiment_df], axis=1)
            
            # Step 2: Volume weighting
            df = self.volume_weighted_signals(df)
            
            # Step 3: Generate trading signals
            signals_df = self.process_trading_signals(df)
            
            # Step 4: Calculate performance metrics
            performance_metrics = self.calculate_performance_metrics(signals_df)
            
            # Step 5: Save signals with metadata
            success = self.save_signals_with_metadata(signals_df, performance_metrics)
            
            if success:
                self.logger.info(f"=== Signal Generation Complete ===")
                self.logger.info(f"Generated {len(signals_df)} trading signals")
                self.logger.info(f"Performance: {performance_metrics.get('quality_metrics', {})}")
                return len(signals_df)
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"Signal generation pipeline failed: {e}")
            raise

if __name__ == "__main__":
    # Test the signal processor
    from config import Config
    
    logging.basicConfig(level=logging.INFO)
    config_obj = Config()
    
    processor = SignalProcessor(config_obj.__dict__)
    signals_generated = processor.generate_trading_signals()
    print(f"Test completed: {signals_generated} signals generated")