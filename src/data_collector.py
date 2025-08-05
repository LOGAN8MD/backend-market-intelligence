import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime, timedelta
import logging
import random
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import List, Dict, Optional
import json
from .utils import clean_tweet_content, setup_logger

class MarketDataCollector:
   
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.get('LOG_FILE'))
        self.collected_tweets = []
        self.error_count = 0
        self.success_count = 0
        self.collection_stats = {
            'start_time': None,
            'end_time': None,
            'total_tweets': 0,
            'unique_users': 0,
            'hashtags_processed': 0,
            'errors_encountered': 0
        }
    
    def extract_tweet_features(self, tweet) -> Dict:
        
        try:
            # Clean content using utility function
            cleaned_content = clean_tweet_content(tweet.content)
            
            # Skip if content too short after cleaning
            if len(cleaned_content) < 10:
                return None
            
            # Extract comprehensive features
            tweet_data = {
                "tweet_id": str(tweet.id),
                "username": tweet.user.username,
                "user_followers": getattr(tweet.user, 'followersCount', 0),
                "user_verified": getattr(tweet.user, 'verified', False),
                "timestamp": tweet.date,
                "content": cleaned_content,
                "original_content": tweet.content,
                "likes": tweet.likeCount or 0,
                "retweets": tweet.retweetCount or 0,
                "replies": tweet.replyCount or 0,
                "quotes": getattr(tweet, 'quoteCount', 0),
                "hashtags": [tag.lower() for tag in (tweet.hashtags or [])],
                "mentions": [user.username for user in (tweet.mentionedUsers or [])],
                "is_retweet": hasattr(tweet, 'retweetedTweet'),
                "is_reply": tweet.inReplyToTweetId is not None,
                "language": getattr(tweet, 'lang', 'unknown'),
                "content_hash": hashlib.md5(cleaned_content.encode()).hexdigest(),
                "collection_timestamp": datetime.now()
            }
            
            # Calculate engagement score with professional weighting
            tweet_data["engagement_score"] = (
                tweet_data["likes"] * 1.0 + 
                tweet_data["retweets"] * 2.0 + 
                tweet_data["replies"] * 3.0 +
                tweet_data["quotes"] * 1.5
            )
            
            # Add market-specific indicators
            tweet_data["market_keywords"] = self.extract_market_keywords(cleaned_content)
            tweet_data["urgency_score"] = self.calculate_urgency_score(cleaned_content)
            
            return tweet_data
            
        except Exception as e:
            self.logger.warning(f"Tweet feature extraction failed: {e}")
            return None
    
    def extract_market_keywords(self, content: str) -> List[str]:
        
        market_keywords = [
            'nifty', 'sensex', 'bse', 'nse', 'intraday', 'swing', 'positional',
            'bullish', 'bearish', 'breakout', 'support', 'resistance', 'target',
            'stoploss', 'profit booking', 'buy', 'sell', 'hold', 'accumulate',
            'momentum', 'volume', 'gap', 'rally', 'correction', 'trend'
        ]
        
        content_lower = content.lower()
        found_keywords = [keyword for keyword in market_keywords if keyword in content_lower]
        return found_keywords
    
    def calculate_urgency_score(self, content: str) -> float:
        """Calculate urgency score based on content indicators"""
        urgency_words = ['urgent', 'alert', 'breaking', 'immediate', 'now', 'quick', 'fast']
        content_lower = content.lower()
        
        score = 0.0
        for word in urgency_words:
            if word in content_lower:
                score += 0.2
        
        # Check for exclamation marks and caps
        if '!' in content:
            score += 0.1
        if content.isupper():
            score += 0.2
            
        return min(score, 1.0)  # Cap at 1.0
    
    def intelligent_delay(self, base_delay: float = None) -> None:
        
        if base_delay is None:
            base_delay = random.uniform(self.config['MIN_DELAY'], self.config['MAX_DELAY'])
        
        # Exponential backoff on errors
        if self.error_count > 0:
            backoff_multiplier = min(2 ** self.error_count, 16)  # Cap at 16x
            base_delay *= backoff_multiplier
            self.logger.debug(f"Applying exponential backoff: {backoff_multiplier}x")
        
        # Add random jitter to avoid pattern detection
        jitter = random.uniform(0.5, 1.5)
        final_delay = base_delay * jitter
        
        # Cap maximum delay for production efficiency
        final_delay = min(final_delay, 60)  # Max 60 seconds
        
        self.logger.debug(f"Intelligent delay: {final_delay:.2f} seconds")
        time.sleep(final_delay)
    
    def collect_hashtag_tweets(self, hashtag: str, max_tweets: int) -> List[Dict]:
        
        tweets = []
        collected_ids = set()
        retry_count = 0
        start_time = datetime.now()
        
        # Time range: last 24 hours for fresh market data
        today = datetime.utcnow()
        yesterday = today - timedelta(days=1)
        
        query = f"{hashtag} since:{yesterday.strftime('%Y-%m-%d')} until:{today.strftime('%Y-%m-%d')}"
        
        self.logger.info(f"Starting collection for {hashtag}: target {max_tweets} tweets")
        
        while retry_count < self.config['MAX_RETRIES']:
            try:
                scraper = sntwitter.TwitterSearchScraper(query)
                tweet_count = 0
                
                for tweet in scraper.get_items():
                    if tweet_count >= max_tweets:
                        break
                    
                    # Skip duplicates during collection
                    if tweet.id in collected_ids:
                        continue
                    
                    # Extract features
                    tweet_data = self.extract_tweet_features(tweet)
                    
                    if tweet_data:
                        tweets.append(tweet_data)
                        collected_ids.add(tweet.id)
                        tweet_count += 1
                    
                    # Progress logging and intelligent rate limiting
                    if tweet_count % 50 == 0 and tweet_count > 0:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        rate = tweet_count / elapsed if elapsed > 0 else 0
                        self.logger.info(f"{hashtag}: {tweet_count}/{max_tweets} tweets ({rate:.1f}/sec)")
                        self.intelligent_delay()
                
                # Success metrics
                duration = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"Successfully collected {len(tweets)} tweets for {hashtag} in {duration:.1f}s")
                self.success_count += 1
                self.error_count = max(0, self.error_count - 1)  # Reduce error count on success
                break
                
            except Exception as e:
                retry_count += 1
                self.error_count += 1
                
                self.logger.error(f"Error collecting {hashtag} tweets (attempt {retry_count}): {e}")
                
                if retry_count < self.config['MAX_RETRIES']:
                    delay = 30 * (2 ** retry_count)  # 30s, 60s, 120s
                    self.logger.info(f"Retrying {hashtag} in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Failed to collect tweets for {hashtag} after {self.config['MAX_RETRIES']} attempts")
        
        return tweets
    
    def advanced_deduplication(self, tweets: List[Dict]) -> pd.DataFrame:
        
        if not tweets:
            return pd.DataFrame()
        
        df = pd.DataFrame(tweets)
        initial_count = len(df)
        
        self.logger.info(f"Starting advanced deduplication for {initial_count} tweets")
        
        # Step 1: Remove exact duplicates by content hash
        df = df.drop_duplicates(subset=['content_hash'], keep='first')
        self.logger.info(f"After content deduplication: {len(df)} tweets")
        
        # Step 2: Remove retweets (we want original content)
        df = df[~df['is_retweet']]
        self.logger.info(f"After retweet removal: {len(df)} tweets")
        
        # Step 3: Remove near-duplicates from same user within time window
        df['timestamp_rounded'] = pd.to_datetime(df['timestamp']).dt.floor('5T')  # 5-minute buckets
        df = df.drop_duplicates(subset=['username', 'timestamp_rounded'], keep='first')
        self.logger.info(f"After temporal deduplication: {len(df)} tweets")
        
        # Step 4: Quality filtering
        # Remove very low engagement tweets (likely spam)
        df = df[df['engagement_score'] >= 1]
        
        # Remove tweets with insufficient content
        df = df[df['content'].str.len() >= 20]
        
        # Remove tweets with too many mentions (likely spam)
        df = df[df['mentions'].apply(len) <= 5]
        
        # Step 5: Keep tweets with market keywords (relevant content)
        df = df[df['market_keywords'].apply(len) > 0]
        
        final_count = len(df)
        removal_rate = ((initial_count - final_count) / initial_count * 100) if initial_count > 0 else 0
        
        self.logger.info(f"Advanced deduplication complete: {initial_count} -> {final_count} tweets ({removal_rate:.1f}% removed)")
        
        return df.drop(columns=['timestamp_rounded'])
    
    def parallel_collection(self) -> List[Dict]:
        
        all_tweets = []
        tweets_per_hashtag = self.config['TWEETS_PER_HASHTAG']
        
        self.logger.info(f"Starting parallel collection: {len(self.config['HASHTAGS'])} hashtags, {tweets_per_hashtag} tweets each")
        
        with ThreadPoolExecutor(max_workers=self.config['MAX_WORKERS']) as executor:
            # Submit collection tasks
            future_to_hashtag = {
                executor.submit(self.collect_hashtag_tweets, hashtag, tweets_per_hashtag): hashtag 
                for hashtag in self.config['HASHTAGS']
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_hashtag):
                hashtag = future_to_hashtag[future]
                try:
                    tweets = future.result(timeout=600)  # 10 minutes timeout
                    all_tweets.extend(tweets)
                    self.logger.info(f"Completed collection for {hashtag}: {len(tweets)} tweets")
                except Exception as e:
                    self.logger.error(f"Failed to collect tweets for {hashtag}: {e}")
                    self.collection_stats['errors_encountered'] += 1
        
        self.collection_stats['hashtags_processed'] = len(self.config['HASHTAGS'])
        return all_tweets
    
    def save_tweets_optimized(self, df: pd.DataFrame) -> bool:
        
        try:
            if df.empty:
                self.logger.warning("No tweets to save")
                return False
            
            # Optimize data types for storage efficiency
            df['likes'] = df['likes'].astype('int32')
            df['retweets'] = df['retweets'].astype('int32')
            df['replies'] = df['replies'].astype('int32')
            df['quotes'] = df['quotes'].astype('int32')
            df['user_followers'] = df['user_followers'].astype('int32')
            df['engagement_score'] = df['engagement_score'].astype('float32')
            df['urgency_score'] = df['urgency_score'].astype('float32')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['collection_timestamp'] = pd.to_datetime(df['collection_timestamp'])
            
            # Save main data file with compression
            df.to_parquet(
                self.config['TWEETS_FILE'], 
                index=False, 
                compression='snappy',
                engine='pyarrow'
            )
            
            # Create timestamped backup
            backup_file = self.config['DATA_DIR'] / 'tweets' / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            df.to_parquet(backup_file, index=False, compression='snappy')
            
            # Save comprehensive collection metadata
            metadata = {
                'collection_info': {
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'hashtags_collected': self.config['HASHTAGS'],
                    'collection_duration_seconds': self.collection_stats.get('duration', 0)
                },
                'data_statistics': {
                    'total_tweets': len(df),
                    'unique_users': int(df['username'].nunique()),
                    'verified_users': int(df['user_verified'].sum()),
                    'total_hashtags': int(df['hashtags'].apply(len).sum()),
                    'total_mentions': int(df['mentions'].apply(len).sum()),
                    'languages': df['language'].value_counts().to_dict()
                },
                'time_range': {
                    'earliest_tweet': df['timestamp'].min().isoformat(),
                    'latest_tweet': df['timestamp'].max().isoformat(),
                    'collection_start': self.collection_stats.get('start_time', '').isoformat() if self.collection_stats.get('start_time') else '',
                    'collection_end': self.collection_stats.get('end_time', '').isoformat() if self.collection_stats.get('end_time') else ''
                },
                'engagement_metrics': {
                    'total_likes': int(df['likes'].sum()),
                    'total_retweets': int(df['retweets'].sum()),
                    'total_replies': int(df['replies'].sum()),
                    'avg_engagement_score': float(df['engagement_score'].mean()),
                    'max_engagement_score': float(df['engagement_score'].max())
                },
                'quality_metrics': {
                    'avg_content_length': float(df['content'].str.len().mean()),
                    'tweets_with_market_keywords': int(df['market_keywords'].apply(len).gt(0).sum()),
                    'avg_urgency_score': float(df['urgency_score'].mean())
                },
                'collection_performance': {
                    'success_rate': f"{(self.success_count / len(self.config['HASHTAGS']) * 100):.1f}%",
                    'errors_encountered': self.collection_stats['errors_encountered'],
                    'hashtags_processed': self.collection_stats['hashtags_processed']
                }
            }
            
            # Save metadata as JSON
            metadata_file = self.config['DATA_DIR'] / 'tweets' / 'collection_metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Successfully saved {len(df)} tweets with metadata to {self.config['TWEETS_FILE']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save tweets: {e}")
            return False
    
    def collect_market_tweets(self) -> int:
        
        try:
            self.collection_stats['start_time'] = datetime.now()
            self.logger.info("=== Starting Professional Market Tweet Collection ===")
            
            # Reset counters
            self.error_count = 0
            self.success_count = 0
            
            # Parallel collection across hashtags
            raw_tweets = self.parallel_collection()
            
            if not raw_tweets:
                self.logger.warning("No tweets collected from any hashtag")
                return 0
            
            # Advanced deduplication and quality filtering
            df = self.advanced_deduplication(raw_tweets)
            
            # Save optimized data with metadata
            success = self.save_tweets_optimized(df)
            
            # Update collection statistics
            self.collection_stats.update({
                'end_time': datetime.now(),
                'total_tweets': len(df),
                'unique_users': df['username'].nunique() if not df.empty else 0,
                'duration': (datetime.now() - self.collection_stats['start_time']).total_seconds()
            })
            
            # Final summary
            duration = self.collection_stats['duration']
            summary = {
                'collection_duration_seconds': round(duration, 2),
                'tweets_collected': len(df),
                'unique_users': self.collection_stats['unique_users'],
                'hashtags_coverage': self.config['HASHTAGS'],
                'success_rate': f"{(self.success_count / len(self.config['HASHTAGS']) * 100):.1f}%",
                'avg_engagement': float(df['engagement_score'].mean()) if not df.empty else 0,
                'tweets_per_second': round(len(df) / duration, 2) if duration > 0 else 0
            }
            
            self.logger.info(f"=== Collection Complete ===")
            self.logger.info(f"Summary: {summary}")
            
            return len(df)
            
        except Exception as e:
            self.logger.error(f"Tweet collection pipeline failed: {e}")
            raise

if __name__ == "__main__":
    # Test the collector with sample configuration
    from config import Config
    
    logging.basicConfig(level=logging.INFO)
    config_obj = Config()
    
    collector = MarketDataCollector(config_obj.__dict__)
    tweets_collected = collector.collect_market_tweets()
    print(f"Test completed: {tweets_collected} tweets collected")