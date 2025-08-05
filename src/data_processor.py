import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .utils import setup_logger, clean_tweet_content

class DataProcessor:
    
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.get('LOG_FILE'))
        
        # Initialize scalers
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # Data quality metrics
        self.quality_metrics = {}
        self.processing_stats = {}
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        
        quality_report = {
            'total_records': len(df),
            'missing_data': {},
            'duplicates': {},
            'outliers': {},
            'data_types': {},
            'content_quality': {},
            'timestamp_analysis': {}
        }
        
        if df.empty:
            return quality_report
        
        # Missing data analysis
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            quality_report['missing_data'][column] = {
                'count': int(missing_count),
                'percentage': float(missing_count / len(df) * 100)
            }
        
        # Duplicate analysis
        content_duplicates = df.duplicated(subset=['content']).sum()
        user_duplicates = df.duplicated(subset=['username', 'timestamp']).sum()
        
        quality_report['duplicates'] = {
            'content_duplicates': int(content_duplicates),
            'user_time_duplicates': int(user_duplicates),
            'exact_duplicates': int(df.duplicated().sum())
        }
        
        # Outlier detection for numerical columns
        numerical_columns = ['likes', 'retweets', 'replies', 'engagement_score']
        for col in numerical_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                quality_report['outliers'][col] = int(outliers)
        
        # Content quality analysis
        if 'content' in df.columns:
            quality_report['content_quality'] = {
                'avg_length': float(df['content'].str.len().mean()),
                'empty_content': int(df['content'].str.len().eq(0).sum()),
                'short_content': int(df['content'].str.len().lt(20).sum()),
                'very_long_content': int(df['content'].str.len().gt(280).sum())
            }
        
        # Timestamp analysis
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            quality_report['timestamp_analysis'] = {
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                },
                'future_dates': int((df['timestamp'] > datetime.now()).sum()),
                'very_old_dates': int((df['timestamp'] < datetime.now() - timedelta(days=30)).sum())
            }
        
        self.quality_metrics = quality_report
        return quality_report
    
    def clean_and_normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if df.empty:
            return df
        
        self.logger.info(f"Starting data cleaning for {len(df)} records")
        original_count = len(df)
        
        # Step 1: Content cleaning
        if 'content' in df.columns:
            df['content'] = df['content'].apply(lambda x: clean_tweet_content(str(x)) if pd.notna(x) else '')
            
            # Remove records with empty content after cleaning
            df = df[df['content'].str.len() > 0]
            self.logger.info(f"Removed {original_count - len(df)} records with empty content")
        
        # Step 2: Handle missing data
        df = self.handle_missing_data(df)
        
        # Step 3: Normalize numerical columns
        numerical_columns = ['likes', 'retweets', 'replies', 'user_followers']
        for col in numerical_columns:
            if col in df.columns:
                # Replace negative values with 0
                df[col] = df[col].clip(lower=0)
                
                # Handle extreme outliers
                df[col] = self.handle_outliers(df[col])
        
        # Step 4: Standardize categorical data
        if 'username' in df.columns:
            df['username'] = df['username'].str.lower().str.strip()
        
        if 'hashtags' in df.columns:
            df['hashtags'] = df['hashtags'].apply(self.normalize_hashtags)
        
        if 'mentions' in df.columns:
            df['mentions'] = df['mentions'].apply(self.normalize_mentions)
        
        # Step 5: Data type optimization
        df = self.optimize_data_types(df)
        
        self.logger.info(f"Data cleaning completed: {len(df)} records remaining")
        return df
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        
        
        # Numerical columns: fill with 0 or median
        numerical_fill_zero = ['likes', 'retweets', 'replies', 'quotes']
        for col in numerical_fill_zero:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # User followers: fill with median of non-zero values
        if 'user_followers' in df.columns:
            non_zero_median = df[df['user_followers'] > 0]['user_followers'].median()
            df['user_followers'] = df['user_followers'].fillna(non_zero_median)
        
        # Boolean columns: fill with False
        boolean_columns = ['user_verified', 'is_retweet', 'is_reply']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].fillna(False)
        
        # List columns: fill with empty list
        list_columns = ['hashtags', 'mentions', 'market_keywords']
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        
        # String columns: fill with empty string
        string_columns = ['language']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
        
        return df
    
    def handle_outliers(self, series: pd.Series, method='clip') -> pd.Series:
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if method == 'clip':
            return series.clip(lower=max(0, lower_bound), upper=upper_bound)
        elif method == 'remove':
            return series[(series >= lower_bound) & (series <= upper_bound)]
        else:
            return series
    
    def normalize_hashtags(self, hashtags) -> List[str]:
        
        if not isinstance(hashtags, list):
            return []
        
        normalized = []
        for tag in hashtags:
            if isinstance(tag, str):
                # Remove # symbol, convert to lowercase, remove special characters
                clean_tag = re.sub(r'[^a-zA-Z0-9]', '', tag.lower().replace('#', ''))
                if clean_tag:
                    normalized.append(clean_tag)
        
        return list(set(normalized))  # Remove duplicates
    
    def normalize_mentions(self, mentions) -> List[str]:
        
        if not isinstance(mentions, list):
            return []
        
        normalized = []
        for mention in mentions:
            if isinstance(mention, str):
                # Remove @ symbol, convert to lowercase
                clean_mention = mention.lower().replace('@', '').strip()
                if clean_mention:
                    normalized.append(clean_mention)
        
        return list(set(normalized))  # Remove duplicates
    
    def optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        
        # Integer columns
        int_columns = ['likes', 'retweets', 'replies', 'quotes', 'user_followers']
        for col in int_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int32')
        
        # Float columns
        float_columns = ['engagement_score', 'urgency_score']
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float32')
        
        # Boolean columns
        boolean_columns = ['user_verified', 'is_retweet', 'is_reply']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype('bool')
        
        # Datetime columns
        datetime_columns = ['timestamp', 'collection_timestamp']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Category columns for memory efficiency
        category_columns = ['language', 'username']
        for col in category_columns:
            if col in df.columns and df[col].nunique() < len(df) * 0.5:
                df[col] = df[col].astype('category')
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if df.empty:
            return df
        
        self.logger.info("Starting feature engineering")
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
            df['is_market_hours'] = df['hour'].between(9, 15)  # Indian market hours
        
        # Engagement ratios
        if all(col in df.columns for col in ['likes', 'retweets', 'replies']):
            df['total_interactions'] = df['likes'] + df['retweets'] + df['replies']
            df['retweet_like_ratio'] = np.where(df['likes'] > 0, df['retweets'] / df['likes'], 0)
            df['reply_engagement_ratio'] = np.where(df['total_interactions'] > 0, 
                                                   df['replies'] / df['total_interactions'], 0)
        
        # User influence metrics
        if 'user_followers' in df.columns:
            df['follower_log'] = np.log1p(df['user_followers'])
            df['influence_score'] = (
                df['follower_log'] * 0.6 +
                df.get('user_verified', 0) * 10 +
                df.get('total_interactions', 0) / 100
            )
        
        # Content quality indicators
        if 'content' in df.columns:
            df['content_length'] = df['content'].str.len()
            df['word_count'] = df['content'].str.split().str.len()
            df['avg_word_length'] = df['content'].apply(lambda x: np.mean([len(word) for word in str(x).split()]) if x else 0)
            df['has_numbers'] = df['content'].str.contains(r'\d+', regex=True)
            df['has_symbols'] = df['content'].str.contains(r'[$₹%]', regex=True)
        
        # Market activity indicators
        if 'hashtags' in df.columns:
            df['hashtag_count'] = df['hashtags'].apply(len)
            df['has_market_hashtags'] = df['hashtags'].apply(
                lambda tags: any(tag in ['nifty50', 'sensex', 'banknifty', 'intraday'] for tag in tags)
            )
        
        if 'mentions' in df.columns:
            df['mention_count'] = df['mentions'].apply(len)
        
        # Normalize engineered features
        feature_columns = ['influence_score', 'content_length', 'word_count']
        for col in feature_columns:
            if col in df.columns:
                df[f'{col}_normalized'] = self.minmax_scaler.fit_transform(df[[col]])
        
        self.logger.info(f"Feature engineering completed: {df.shape[1]} total features")
        return df
    
    def create_data_summary(self, df: pd.DataFrame) -> Dict:
        
        if df.empty:
            return {}
        
        summary = {
            'data_overview': {
                'total_records': len(df),
                'total_features': df.shape[1],
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'processing_timestamp': datetime.now().isoformat()
            },
            'temporal_distribution': {},
            'user_statistics': {},
            'content_statistics': {},
            'engagement_statistics': {}
        }
        
        # Temporal distribution
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            summary['temporal_distribution'] = {
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                },
                'hourly_distribution': df.groupby(df['timestamp'].dt.hour).size().to_dict(),
                'daily_distribution': df.groupby(df['timestamp'].dt.date).size().to_dict()
            }
        
        # User statistics
        if 'username' in df.columns:
            summary['user_statistics'] = {
                'unique_users': int(df['username'].nunique()),
                'avg_tweets_per_user': float(len(df) / df['username'].nunique()),
                'most_active_users': df['username'].value_counts().head(10).to_dict()
            }
        
        # Content statistics
        if 'content' in df.columns:
            summary['content_statistics'] = {
                'avg_content_length': float(df['content'].str.len().mean()),
                'content_length_distribution': {
                    'min': int(df['content'].str.len().min()),
                    'max': int(df['content'].str.len().max()),
                    'std': float(df['content'].str.len().std())
                }
            }
        
        # Engagement statistics
        engagement_cols = ['likes', 'retweets', 'replies']
        if all(col in df.columns for col in engagement_cols):
            summary['engagement_statistics'] = {
                'total_likes': int(df['likes'].sum()),
                'total_retweets': int(df['retweets'].sum()),
                'total_replies': int(df['replies'].sum()),
                'avg_engagement_per_tweet': float(df[engagement_cols].sum(axis=1).mean())
            }
        
        return summary
    
    def process_data_pipeline(self, input_file: str = None) -> bool:
        
        try:
            self.logger.info("=== Starting Professional Data Processing ===")
            
            # Load data
            if input_file:
                data_path = input_file
            else:
                data_path = self.config['TWEETS_FILE']
            
            if not data_path.exists():
                self.logger.error("Input data file not found")
                return False
            
            df = pd.read_parquet(data_path)
            
            if df.empty:
                self.logger.warning("No data found in input file")
                return False
            
            self.logger.info(f"Loaded {len(df)} records for processing")
            
            # Step 1: Data quality validation
            quality_report = self.validate_data_quality(df)
            self.logger.info(f"Data quality assessment completed")
            
            # Step 2: Clean and normalize data
            df_cleaned = self.clean_and_normalize_data(df)
            
            # Step 3: Feature engineering
            df_enhanced = self.engineer_features(df_cleaned)
            
            # Step 4: Create data summary
            data_summary = self.create_data_summary(df_enhanced)
            
            # Step 5: Save processed data
            processed_file = self.config['DATA_DIR'] / 'tweets' / 'processed_tweets.parquet'
            df_enhanced.to_parquet(processed_file, index=False, compression='snappy')
            
            # Save processing metadata
            metadata = {
                'processing_info': {
                    'timestamp': datetime.now().isoformat(),
                    'input_records': len(df),
                    'output_records': len(df_enhanced),
                    'processing_version': '1.0.0'
                },
                'quality_report': quality_report,
                'data_summary': data_summary
            }
            
            metadata_file = self.config['DATA_DIR'] / 'tweets' / 'processing_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"=== Data Processing Complete ===")
            self.logger.info(f"Processed {len(df_enhanced)} records with {df_enhanced.shape[1]} features")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data processing pipeline failed: {e}")
            return False

if __name__ == "__main__":
    # Test the data processor
    from config import Config
    
    logging.basicConfig(level=logging.INFO)
    config_obj = Config()
    
    processor = DataProcessor(config_obj.__dict__)
    success = processor.process_data_pipeline()
    print(f"Test completed: {'Success' if success else 'Failed'}")