import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import setup_logger

class MarketAnalyzer:
    
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__, config.get('LOG_FILE'))
        
        # Analysis components
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.pca = PCA(n_components=2)
        
        # Analysis results storage
        self.analysis_results = {}
        self.trend_patterns = {}
        self.risk_metrics = {}
    
    def analyze_market_trends(self) -> bool:
        
        try:
            self.logger.info("=== Starting Market Trend Analysis ===")
            
            # Load signals data
            if not self.config['SIGNALS_FILE'].exists():
                self.logger.error("No signals data found. Run signal processing first.")
                return False
            
            signals_df = pd.read_parquet(self.config['SIGNALS_FILE'])
            
            if signals_df.empty:
                self.logger.warning("No signals found in data file")
                return False
            
            # Load tweet data for additional context
            tweets_df = pd.DataFrame()
            if self.config['TWEETS_FILE'].exists():
                tweets_df = pd.read_parquet(self.config['TWEETS_FILE'])
            
            # Perform comprehensive analysis
            trend_analysis = self.perform_trend_analysis(signals_df)
            sentiment_analysis = self.perform_sentiment_clustering(signals_df)
            volatility_analysis = self.perform_volatility_analysis(signals_df)
            pattern_analysis = self.perform_pattern_recognition(signals_df)
            
            if not tweets_df.empty:
                user_analysis = self.perform_user_influence_analysis(tweets_df)
                content_analysis = self.perform_content_analysis(tweets_df)
            else:
                user_analysis = {}
                content_analysis = {}
            
            # Combine all analyses
            comprehensive_analysis = {
                'trend_analysis': trend_analysis,
                'sentiment_analysis': sentiment_analysis,
                'volatility_analysis': volatility_analysis,
                'pattern_analysis': pattern_analysis,
                'user_analysis': user_analysis,
                'content_analysis': content_analysis,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Save analysis results
            success = self.save_analysis_results(comprehensive_analysis)
            
            if success:
                self.logger.info("=== Market Analysis Complete ===")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Market trend analysis failed: {e}")
            return False
    
    def perform_trend_analysis(self, signals_df: pd.DataFrame) -> Dict:
        
        
        # Ensure hour column is datetime
        signals_df['hour'] = pd.to_datetime(signals_df['hour'])
        signals_df = signals_df.sort_values('hour')
        
        # Calculate trend metrics
        sentiment_trend = signals_df['avg_sentiment'].rolling(window=6).mean()  # 6-hour moving average
        sentiment_momentum = signals_df['avg_sentiment'].diff()
        
        # Trend strength
        trend_strength = abs(sentiment_trend).rolling(window=12).mean()
        
        # Market regime periods
        bullish_periods = (signals_df['avg_sentiment'] > 0.1).sum()
        bearish_periods = (signals_df['avg_sentiment'] < -0.1).sum()
        neutral_periods = len(signals_df) - bullish_periods - bearish_periods
        
        # Trend reversals (sentiment momentum changes)
        reversals = ((sentiment_momentum.shift(1) > 0) & (sentiment_momentum < 0)) | \
                   ((sentiment_momentum.shift(1) < 0) & (sentiment_momentum > 0))
        reversal_count = reversals.sum()
        
        trend_analysis = {
            'overall_trend': {
                'direction': 'Bullish' if signals_df['avg_sentiment'].mean() > 0.05 
                          else 'Bearish' if signals_df['avg_sentiment'].mean() < -0.05 
                          else 'Neutral',
                'strength': float(trend_strength.mean()),
                'consistency': float(1 - (signals_df['avg_sentiment'].std() / abs(signals_df['avg_sentiment'].mean())) 
                                   if signals_df['avg_sentiment'].mean() != 0 else 0)
            },
            'regime_distribution': {
                'bullish_periods': int(bullish_periods),
                'bearish_periods': int(bearish_periods),
                'neutral_periods': int(neutral_periods),
                'total_periods': len(signals_df)
            },
            'momentum_indicators': {
                'avg_momentum': float(sentiment_momentum.mean()),
                'momentum_volatility': float(sentiment_momentum.std()),
                'trend_reversals': int(reversal_count),
                'reversal_frequency': float(reversal_count / len(signals_df)) if len(signals_df) > 0 else 0
            },
            'key_levels': {
                'resistance_level': float(signals_df['avg_sentiment'].quantile(0.9)),
                'support_level': float(signals_df['avg_sentiment'].quantile(0.1)),
                'median_sentiment': float(signals_df['avg_sentiment'].median())
            }
        }
        
        return trend_analysis
    
    def perform_sentiment_clustering(self, signals_df: pd.DataFrame) -> Dict:
        
        
        # Prepare features for clustering
        features = signals_df[['avg_sentiment', 'sentiment_std', 'tweet_count', 'total_engagement']].fillna(0)
        
        # Normalize features
        features_normalized = (features - features.mean()) / features.std()
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(features_normalized)
        signals_df['sentiment_cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(3):
            cluster_data = signals_df[signals_df['sentiment_cluster'] == cluster_id]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'avg_sentiment': float(cluster_data['avg_sentiment'].mean()),
                'avg_engagement': float(cluster_data['total_engagement'].mean()),
                'characteristics': self.characterize_cluster(cluster_data)
            }
        
        # PCA for dimensionality reduction and visualization
        pca_features = self.pca.fit_transform(features_normalized)
        
        sentiment_clustering = {
            'cluster_analysis': cluster_analysis,
            'cluster_centers': self.kmeans.cluster_centers_.tolist(),
            'pca_variance_ratio': self.pca.explained_variance_ratio_.tolist()
        }
        
        return sentiment_clustering
    
    def characterize_cluster(self, cluster_data: pd.DataFrame) -> str:
        
        avg_sentiment = cluster_data['avg_sentiment'].mean()
        avg_engagement = cluster_data['total_engagement'].mean()
        
        if avg_sentiment > 0.1 and avg_engagement > cluster_data['total_engagement'].median():
            return "High Positive Sentiment with High Engagement"
        elif avg_sentiment > 0.1:
            return "Positive Sentiment"
        elif avg_sentiment < -0.1 and avg_engagement > cluster_data['total_engagement'].median():
            return "High Negative Sentiment with High Engagement"
        elif avg_sentiment < -0.1:
            return "Negative Sentiment"
        else:
            return "Neutral Sentiment"
    
    def perform_volatility_analysis(self, signals_df: pd.DataFrame) -> Dict:
        
        
        # Calculate volatility metrics
        sentiment_volatility = signals_df['avg_sentiment'].rolling(window=6).std()
        volume_volatility = signals_df['tweet_count'].rolling(window=6).std()
        
        # Risk metrics
        var_95 = signals_df['avg_sentiment'].quantile(0.05)  # Value at Risk (95%)
        max_drawdown = self.calculate_max_drawdown(signals_df['avg_sentiment'])
        
        # Volatility regimes
        high_vol_threshold = sentiment_volatility.quantile(0.75)
        high_vol_periods = (sentiment_volatility > high_vol_threshold).sum()
        
        volatility_analysis = {
            'volatility_metrics': {
                'avg_sentiment_volatility': float(sentiment_volatility.mean()),
                'avg_volume_volatility': float(volume_volatility.mean()),
                'volatility_of_volatility': float(sentiment_volatility.std())
            },
            'risk_metrics': {
                'value_at_risk_95': float(var_95),
                'max_drawdown': float(max_drawdown),
                'downside_deviation': float(signals_df[signals_df['avg_sentiment'] < 0]['avg_sentiment'].std())
            },
            'volatility_regimes': {
                'high_volatility_periods': int(high_vol_periods),
                'low_volatility_periods': int(len(signals_df) - high_vol_periods),
                'volatility_clustering': self.detect_volatility_clustering(sentiment_volatility)
            }
        }
        
        return volatility_analysis
    
    def calculate_max_drawdown(self, series: pd.Series) -> float:
        
        cumulative = (1 + series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def detect_volatility_clustering(self, volatility_series: pd.Series) -> Dict:
        
        # Calculate autocorrelation of squared returns (volatility clustering indicator)
        squared_series = volatility_series.fillna(0) ** 2
        
        clustering_metrics = {
            'autocorr_lag1': float(squared_series.autocorr(lag=1)),
            'autocorr_lag3': float(squared_series.autocorr(lag=3)),
            'persistence': float(volatility_series.autocorr(lag=1))
        }
        
        return clustering_metrics
    
    def perform_pattern_recognition(self, signals_df: pd.DataFrame) -> Dict:
        
        
        # Signal patterns
        signal_sequences = signals_df['signal_strength'].rolling(window=3).apply(
            lambda x: ''.join(map(str, x.astype(int))), raw=False
        )
        
        pattern_counts = signal_sequences.value_counts().head(10).to_dict()
        
        # Trend patterns
        sentiment_direction = signals_df['avg_sentiment'].diff().apply(
            lambda x: 'U' if x > 0 else 'D' if x < 0 else 'S'
        )
        
        trend_patterns = sentiment_direction.rolling(window=4).apply(
            lambda x: ''.join(x), raw=False
        ).value_counts().head(10).to_dict()
        
        pattern_analysis = {
            'signal_patterns': {
                'most_common_sequences': pattern_counts,
                'pattern_diversity': len(signal_sequences.unique())
            },
            'trend_patterns': {
                'direction_sequences': trend_patterns,
                'trend_persistence': float(sentiment_direction.eq(sentiment_direction.shift()).mean())
            },
            'cyclical_patterns': self.detect_cyclical_patterns(signals_df)
        }
        
        return pattern_analysis
    
    def detect_cyclical_patterns(self, signals_df: pd.DataFrame) -> Dict:
        
        
        # Add time features
        signals_df['hour'] = pd.to_datetime(signals_df['hour'])
        signals_df['hour_of_day'] = signals_df['hour'].dt.hour
        signals_df['day_of_week'] = signals_df['hour'].dt.dayofweek
        
        # Hourly patterns
        hourly_sentiment = signals_df.groupby('hour_of_day')['avg_sentiment'].mean()
        
        # Daily patterns
        daily_sentiment = signals_df.groupby('day_of_week')['avg_sentiment'].mean()
        
        cyclical_patterns = {
            'hourly_patterns': {
                'peak_sentiment_hour': int(hourly_sentiment.idxmax()),
                'trough_sentiment_hour': int(hourly_sentiment.idxmin()),
                'hourly_variance': float(hourly_sentiment.var())
            },
            'daily_patterns': {
                'best_day': int(daily_sentiment.idxmax()),
                'worst_day': int(daily_sentiment.idxmin()),
                'daily_variance': float(daily_sentiment.var())
            }
        }
        
        return cyclical_patterns
    
    def perform_user_influence_analysis(self, tweets_df: pd.DataFrame) -> Dict:
        
        
        # Top influential users
        user_influence = tweets_df.groupby('username').agg({
            'engagement_score': 'sum',
            'user_followers': 'first',
            'content': 'count'
        }).rename(columns={'content': 'tweet_count'})
        
        user_influence['influence_ratio'] = user_influence['engagement_score'] / user_influence['tweet_count']
        top_users = user_influence.nlargest(10, 'influence_ratio')
        
        user_analysis = {
            'influence_metrics': {
                'total_unique_users': int(tweets_df['username'].nunique()),
                'avg_tweets_per_user': float(len(tweets_df) / tweets_df['username'].nunique()),
                'user_concentration': float(tweets_df['username'].value_counts().head(10).sum() / len(tweets_df))
            },
            'top_influencers': top_users.to_dict('index'),
            'verification_impact': {
                'verified_users': int(tweets_df.get('user_verified', pd.Series([False])).sum()),
                'verified_engagement_avg': float(tweets_df[tweets_df.get('user_verified', False)]['engagement_score'].mean())
                if 'user_verified' in tweets_df.columns else 0
            }
        }
        
        return user_analysis
    
    def perform_content_analysis(self, tweets_df: pd.DataFrame) -> Dict:
        
        
        # Content characteristics
        content_stats = {
            'avg_length': float(tweets_df['content'].str.len().mean()),
            'length_variance': float(tweets_df['content'].str.len().var()),
            'word_count_avg': float(tweets_df['content'].str.split().str.len().mean())
        }
        
        # Hashtag analysis
        all_hashtags = []
        for hashtag_list in tweets_df['hashtags']:
            if isinstance(hashtag_list, list):
                all_hashtags.extend(hashtag_list)
        
        hashtag_freq = pd.Series(all_hashtags).value_counts().head(20).to_dict()
        
        # Market keyword analysis
        market_keywords_freq = {}
        if 'market_keywords' in tweets_df.columns:
            all_keywords = []
            for keyword_list in tweets_df['market_keywords']:
                if isinstance(keyword_list, list):
                    all_keywords.extend(keyword_list)
            market_keywords_freq = pd.Series(all_keywords).value_counts().head(15).to_dict()
        
        content_analysis = {
            'content_statistics': content_stats,
            'hashtag_analysis': {
                'most_popular_hashtags': hashtag_freq,
                'hashtag_diversity': len(set(all_hashtags))
            },
            'keyword_analysis': {
                'market_keywords_frequency': market_keywords_freq,
                'tweets_with_keywords': int(tweets_df['market_keywords'].apply(len).gt(0).sum())
                if 'market_keywords' in tweets_df.columns else 0
            }
        }
        
        return content_analysis
    
    def save_analysis_results(self, analysis_results: Dict) -> bool:
        
        try:
            # Convert analysis to DataFrame for Parquet storage
            analysis_summary = {
                'analysis_timestamp': analysis_results['analysis_timestamp'],
                'trend_direction': analysis_results['trend_analysis']['overall_trend']['direction'],
                'trend_strength': analysis_results['trend_analysis']['overall_trend']['strength'],
                'bullish_periods': analysis_results['trend_analysis']['regime_distribution']['bullish_periods'],
                'bearish_periods': analysis_results['trend_analysis']['regime_distribution']['bearish_periods'],
                'sentiment_volatility': analysis_results['volatility_analysis']['volatility_metrics']['avg_sentiment_volatility'],
                'max_drawdown': analysis_results['volatility_analysis']['risk_metrics']['max_drawdown'],
                'pattern_diversity': analysis_results['pattern_analysis']['signal_patterns']['pattern_diversity']
            }
            
            # Save summary as DataFrame
            summary_df = pd.DataFrame([analysis_summary])
            summary_df.to_parquet(self.config['ANALYSIS_FILE'], index=False)
            
            # Save detailed results as JSON
            detailed_file = self.config['DATA_DIR'] / 'analysis' / 'detailed_analysis.json'
            with open(detailed_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            # Save analysis metadata
            metadata = {
                'analysis_info': {
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'components_analyzed': list(analysis_results.keys())
                },
                'key_insights': {
                    'market_direction': analysis_results['trend_analysis']['overall_trend']['direction'],
                    'risk_level': 'High' if analysis_results['volatility_analysis']['volatility_metrics']['avg_sentiment_volatility'] > 0.3 else 'Normal',
                    'activity_level': 'High' if analysis_results.get('user_analysis', {}).get('influence_metrics', {}).get('total_unique_users', 0) > 100 else 'Normal'
                }
            }
            
            metadata_file = self.config['DATA_DIR'] / 'analysis' / 'analysis_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Analysis results saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save analysis results: {e}")
            return False

if __name__ == "__main__":
    # Test the analyzer
    from config import Config
    
    logging.basicConfig(level=logging.INFO)
    config_obj = Config()
    
    analyzer = MarketAnalyzer(config_obj.__dict__)
    success = analyzer.analyze_market_trends()
    print(f"Test completed: {'Success' if success else 'Failed'}")