import os
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit
import pandas as pd
import json

from config import config
from src.data_collector import MarketDataCollector
from src.signal_processor import SignalProcessor
from src.analyzer import MarketAnalyzer
from src.generate_dummy_data import generate_dummy_data
from pathlib import Path

# Initialize Config
app_config = config['default']()

# Check and Generate Dummy Data if Files Don't Exist
if not app_config.TWEETS_FILE.exists() or not app_config.SIGNALS_FILE.exists() or not app_config.ANALYSIS_FILE.exists():
    print("Generating dummy data as Parquet files are missing...")
    generate_dummy_data(app_config.DATA_DIR)
else:
    print("Parquet data files already exist. Skipping dummy data generation.")

def create_app(config_name=None):
    """Professional application factory"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'production')
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Enable CORS for React frontend
    # CORS(app, origins=app.config['CORS_ORIGINS'])
    CORS(app)
    
    # Setup professional logging
    setup_logging(app)
    
    # Initialize core components
    data_collector = MarketDataCollector(app.config)
    signal_processor = SignalProcessor(app.config)
    analyzer = MarketAnalyzer(app.config)
    
    # Setup automated scheduler
    scheduler = BackgroundScheduler()
    scheduler.start()
    
    # Schedule professional data collection every 4 hours
    scheduler.add_job(
        func=collect_and_process_data,
        trigger=IntervalTrigger(hours=4),
        id='market_data_collection',
        name='Professional Market Data Collection',
        replace_existing=True,
        args=[data_collector, signal_processor, analyzer, app.logger]
    )
    
    atexit.register(lambda: scheduler.shutdown())
    
    # Professional API endpoints
    @app.route('/api/health')
    def health_check():
        """Professional health check endpoint"""
        try:
            status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "environment": app.config['ENV'],
                "data_files": {
                    "tweets": app.config['TWEETS_FILE'].exists(),
                    "signals": app.config['SIGNALS_FILE'].exists(),
                    "analysis": app.config['ANALYSIS_FILE'].exists()
                },
                "system_info": {
                    "max_tweets": app.config['MAX_TWEETS'],
                    "hashtags": app.config['HASHTAGS']
                }
            }
            return jsonify(status)
        except Exception as e:
            app.logger.error(f"Health check failed: {e}")
            return jsonify({"status": "unhealthy", "error": str(e)}), 500
    
    @app.route('/api/summary')
    def get_summary():
        """Get professional dashboard summary"""
        try:
            summary = {
                "last_updated": "Never",
                "tweets": 0,
                "signals": 0,
                "unique_users": 0,
                "avg_sentiment": 0.0,
                "bullish_signals": 0,
                "bearish_signals": 0,
                "neutral_signals": 0
            }
            
            # Get tweets summary
            if app.config['TWEETS_FILE'].exists():
                df = pd.read_parquet(app.config['TWEETS_FILE'])
                summary.update({
                    "tweets": len(df),
                    "unique_users": df['username'].nunique() if not df.empty else 0,
                    "last_updated": df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S') if not df.empty else "Never"
                })
            
            # Get signals summary
            if app.config['SIGNALS_FILE'].exists():
                signals_df = pd.read_parquet(app.config['SIGNALS_FILE'])
                if not signals_df.empty:
                    summary.update({
                        "signals": len(signals_df),
                        "avg_sentiment": float(signals_df['avg_sentiment'].mean()),
                        "bullish_signals": int((signals_df['signal_strength'] > 0).sum()),
                        "bearish_signals": int((signals_df['signal_strength'] < 0).sum()),
                        "neutral_signals": int((signals_df['signal_strength'] == 0).sum())
                    })

            # tweets_file = 'data/tweets/market_tweets.parquet'
            # signals_file = 'data/signals/trading_signals.parquet'

            # df_tweets = pd.read_parquet(tweets_file)
            # df_signals = pd.read_parquet(signals_file)

            # print("Tweets File Rows:", len(df_tweets))
            # print("Signals File Rows:", len(df_signals))

            
            return jsonify(summary)
        except Exception as e:
            app.logger.error(f"Summary API error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/signals')
    def get_signals():
        """Get professional trading signals"""
        try:
            if not app.config['SIGNALS_FILE'].exists():
                return jsonify({"signals": []})
            
            df = pd.read_parquet(app.config['SIGNALS_FILE'])
            
            # Get recent signals (last 48 hours)
            signals = df.tail(48).to_dict('records')
            
            # Format for frontend
            formatted_signals = []
            for signal in signals:
                formatted_signals.append({
                    "hour": signal.get('hour', ''),
                    "avg_sentiment": round(float(signal.get('avg_sentiment', 0)), 4),
                    "signal_strength": int(signal.get('signal_strength', 0)),
                    "confidence_interval": round(float(signal.get('confidence_interval', 0)), 4),
                    "signal_quality": round(float(signal.get('signal_quality', 0)), 2),
                    "tweet_count": int(signal.get('tweet_count', 0)),
                    "total_engagement": int(signal.get('total_engagement', 0))
                })
            
            return jsonify({"signals": formatted_signals})
        except Exception as e:
            app.logger.error(f"Signals API error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/analysis')
    def get_analysis():
        """Get professional market analysis"""
        try:
            if not app.config['ANALYSIS_FILE'].exists():
                return jsonify({"analysis": []})
            
            df = pd.read_parquet(app.config['ANALYSIS_FILE'])
            analysis = df.to_dict('records')
            
            return jsonify({"analysis": analysis})
        except Exception as e:
            app.logger.error(f"Analysis API error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/collect', methods=['GET','POST'])
    def trigger_collection():
        """Manually trigger professional data collection"""
        try:
            app.logger.info("Manual data collection triggered via API")
            result = collect_and_process_data(data_collector, signal_processor, analyzer, app.logger)
            return jsonify({
                "status": "success",
                "message": "Data collection completed successfully",
                "result": result
            })
        except Exception as e:
            app.logger.error(f"Manual collection error: {e}")
            return jsonify({
                "status": "error",
                "message": "Data collection failed",
                "error": str(e)
            }), 500
    
    @app.route('/api/status')
    def get_status():
        """Get detailed system status"""
        try:
            status = {
                "system": "Market Intelligence System",
                "timestamp": datetime.now().isoformat(),
                "uptime": "Active",
                "data_collection": {
                    "last_run": "Unknown",
                    "next_run": "Scheduled",
                    "status": "Ready"
                },
                "performance": {
                    "memory_usage": "Normal",
                    "cpu_usage": "Normal"
                }
            }
            
            # Check if data exists and get timestamps
            if app.config['TWEETS_FILE'].exists():
                df = pd.read_parquet(app.config['TWEETS_FILE'])
                if not df.empty:
                    status["data_collection"]["last_run"] = df['collection_timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            
            return jsonify(status)
        except Exception as e:
            app.logger.error(f"Status API error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/validate-data', methods=['GET'])
    def validate_data():
        """Quick Validation for Signals and Analysis Files"""
        result = {}

        try:
            df = pd.read_parquet(app.config['SIGNALS_FILE'])
            result['signals_shape'] = df.shape
            result['signals_sample'] = df.head(5).to_dict(orient='records')
        except Exception as e:
            result['signals_error'] = str(e)

        try:
            analysis_df = pd.read_parquet(app.config['ANALYSIS_FILE'])
            result['analysis_shape'] = analysis_df.shape
            result['analysis_sample'] = analysis_df.head(5).to_dict(orient='records')
        except Exception as e:
            result['analysis_error'] = str(e)

        return jsonify(result)
    
    return app

def setup_logging(app):
    """Setup professional logging system"""
    if not app.debug:
        log_file = app.config['LOG_FILE']
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s : %(message)s'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Professional Market Intelligence System startup')

def collect_and_process_data(collector, processor, analyzer, logger):
    """Professional data collection and processing pipeline"""
    try:
        logger.info("Starting professional data collection pipeline")
        
        # Step 1: Collect market tweets
        tweets_collected = collector.collect_market_tweets()
        logger.info(f"Collected {tweets_collected} tweets")
        
        # Step 2: Generate trading signals
        signals_generated = processor.generate_trading_signals()
        logger.info(f"Generated {signals_generated} trading signals")
        
        # Step 3: Run market analysis
        analysis_results = analyzer.analyze_market_trends()
        logger.info(f"Completed market analysis: {analysis_results}")
        
        return {
            "tweets_collected": tweets_collected,
            "signals_generated": signals_generated,
            "analysis_completed": analysis_results,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Data processing pipeline failed: {e}")
        raise

    

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 9999))
    app.run(host='0.0.0.0', port=port, debug=False)