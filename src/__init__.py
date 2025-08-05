__version__ = "1.0.0"
__author__ = "Market Intelligence Team"
__description__ = "Advanced Twitter sentiment analysis for Indian stock market trading signals"

# Import main classes for easy access
from .data_collector import MarketDataCollector
from .signal_processor import SignalProcessor
from .data_processor import DataProcessor
from .analyzer import MarketAnalyzer
from .utils import *

__all__ = [
    'MarketDataCollector',
    'SignalProcessor', 
    'DataProcessor',
    'MarketAnalyzer'
]