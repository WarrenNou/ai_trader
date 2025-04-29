from lumibot.entities import Asset, TradingFee
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime, timedelta
from finbert_utils import SentimentAnalyzer
from colorama import Fore, init
import numpy as np
import pandas as pd
import os
import logging
from alpaca_trade_api.rest import REST
import gc  # Import garbage collector module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API credentials for Alpaca
API_KEY = "PK71L3FGRWSNX7OHFWP0"
API_SECRET = "WsDYxovs8GbTcrA7PYelukQwDDqZaDQU0wjIlGwy"
BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {"API_KEY": API_KEY, "API_SECRET": API_SECRET, "PAPER": True}

class MultiStockTrader(Strategy):
    def initialize(self, stocks_config=None, cash_at_risk=0.2):
        self.set_market("NYSE")  # or "NASDAQ" depending on your preference
        self.sleeptime = "1D"
        
        # Default stock configuration if none provided
        if stocks_config is None:
            self.stocks_config = [
                {"symbol": "AAPL", "name": "Apple", "weight": 0.33},
                {"symbol": "AMZN", "name": "Amazon", "weight": 0.33},
                {"symbol": "TSLA", "name": "Tesla", "weight": 0.34}
            ]
        else:
            self.stocks_config = stocks_config
            
        # Validate weights sum to 1.0
        total_weight = sum(stock["weight"] for stock in self.stocks_config)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Stock weights sum to {total_weight}, not 1.0. Normalizing...")
            for stock in self.stocks_config:
                stock["weight"] = stock["weight"] / total_weight
        
        self.cash_at_risk = cash_at_risk
        self.position_trackers = {stock["symbol"]: None for stock in self.stocks_config}  # Renamed from self.positions
        self.last_trades = {stock["symbol"]: None for stock in self.stocks_config}
        
        # Performance tracking
        self.trade_history = []
        
        # Initialize API always, needed for news in backtesting too
        # Using Alpaca API for news fetching in both live and backtesting modes.
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        
        # Initialize Sentiment Analyzer once
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            logger.info("Sentiment Analyzer initialized successfully.")
        except NameError:
            logger.error("SentimentAnalyzer class not found. Make sure it's defined in and imported from finbert_utils.py.")
            self.sentiment_analyzer = None
        except Exception as e:
            logger.error(f"Failed to initialize Sentiment Analyzer: {e}")
            self.sentiment_analyzer = None

        logger.info(f"Initialized MultiStockTrader with {len(self.stocks_config)} stocks")
        for stock in self.stocks_config:
            logger.info(f"  - {stock['symbol']} ({stock['name']}): {stock['weight']*100:.1f}%")

    def position_sizing(self, symbol, weight):
        """Calculate position size for a specific stock"""
        cash = self.get_cash() * self.cash_at_risk * weight
        last_price = self.get_last_price(Asset(symbol=symbol, asset_type="stock"))
        
        if last_price is None:
            logger.warning(f"Could not get last price for {symbol}")
            return cash, last_price, 0
            
        quantity = int(cash / last_price)  # Whole shares only
        return cash, last_price, quantity

    def get_dates(self):
        """Get date range for news retrieval"""
        # Use the strategy's current datetime, which reflects the backtest time
        today = self.get_datetime()
        three_days_prior = today - timedelta(days=3)
        # Ensure dates are timezone-naive or consistently timezone-aware for Alpaca API
        # Alpaca API typically expects RFC3339 format or YYYY-MM-DD
        # Using YYYY-MM-DD format here
        return today.strftime("%Y-%m-%d"), three_days_prior.strftime("%Y-%m-%d")

    def get_sentiment(self, symbol, company_name):
        """Get sentiment analysis for a specific stock using the pre-loaded analyzer"""
        if not self.sentiment_analyzer:
            logger.warning("Sentiment Analyzer not available. Returning neutral.")
            return 0.5, "neutral"

        today_str, three_days_prior_str = self.get_dates()
        logger.info(f"Fetching news for {symbol} from {three_days_prior_str} to {today_str}")
        try:
            # Limit the number of news items fetched from Alpaca directly
            # Note: Check Alpaca API documentation if 'limit' is supported this way, 
            # otherwise, fetch all and slice afterwards. Assuming slicing for now.
            news = self.api.get_news(symbol=symbol, start=three_days_prior_str, end=today_str) 
            
            headlines = [ev.__dict__["_raw"]["headline"] for ev in news if "headline" in ev.__dict__["_raw"]]

            # <<< Explicitly limit the number of headlines to analyze >>>
            max_headlines_to_analyze = 25 # Adjust this number as needed
            if len(headlines) > max_headlines_to_analyze:
                logger.info(f"Analyzing only the latest {max_headlines_to_analyze} headlines for {symbol} (out of {len(headlines)})")
                # Assuming Alpaca returns news chronologically (newest first might be safer if available)
                # Slicing the list to keep only the most recent ones based on typical API return order
                headlines = headlines[:max_headlines_to_analyze] 

            if not headlines:
                logger.warning(f"No news headlines found for {symbol} between {three_days_prior_str} and {today_str} via Alpaca.")
                return 0.5, "neutral"

            probability, sentiment = self.sentiment_analyzer.analyze(headlines)
            logger.info(f"Sentiment for {symbol} based on {len(headlines)} analyzed headlines: {sentiment} ({probability:.2f})")
            return probability, sentiment

        except Exception as e:
            # Consider more specific exception handling if possible
            logger.error(f"Error getting news or sentiment for {symbol}: {str(e)}")
            return 0.5, "neutral"

    def on_trading_iteration(self):
        """Main trading logic executed on each iteration"""
        logger.info(f"Starting trading iteration at {self.get_datetime()}")
        
        # Process each stock in our configuration
        for stock in self.stocks_config:
            symbol = stock["symbol"]
            name = stock["name"]
            weight = stock["weight"]
            
            # Get cash and position sizing
            cash, last_price, quantity = self.position_sizing(symbol, weight)
            
            if quantity < 1 or last_price is None: # Added check for last_price
                logger.info(f"Skipping {symbol} - quantity {quantity} or price {last_price} invalid")
                continue
                
            # Get sentiment analysis
            probability, sentiment = self.get_sentiment(symbol, name)
            
            # Get current position if any
            position = self.get_position(Asset(symbol=symbol, asset_type="stock"))
            
            trade_executed = False # Flag to track if a trade happened for this stock
            action_taken = None # Track the action taken

            # Trading logic based on sentiment
            if sentiment == "positive" and probability > 0.85:
                # Buy on positive sentiment
                if position is None or position.quantity == 0:
                    order = self.create_order(
                        Asset(symbol=symbol, asset_type="stock"),
                        quantity,
                        "buy",
                        type="market"
                    )
                    logger.info(f"BUY {symbol}: {quantity} shares at ~${last_price:.2f}")
                    self.submit_order(order)
                    self.last_trades[symbol] = "buy"
                    trade_executed = True
                    action_taken = "buy"
                    
            elif sentiment == "negative" and probability > 0.85:
                # Sell on negative sentiment
                if position is not None and position.quantity > 0:
                    sell_quantity = position.quantity # Sell the entire position
                    order = self.create_order(
                        Asset(symbol=symbol, asset_type="stock"),
                        sell_quantity, 
                        "sell",
                        type="market"
                    )
                    logger.info(f"SELL {symbol}: {sell_quantity} shares at ~${last_price:.2f}")
                    self.submit_order(order)
                    self.last_trades[symbol] = "sell"
                    trade_executed = True
                    action_taken = "sell"
            
            # Record trade data ONLY if a trade was executed
            if trade_executed:
                self.trade_history.append({
                    "date": self.get_datetime(),
                    "symbol": symbol,
                    "action": action_taken, # Use the action taken in this iteration
                    "sentiment": sentiment, # Log sentiment that triggered the trade
                    "probability": probability, # Log probability for the trade
                    "price": last_price,
                    # Log actual quantity bought or sold
                    "quantity": quantity if action_taken == "buy" else sell_quantity 
                })
            
        # Log portfolio status at end of iteration
        self.log_portfolio_status()
    
    def log_portfolio_status(self):
        """Log current portfolio status"""
        portfolio_value = self.portfolio_value
        cash = self.get_cash()
        
        logger.info(f"Portfolio value: ${portfolio_value:.2f}")
        logger.info(f"Cash: ${cash:.2f}")
        
        # Log positions
        positions = self.get_positions()
        if positions:
            logger.info("Current positions:")
            for position in positions:
                symbol = position.symbol
                quantity = position.quantity
                price = self.get_last_price(Asset(symbol=symbol, asset_type="stock"))
                value = quantity * price if price else 0
                logger.info(f"  - {symbol}: {quantity} shares, ${value:.2f}")
        else:
            logger.info("No open positions")

if __name__ == "__main__":
    # For backtesting
    start_date = datetime(2025, 4, 1) 
    # Use a fixed end date for reproducible backtests, 
    # or keep datetime.now() for testing up to the current moment.
    # Example: end_date = datetime(2025, 4, 28) 
    end_date = datetime.now() 
    
    # Stock configuration with custom weights
    stocks_config = [
        {"symbol": "AAPL", "name": "Apple", "weight": 0.25},
        {"symbol": "AMZN", "name": "Amazon", "weight": 0.25},
        {"symbol": "TSLA", "name": "Tesla", "weight": 0.25},
        {"symbol": "MSFT", "name": "Microsoft", "weight": 0.25}
    ]
    
    logger.info("--- Starting Backtest ---")
    # Run backtest
    MultiStockTrader.run_backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        benchmark_asset="SPY", # Using SPY as benchmark
        parameters={
            "stocks_config": stocks_config,
            "cash_at_risk": 1.0 # Using 100% cash at risk for backtest allocation
        }
    )
    logger.info("--- Backtest Finished ---")
    
    # For live trading (Keep this commented out unless running live)
    """ 
    from lumibot.brokers import Alpaca
    from lumibot.traders import Trader
    
    logger.info("--- Setting up Live Trading ---")
    # Initialize broker
    broker = Alpaca(ALPACA_CREDS)
    
    # Create strategy instance for live trading
    strategy = MultiStockTrader(
        name="MultiStockTrader_Live", # Give live instance a unique name
        broker=broker,
        parameters={
            "stocks_config": stocks_config,
            "cash_at_risk": 0.25 # Use desired live cash_at_risk
        }
    )
    
    # Run strategy live
    trader = Trader()
    trader.add_strategy(strategy)
    logger.info("--- Starting Live Trading ---")
    trader.run_all()
    """
