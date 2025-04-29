from lumibot.entities import Asset, TradingFee
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime, timedelta
from finbert_utils import estimate_sentiment
from colorama import Fore, init
import numpy as np
import pandas as pd
import os
import logging

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
        
        # Initialize API if in live trading
        if not self.is_backtesting:
            from alpaca_trade_api.rest import REST
            self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        
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
        today = self.get_datetime()
        three_days_prior = today - timedelta(days=3)
        return today.strftime("%Y-%m-%d"), three_days_prior.strftime("%Y-%m-%d")

    def get_sentiment(self, symbol, company_name):
        """Get sentiment analysis for a specific stock"""
        if self.is_backtesting:
            # In backtesting, simulate sentiment
            sentiment_options = ["positive", "negative", "neutral"]
            probabilities = [0.3, 0.3, 0.4]  # Slightly biased toward neutral
            sentiment = np.random.choice(sentiment_options, p=probabilities)
            probability = np.random.uniform(0.7, 0.99)
            logger.info(f"Simulated sentiment for {symbol}: {sentiment} ({probability:.2f})")
            return probability, sentiment
            
        # For live trading, get real news
        today, three_days_prior = self.get_dates()
        try:
            news = self.api.get_news(symbol=symbol, start=three_days_prior, end=today)
            news = [ev.__dict__["_raw"]["headline"] for ev in news]
            
            if not news:
                logger.warning(f"No news found for {symbol}")
                return 0.5, "neutral"  # Default to neutral when no news
                
            probability, sentiment = estimate_sentiment(news)
            logger.info(f"Sentiment for {symbol}: {sentiment} ({probability:.2f})")
            return probability, sentiment
            
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
            return 0.5, "neutral"  # Default to neutral on error

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
            
            if quantity < 1:
                logger.info(f"Skipping {symbol} - quantity too small ({quantity})")
                continue
                
            # Get sentiment analysis
            probability, sentiment = self.get_sentiment(symbol, name)
            
            # Get current position if any
            position = self.get_position(Asset(symbol=symbol, asset_type="stock"))
            
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
                    
            elif sentiment == "negative" and probability > 0.85:
                # Sell on negative sentiment
                if position is not None and position.quantity > 0:
                    order = self.create_order(
                        Asset(symbol=symbol, asset_type="stock"),
                        position.quantity,
                        "sell",
                        type="market"
                    )
                    logger.info(f"SELL {symbol}: {position.quantity} shares at ~${last_price:.2f}")
                    self.submit_order(order)
                    self.last_trades[symbol] = "sell"
            
            # Record trade data for analysis
            self.trade_history.append({
                "date": self.get_datetime(),
                "symbol": symbol,
                "action": self.last_trades[symbol],
                "sentiment": sentiment,
                "probability": probability,
                "price": last_price,
                "quantity": quantity if self.last_trades[symbol] == "buy" else (position.quantity if position else 0)
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
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Stock configuration with custom weights
    stocks_config = [
        {"symbol": "AAPL", "name": "Apple", "weight": 0.25},
        {"symbol": "AMZN", "name": "Amazon", "weight": 0.25},
        {"symbol": "TSLA", "name": "Tesla", "weight": 0.25},
        {"symbol": "MSFT", "name": "Microsoft", "weight": 0.25}
    ]
    
    # Run backtest
    MultiStockTrader.run_backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        benchmark_asset="SPY",
        parameters={
            "stocks_config": stocks_config,
            "cash_at_risk": 1
        }
    )
    
    # For live trading (uncomment to use)
    
    from lumibot.brokers import Alpaca
    from lumibot.traders import Trader
    
    # Initialize broker
    broker = Alpaca(ALPACA_CREDS)
    
    # Create strategy
    strategy = MultiStockTrader(
        name="MultiStockTrader",
        broker=broker,
        parameters={
            "stocks_config": stocks_config,
            "cash_at_risk": 0.25
        }
    )
    
    # Run strategy
    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()
