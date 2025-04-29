# Add Trading fee to imports
from lumibot.entities import Asset, TradingFee
from lumibot.backtesting import CcxtBacktesting, YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime
from timedelta import Timedelta
from alpaca_trade_api.rest import REST, TimeFrame  # Add TimeFrame here
from finbert_utils import estimate_sentiment
from colorama import Fore, init
import numpy as np


API_KEY = "PK71L3FGRWSNX7OHFWP0"
API_SECRET = "WsDYxovs8GbTcrA7PYelukQwDDqZaDQU0wjIlGwy"
BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {"API_KEY": API_KEY, "API_SECRET": API_SECRET, "PAPER": True}


class MLTrader(Strategy):

    def initialize(
        self, coin: str = "LTC", coin_name: str = "litecoin", cash_at_risk: float = 0.2
    ):
        self.set_market("24/7")
        self.sleeptime = "5H"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.coin = coin
        self.coin_name = coin_name
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(
            Asset(symbol=self.coin, asset_type=Asset.AssetType.CRYPTO),
            quote=Asset(symbol="USD", asset_type="crypto"),
        )
        if last_price == None:
            quantity = 0
        else:
            quantity = cash * self.cash_at_risk / last_price
        return cash, last_price, quantity

    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime("%Y-%m-%d"), three_days_prior.strftime("%Y-%m-%d")

    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(
            symbol=f"{self.coin}/USD", start=three_days_prior, end=today
        )
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    def on_trading_iteration(self):
        # Add this at the start of the method
        self.log_message(f"Starting trading iteration at {self.get_datetime()}")
        
        # Get portfolio data
        cash = self.get_cash()
        portfolio_value = self.portfolio_value
        positions = self.get_positions()
        equity_value = 0
        
        for position in positions:
            equity_value += position.market_value
        
        # Write portfolio data to file for the server to read
        with open("portfolio_data.json", "w") as f:
            import json
            json.dump({
                "cash": cash,
                "equity": equity_value,
                "total_value": portfolio_value,
                "timestamp": str(self.get_datetime())
            }, f)
        
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        # Add more logging throughout
        self.log_message(f"Current sentiment: {sentiment} with probability {probability:.4f}")

        if last_price is None:
            return

        print(Fore.YELLOW + f"{probability}, {sentiment}" + Fore.RESET)
        
        # Get current position if any
        position = self.get_position(Asset(symbol=self.coin, asset_type="crypto"))
        
        # Buy on negative sentiment (contrarian)
        if sentiment == "negative" and probability > 0.95 and cash > last_price:
            order = self.create_order(
                self.coin,
                quantity,
                "buy",
                type="bracket",
                take_profit_price=last_price * 1.5,
                stop_loss_price=last_price * 0.7,
                quote=Asset(symbol="USD", asset_type="crypto"),
            )
            print(Fore.LIGHTMAGENTA_EX + f"BUY ORDER: {order}" + Fore.RESET)
            self.submit_order(order)
            self.last_trade = "buy"
        
        # Sell on positive sentiment (contrarian)
        elif sentiment == "positive" and probability > 0.95 and self.last_trade == "buy":
            if position is not None and position.quantity > 0:
                order = self.create_order(
                    self.coin,
                    position.quantity,
                    "sell",
                    quote=Asset(symbol="USD", asset_type="crypto"),
                )
                print(Fore.LIGHTCYAN_EX + f"SELL ORDER: {order}" + Fore.RESET)
                self.submit_order(order)
                self.last_trade = "sell"
        
        # Handle neutral sentiment - add a simpler trend following approach
        elif sentiment == "neutral" and probability > 0.85:
            # For neutral sentiment, let's use Lumibot's data fetching
            try:
                # Define the asset for Lumibot's method
                crypto_asset = Asset(symbol=self.coin, asset_type="crypto")
                # Yahoo Finance often uses Ticker-USD format (e.g., BTC-USD)
                # Lumibot's get_historical_prices might handle the conversion,
                # or you might need to adjust self.coin if it fails.

                # Use Lumibot's get_historical_prices
                # It returns a Pandas DataFrame
                historical_data_df = self.get_historical_prices(
                    asset=crypto_asset,
                    length=4, # Get 4 periods (days due to sleeptime="1D")
                    quote=Asset(symbol="USD", asset_type="crypto") # Specify quote currency
                )

                # Check if DataFrame is valid and has enough data
                if historical_data_df is not None and not historical_data_df.empty and len(historical_data_df) >= 2:
                    # DataFrame index is usually datetime, already sorted
                    # Access the 'close' column
                    close_prices = historical_data_df['close']

                    # Calculate price change using DataFrame values
                    oldest_price = close_prices.iloc[0]    # First row (oldest)
                    newest_price = close_prices.iloc[-1]   # Last row (newest)

                    # Avoid division by zero if oldest price is 0
                    if oldest_price == 0:
                         price_change = 0
                    else:
                         price_change = (newest_price - oldest_price) / oldest_price

                    # Get current position (needed for trading decisions)
                    position = self.get_position(crypto_asset) # Use the asset object
                    # Get last price and quantity for placing orders
                    # Note: position_sizing already calls get_last_price which might use Yahoo too
                    cash, last_price, quantity = self.position_sizing()
                    if last_price is None: # Need last_price for bracket orders
                        self.log_message("Could not get last price for position sizing.", color="red")
                        return

                    # Trading decisions based on trend
                    if price_change > 0.05 and (position is None or position.quantity == 0):
                        # Buy order logic remains the same
                        order = self.create_order(
                            crypto_asset, # Use asset object
                            quantity * 0.5,  # Half position size
                            "buy",
                            type="bracket",
                            take_profit_price=last_price * 1.3,
                            stop_loss_price=last_price * 0.9,
                            # quote= is implicitly USD via the asset pair
                        )
                        print(Fore.LIGHTBLUE_EX + f"TREND BUY (Yahoo Data): {order}" + Fore.RESET)
                        self.submit_order(order)
                        self.last_trade = "buy"

                    # If price is trending down at least 5% and we have a position, sell
                    elif price_change < -0.05 and position is not None and position.quantity > 0:
                        order = self.create_order(
                            crypto_asset, # Use asset object
                            position.quantity,
                            "sell",
                            # quote= is implicitly USD
                        )
                        print(Fore.LIGHTRED_EX + f"TREND SELL (Yahoo Data): {order}" + Fore.RESET)
                        self.submit_order(order)
                        self.last_trade = "sell"
                else:
                    print(Fore.YELLOW + f"Not enough historical data received via get_historical_prices for {self.coin}." + Fore.RESET)

            except Exception as e:
                # More specific error message
                print(Fore.RED + f"Error using get_historical_prices for {self.coin}: {str(e)}" + Fore.RESET)


if __name__ == "__main__":
    from lumibot.brokers import Alpaca
    from lumibot.traders import Trader
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize Alpaca broker using the credentials dictionary
    broker = Alpaca(ALPACA_CREDS)
    
    # Parameters for the strategy
    coin = "BTC"
    coin_name = "bitcoin"
    
    # Create the strategy
    strategy = MLTrader(
        name="MLCryptoTrader",
        broker=broker,
        parameters={
            "cash_at_risk": 0.40,
            "coin": coin,
            "coin_name": coin_name,
        }
    )
    
    # Create trader and run the strategy
    trader = Trader()
    trader.add_strategy(strategy)
    trader.run_all()
