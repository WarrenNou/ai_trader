# Add Trading fee to imports
from lumibot.entities import Asset, TradingFee
from lumibot.backtesting import CcxtBacktesting, YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime
from timedelta import Timedelta
from alpaca_trade_api import REST
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
        self.sleeptime = "1D"
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
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

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
            # For neutral sentiment, let's use a simpler approach
            today = self.get_datetime()
            
            # Get historical prices with length parameter
            asset = Asset(symbol=self.coin, asset_type=Asset.AssetType.CRYPTO)
            quote = Asset(symbol="USD", asset_type="crypto")
            
            try:
                # Simplified approach for getting historical prices
                historical_data = self.get_historical_prices(
                    self.coin,  # Pass symbol string directly
                    4,          # Number of bars
                    resolution="1day",  # Add explicit resolution parameter
                    quote="USD"  # Quote currency
                )
                
                # Continue processing if data is available
                if historical_data is not None and len(historical_data) >= 2:  # Need at least 2 bars for trend
                    # Sort by date, oldest first
                    sorted_data = sorted(historical_data, key=lambda x: x.timestamp)
                    
                    # Calculate price change
                    oldest_price = sorted_data[0].close
                    newest_price = sorted_data[-1].close
                    price_change = (newest_price - oldest_price) / oldest_price
                    
                    # Trading decisions based on trend
                    if price_change > 0.05 and (position is None or position.quantity == 0):
                        # Buy order logic remains the same
                        order = self.create_order(
                            self.coin,
                            quantity * 0.5,  # Half position size for trend following
                            "buy",
                            type="bracket",
                            take_profit_price=last_price * 1.3,
                            stop_loss_price=last_price * 0.9,
                            quote=Asset(symbol="USD", asset_type="crypto"),
                        )
                        print(Fore.LIGHTBLUE_EX + f"TREND BUY: {order}" + Fore.RESET)
                        self.submit_order(order)
                        self.last_trade = "buy"
                    
                    # If price is trending down at least 5% and we have a position, sell
                    elif price_change < -0.05 and position is not None and position.quantity > 0:
                        order = self.create_order(
                            self.coin,
                            position.quantity,
                            "sell",
                            quote=Asset(symbol="USD", asset_type="crypto"),
                        )
                        print(Fore.LIGHTRED_EX + f"TREND SELL: {order}" + Fore.RESET)
                        self.submit_order(order)
                        self.last_trade = "sell"
            except Exception as e:
                print(Fore.RED + f"Error getting historical data: {str(e)}" + Fore.RESET)


if __name__ == "__main__":
    from lumibot.backtesting import YahooDataBacktesting
    import numpy as np
    
    start_date = datetime(2025, 1, 1)
    end_date = datetime.now()
    
    coin = "BTC"
    coin_name = "bitcoin"
    
    # Use Yahoo Finance for backtesting
    results, strat_obj = MLTrader.run_backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        benchmark_asset="BTC-USD", 
        quote_asset=Asset(symbol="USD", asset_type="crypto"),
        # More realistic fees
        buy_trading_fees=[TradingFee(percent_fee=0.0035)],  # 0.35% is typical for crypto exchanges
        sell_trading_fees=[TradingFee(percent_fee=0.0035)],
        # Slippage to account for market impact
        slippage_model=lambda price, order_side, size: price * (1 + (0.001 * (-1 if order_side == "buy" else 1))),
        initial_capital=100000,
        parameters={
            "cash_at_risk": 0.40,  # Slightly more conservative
            "coin": coin,
            "coin_name": coin_name,
        }
    )
    
    # Print key performance metrics with error handling
    if results is not None and isinstance(results, dict):
        # Handle return
        strat_return = results.get('return', 0)
        if isinstance(strat_return, (int, float)):
            print(f"Strategy Return: {strat_return:.2%}")
        else:
            print(f"Strategy Return: {strat_return}")
        
        # Handle sharpe
        sharpe = results.get('sharpe', 0)
        if isinstance(sharpe, (int, float)):
            print(f"Strategy Sharpe: {sharpe:.2f}")
        else:
            print(f"Strategy Sharpe: {sharpe}")
        
        # Handle max_drawdown
        max_dd = results.get('max_drawdown', 0)
        if isinstance(max_dd, (int, float)):
            print(f"Strategy Max Drawdown: {max_dd:.2%}")
        else:
            print(f"Strategy Max Drawdown: {max_dd}")
        
        # Handle benchmark comparison
        benchmark_return = results.get('benchmark_return', 0)
        if isinstance(benchmark_return, (int, float)) and isinstance(strat_return, (int, float)):
            print(f"Strategy vs Benchmark: {strat_return - benchmark_return:.2%}")
        else:
            print("Strategy vs Benchmark: Not available")
    else:
        print("Backtest did not complete successfully.")
