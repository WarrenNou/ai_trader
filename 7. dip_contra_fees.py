# Add Trading fee to imports
from lumibot.entities import Asset, TradingFee
from lumibot.backtesting import CcxtBacktesting, YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime
from timedelta import Timedelta
from alpaca_trade_api import REST
from finbert_utils import estimate_sentiment
from colorama import Fore, init


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

        if last_price != None:
            print(Fore.YELLOW + f"{probability}, {sentiment}" + Fore.RESET)
            
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
                position = self.get_position(Asset(symbol=self.coin, asset_type="crypto"))
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


if __name__ == "__main__":
    from lumibot.backtesting import YahooDataBacktesting
    
    start_date = datetime(2024, 10, 15)
    end_date = datetime(2025, 4, 1)
    
    coin = "BTC"
    coin_name = "bitcoin"
    
    # Use Yahoo Finance for backtesting
    results, strat_obj = MLTrader.run_backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        benchmark_asset="BTC-USD", 
        quote_asset=Asset(symbol="USD", asset_type="crypto"),
        buy_trading_fees=[TradingFee(percent_fee=0.001)],
        sell_trading_fees=[TradingFee(percent_fee=0.001)],
        parameters={
            "cash_at_risk": 0.50,
            "coin": coin,
            "coin_name": coin_name,
        }
    )
