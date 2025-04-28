from lumibot.entities import Asset
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime, timedelta
from colorama import Fore
import json
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from llmprompts import get_web_deets, prompt_template


class Response(BaseModel):
    sentiment: str
    score: float


class StockTrader(Strategy):
    def initialize(self, cash_at_risk: float = 0.2, ticker: str = "AAPL", company_name: str = "Apple"):
        self.set_market("NYSE")  # Changed from "us_equities" to "NYSE"
        self.sleeptime = "1D"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.ticker = ticker
        self.company_name = company_name
        self.llm = OllamaLLM(model="deepseek-r1:7b", format="json")

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(
            Asset(symbol=self.ticker, asset_type="stock")
        )
        if last_price is None:
            quantity = 0
        else:
            quantity = int(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity

    def get_dates(self):
        today = self.get_datetime()
        day_prior = today - timedelta(days=3)  # More days for stock news
        return today.strftime("%Y-%m-%d"), day_prior.strftime("%Y-%m-%d")

    def get_sentiment(self):
        """Get sentiment from news data"""
        # Get dates for news retrieval
        today_date, prior_date = self.get_dates()
        
        # Get news data for the company with date parameters
        news = get_web_deets(prior_date, today_date, self.company_name)
        
        # Get the prompt from prompt_template function - pass only the news data
        prompt = prompt_template(news)
        
        try:
            # Get sentiment from LLM
            response = self.llm.invoke(prompt)
            
            # Try to parse the response as JSON
            try:
                # If response is already a dict, use it directly
                if isinstance(response, dict):
                    parsed_response = response
                else:
                    # Otherwise try to parse it as JSON string
                    import json
                    parsed_response = json.loads(response)
                
                # Ensure the required keys exist
                if "sentiment" not in parsed_response or "score" not in parsed_response:
                    # Default to neutral if missing keys
                    return {"sentiment": "neutral", "score": 0.5}
                
                return parsed_response
                
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                print(f"Raw response: {response}")
                # Return neutral sentiment as fallback
                return {"sentiment": "neutral", "score": 0.5}
                
        except Exception as e:
            print(f"Error getting sentiment: {e}")
            # Return neutral sentiment as fallback
            return {"sentiment": "neutral", "score": 0.5}

    def on_trading_iteration(self):
        # Remove market open check as it's not available
        cash, last_price, quantity = self.position_sizing()
        
        # Skip if quantity is too small
        if quantity < 1:
            print(Fore.RED + f"Calculated quantity ({quantity}) too small to trade" + Fore.RESET)
            return
        
        # Get News, Sentiment and Probability
        news_data = self.get_sentiment()
        sentiment = news_data["sentiment"]
        probability = news_data["score"]

        if last_price is not None:
            if cash > (quantity * last_price):
                # Trade based on sentiment with confidence threshold
                if sentiment == "positive" and probability >= 0.7:
                    if self.last_trade == "sell":
                        self.sell_all()
                    order = self.create_order(
                        Asset(symbol=self.ticker, asset_type="stock"),
                        quantity,
                        "buy",
                        type="market"
                    )
                    print(Fore.LIGHTMAGENTA_EX + str(order) + Fore.RESET)
                    self.submit_order(order)
                    self.last_trade = "buy"
                if sentiment == "negative" and probability >= 0.7:
                    if self.last_trade == "buy":
                        self.sell_all()
                    order = self.create_order(
                        Asset(symbol=self.ticker, asset_type="stock"),
                        quantity,
                        "sell",
                        type="market"
                    )
                    print(Fore.LIGHTMAGENTA_EX + str(order) + Fore.RESET)
                    self.submit_order(order)
                    self.last_trade = "sell"


if __name__ == "__main__":
    # Backtest period
    start_date = datetime(2023, 1, 1)
    end_date = datetime.now()

    # Run backtest
    StockTrader.run_backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        benchmark_asset="SPY",  # S&P 500 as benchmark
        parameters={
            "cash_at_risk": 0.25, 
            "ticker": "AAPL", 
            "company_name": "Apple"
        }
    )
