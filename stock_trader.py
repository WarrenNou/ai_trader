from lumibot.entities import Asset
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime, timedelta
from colorama import Fore
import json
import numpy as np
import pandas as pd
import os
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from llmprompts import get_web_deets, prompt_template


class Response(BaseModel):
    sentiment: str
    score: float


class StockTrader(Strategy):
    def initialize(self, cash_at_risk: float = 0.2, ticker: str = "AAPL", company_name: str = "Apple", 
                  learning_rate: float = 0.01, discount_factor: float = 0.95, feedback_memory_size: int = 5):
        self.set_market("NYSE")
        self.sleeptime = "1D"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.ticker = ticker
        self.company_name = company_name
        self.llm = OllamaLLM(model="deepseek-r1:7b", format="json")
        
        # RL parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.feedback_memory_size = feedback_memory_size
        
        # Track trading history
        self.trade_history = []
        self.confidence_threshold = 0.7  # Initial threshold
        self.portfolio_values = []
        self.sentiment_accuracy = {"positive": 0.5, "negative": 0.5, "neutral": 0.5}
        self.trades_count = {"positive": 0, "negative": 0, "neutral": 0}
        self.successful_trades = {"positive": 0, "negative": 0, "neutral": 0}
        
        # Load historical feedback if available
        self.feedback_history = []
        self.feedback_file = f"{self.ticker}_feedback_history.csv"
        self.load_feedback_history()

    def load_feedback_history(self):
        """Load historical feedback data from CSV file if it exists"""
        if os.path.exists(self.feedback_file):
            try:
                df = pd.read_csv(self.feedback_file)
                self.feedback_history = df.to_dict('records')
                print(f"Loaded {len(self.feedback_history)} historical feedback entries")
                
                # Update accuracy and trade count metrics from history
                for entry in self.feedback_history:
                    sentiment = entry.get('sentiment')
                    reward = entry.get('reward', 0)
                    if sentiment in self.trades_count:
                        self.trades_count[sentiment] += 1
                        if reward > 0:
                            self.successful_trades[sentiment] += 1
                
                # Recalculate sentiment accuracies
                for sentiment in self.sentiment_accuracy:
                    if self.trades_count[sentiment] > 0:
                        self.sentiment_accuracy[sentiment] = (
                            self.successful_trades[sentiment] / self.trades_count[sentiment]
                        )
                        
                print(f"Initialized sentiment accuracy from history: {self.sentiment_accuracy}")
            except Exception as e:
                print(f"Error loading feedback history: {e}")

    def save_feedback_history(self):
        """Save feedback history to CSV file"""
        try:
            df = pd.DataFrame(self.feedback_history)
            df.to_csv(self.feedback_file, index=False)
            print(f"Saved {len(self.feedback_history)} feedback entries to {self.feedback_file}")
        except Exception as e:
            print(f"Error saving feedback history: {e}")

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
        day_prior = today - timedelta(days=3)
        return today.strftime("%Y-%m-%d"), day_prior.strftime("%Y-%m-%d")

    def get_recent_examples(self, sentiment_type=None, limit=3):
        """Get recent examples of correct/incorrect predictions for a sentiment"""
        examples = []
        
        # Filter by sentiment type if specified
        filtered_history = [entry for entry in self.feedback_history 
                         if sentiment_type is None or entry.get('sentiment') == sentiment_type]
        
        # Get successful examples (positive reward)
        successful = [entry for entry in filtered_history if entry.get('reward', 0) > 0]
        successful = sorted(successful, key=lambda x: x.get('date', ''), reverse=True)[:limit]
        
        # Get unsuccessful examples (negative reward)
        unsuccessful = [entry for entry in filtered_history if entry.get('reward', 0) < 0]
        unsuccessful = sorted(unsuccessful, key=lambda x: x.get('date', ''), reverse=True)[:limit]
        
        return successful, unsuccessful

    def create_feedback_context(self):
        """Create a context of recent examples for the LLM"""
        feedback_context = "Recent market reactions to predictions:\n\n"
        
        # Add examples for each sentiment type
        for sentiment in ["positive", "negative", "neutral"]:
            successful, unsuccessful = self.get_recent_examples(sentiment, limit=2)
            
            feedback_context += f"For {sentiment.upper()} sentiment predictions:\n"
            
            # Add successful examples
            if successful:
                feedback_context += "- Correct predictions:\n"
                for ex in successful:
                    action = ex.get('action', 'unknown')
                    price_before = ex.get('price', 0)
                    news_snippet = ex.get('news_snippet', '')[:100] + "..."
                    date = ex.get('date', '')
                    feedback_context += f"  * {date}: When predicting {sentiment} sentiment and taking {action} action, the market agreed. News: '{news_snippet}'\n"
            
            # Add unsuccessful examples
            if unsuccessful:
                feedback_context += "- Incorrect predictions:\n"
                for ex in unsuccessful:
                    action = ex.get('action', 'unknown')
                    price_before = ex.get('price', 0)
                    news_snippet = ex.get('news_snippet', '')[:100] + "..."
                    date = ex.get('date', '')
                    feedback_context += f"  * {date}: When predicting {sentiment} sentiment and taking {action} action, the market disagreed. News: '{news_snippet}'\n"
            
            feedback_context += "\n"
        
        return feedback_context

    def get_sentiment(self):
        """Get sentiment from news data with enhanced feedback"""
        today_date, prior_date = self.get_dates()
        news = get_web_deets(prior_date, today_date, self.company_name)
        
        # Store news for feedback history
        current_news_snippet = news[:500]  # Truncate for storage
        
        # Create base prompt
        base_prompt = prompt_template(news)
        
        # If we have feedback history, enhance the prompt with examples
        if len(self.feedback_history) >= 3:
            feedback_context = self.create_feedback_context()
            enhanced_prompt = (
                f"I'll provide news and examples of previous market predictions.\n\n"
                f"{feedback_context}\n"
                f"Current sentiment accuracy rates:\n"
                f"- Positive sentiment predictions are correct {self.sentiment_accuracy['positive']:.2%} of the time\n"
                f"- Negative sentiment predictions are correct {self.sentiment_accuracy['negative']:.2%} of the time\n"
                f"- Neutral sentiment predictions are correct {self.sentiment_accuracy['neutral']:.2%} of the time\n\n"
                f"Based on this historical performance and the following news, analyze the sentiment:\n\n"
                f"{news}\n\n"
                f"Respond with the sentiment ('positive', 'negative', or 'neutral') and a confidence score (0-1)."
            )
            prompt = enhanced_prompt
        else:
            prompt = base_prompt
        
        try:
            response = self.llm.invoke(prompt)
            
            try:
                if isinstance(response, dict):
                    parsed_response = response
                else:
                    parsed_response = json.loads(response)
                
                if "sentiment" not in parsed_response or "score" not in parsed_response:
                    result = {"sentiment": "neutral", "score": 0.5}
                else:
                    # Apply the learned confidence adjustments
                    sentiment = parsed_response["sentiment"]
                    if sentiment in self.sentiment_accuracy:
                        # Adjust the confidence score based on past performance
                        parsed_response["score"] *= self.sentiment_accuracy[sentiment]
                    
                    result = parsed_response
                
                # Store the current news snippet for future feedback
                result["news_snippet"] = current_news_snippet
                return result
                
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                print(f"Raw response: {response}")
                return {"sentiment": "neutral", "score": 0.5, "news_snippet": current_news_snippet}
                
        except Exception as e:
            print(f"Error getting sentiment: {e}")
            return {"sentiment": "neutral", "score": 0.5, "news_snippet": current_news_snippet}

    def calculate_reward(self, action, initial_price):
        """Calculate reward based on portfolio performance after a trade"""
        current_price = self.get_last_price(Asset(symbol=self.ticker, asset_type="stock"))
        if current_price is None or initial_price is None:
            return 0
            
        if action == "buy":
            # Positive reward if price went up after buying, negative if it went down
            return (current_price - initial_price) / initial_price
        elif action == "sell":
            # Positive reward if price went down after selling, negative if it went up
            return (initial_price - current_price) / initial_price
        else:  # hold
            # Small positive reward for holding during uptrend, small negative for downtrend
            return 0.1 * (current_price - initial_price) / initial_price

    def update_model(self, sentiment, action, reward, news_snippet):
        """Update the model based on the reward received"""
        # Track trade performance for this sentiment
        self.trades_count[sentiment] = self.trades_count.get(sentiment, 0) + 1
        
        if reward > 0:
            self.successful_trades[sentiment] = self.successful_trades.get(sentiment, 0) + 1
        
        # Update sentiment accuracy based on success rate
        if self.trades_count[sentiment] > 0:
            self.sentiment_accuracy[sentiment] = (
                self.successful_trades[sentiment] / self.trades_count[sentiment]
            )
        
        # Add to feedback history
        feedback_entry = {
            "date": self.get_datetime().strftime("%Y-%m-%d"),
            "sentiment": sentiment,
            "action": action,
            "reward": reward,
            "price": self.get_last_price(Asset(symbol=self.ticker, asset_type="stock")),
            "news_snippet": news_snippet
        }
        self.feedback_history.append(feedback_entry)
        
        # Keep history at reasonable size
        if len(self.feedback_history) > 500:
            self.feedback_history = self.feedback_history[-500:]
        
        # Save feedback history periodically
        if len(self.feedback_history) % 10 == 0:
            self.save_feedback_history()
        
        # Adjust confidence threshold based on overall accuracy
        total_trades = sum(self.trades_count.values())
        total_successful = sum(self.successful_trades.values())
        
        if total_trades > 10:  # Only adjust after collecting enough data
            overall_accuracy = total_successful / total_trades if total_trades > 0 else 0.5
            
            # Dynamically adjust confidence threshold
            if overall_accuracy < 0.4:
                # If accuracy is poor, increase threshold to be more selective
                self.confidence_threshold = min(0.9, self.confidence_threshold + self.learning_rate)
            elif overall_accuracy > 0.6:
                # If accuracy is good, we can be less strict with threshold
                self.confidence_threshold = max(0.5, self.confidence_threshold - self.learning_rate)
                
        print(f"Updated sentiment accuracy: {self.sentiment_accuracy}")
        print(f"Current confidence threshold: {self.confidence_threshold}")

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        
        # Skip if quantity is too small
        if quantity < 1:
            print(Fore.RED + f"Calculated quantity ({quantity}) too small to trade" + Fore.RESET)
            return
        
        # Store portfolio value for performance tracking
        portfolio_value = self.portfolio_value  # Changed from self.portfolio_value() to self.portfolio_value
        self.portfolio_values.append(portfolio_value)
        
        # Get sentiment analysis (now with potential feedback context)
        news_data = self.get_sentiment()
        sentiment = news_data["sentiment"]
        probability = news_data["score"]
        news_snippet = news_data.get("news_snippet", "")

        action = "hold"  # Default action
        
        if last_price is not None and cash > (quantity * last_price):
            # Use learned confidence threshold instead of fixed value
            if sentiment == "positive" and probability >= self.confidence_threshold:
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
                action = "buy"
                
            elif sentiment == "negative" and probability >= self.confidence_threshold:
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
                action = "sell"
        
        # Store trade information for reinforcement learning
        self.trade_history.append({
            "date": self.get_datetime(),
            "action": action,
            "sentiment": sentiment,
            "probability": probability,
            "price": last_price,
            "portfolio_value": portfolio_value,
            "news_snippet": news_snippet
        })
        
        # Calculate reward if we have at least 2 data points to compare
        if len(self.trade_history) >= 2:
            prev_trade = self.trade_history[-2]
            current_trade = self.trade_history[-1]
            
            if prev_trade["action"] != "hold":  # Only calculate reward for actual trades
                reward = self.calculate_reward(
                    prev_trade["action"], 
                    prev_trade["price"]
                )
                
                # Update model based on reward, now including news snippet
                self.update_model(
                    prev_trade["sentiment"],
                    prev_trade["action"],
                    reward,
                    prev_trade.get("news_snippet", "")
                )
                
                print(f"Action: {prev_trade['action']}, Sentiment: {prev_trade['sentiment']}")
                print(f"Reward: {reward:.4f}")

    def on_teardown(self):
        """Save the feedback history when the strategy is stopped"""
        self.save_feedback_history()
        print(f"Strategy teardown complete. Saved feedback history to {self.feedback_file}")


if __name__ == "__main__":
    # Backtest period
    start_date = datetime(2023, 1, 1)
    end_date = datetime.now()

    # Run backtest with reinforcement learning parameters
    StockTrader.run_backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        benchmark_asset="SPY",  # S&P 500 as benchmark
        parameters={
            "cash_at_risk": 0.25, 
            "ticker": "AAPL", 
            "company_name": "Apple",
            "learning_rate": 0.01,
            "discount_factor": 0.95,
            "feedback_memory_size": 5  # Number of examples to include in feedback
        }
    )
