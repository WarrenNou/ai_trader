
from lumibot.entities import Asset
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime, timedelta
from colorama import Fore
import json
import numpy as np
import pandas as pd
import os
from finbert_utils import estimate_sentiment
from lumibot.brokers import Alpaca
from lumibot.traders import Trader

class MultiStockTrader(Strategy):
    def initialize(self, cash_at_risk: float = 0.3, stocks_config=None):
        self.set_market("NYSE")
        self.sleeptime = "1M"  # Changed from "1D" to "1M" for 1 minute intervals
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        
        # Default stock configuration if none provided
        if stocks_config is None:
            self.stocks_config = get_top_stocks()
        else:
            self.stocks_config = stocks_config
            
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(stock["weight"] for stock in self.stocks_config)
        if total_weight != 1.0:
            for stock in self.stocks_config:
                stock["weight"] = stock["weight"] / total_weight
        
        # Track sentiment history
        self.sentiment_history = {}
        
        # Track portfolio allocation
        self.portfolio_allocation = {}
        
        # Load existing portfolio state
        self.load_portfolio_state()
        
        # Display initial portfolio state
        print(f"Initialized MultiStockTrader with {len(self.stocks_config)} stocks")
        print(f"Trading frequency: Every 1 minute")
        for stock in self.stocks_config:
            print(f"  - {stock['symbol']} ({stock['name']}): {stock['weight']*100:.1f}% allocation")
        
        # Test Alpaca connection
        self.test_alpaca_connection()

    def load_portfolio_state(self):
        """Load existing portfolio state to ensure continuity after restarts"""
        try:
            # Get current portfolio positions
            positions = self.get_positions()
            portfolio_value = self.portfolio_value
            cash = self.cash
            
            print(f"\n{Fore.CYAN}Loading existing portfolio state...{Fore.RESET}")
            
            # Handle None values properly
            if portfolio_value is None:
                print(f"{Fore.YELLOW}Warning: Portfolio value is None, using 0.00{Fore.RESET}")
                portfolio_value = 0.00
            else:
                print(f"{Fore.CYAN}Portfolio value: ${portfolio_value:.2f}{Fore.RESET}")
            
            if cash is None:
                print(f"{Fore.YELLOW}Warning: Cash value is None, using 0.00{Fore.RESET}")
                cash = 0.00
            else:
                print(f"{Fore.CYAN}Cash available: ${cash:.2f}{Fore.RESET}")
            
            print(f"{Fore.CYAN}Found {len(positions)} existing positions:{Fore.RESET}")
            
            # Store initial portfolio state
            self.initial_portfolio = {
                "value": portfolio_value,
                "cash": cash,
                "positions": {}
            }
            
            # Process each position
            for position in positions:
                symbol = position.symbol
                quantity = position.quantity
                
                # Get current price
                price = self.get_last_price(symbol)
                if price is None:
                    print(f"{Fore.YELLOW}Could not get price for {symbol}, using position value{Fore.RESET}")
                    # Handle case where quantity might be 0
                    if position.quantity > 0 and hasattr(position, 'market_value') and position.market_value is not None:
                        price = position.market_value / position.quantity
                    else:
                        price = 0.00
                
                position_value = quantity * price
                
                # Store in initial portfolio
                self.initial_portfolio["positions"][symbol] = {
                    "quantity": quantity,
                    "price": price,
                    "value": position_value
                }
                
                print(f"{Fore.GREEN}  - {symbol}: {quantity} shares at ${price:.2f} = ${position_value:.2f}{Fore.RESET}")
                
            # Save portfolio state to file for recovery
            self.save_portfolio_state()
            
        except Exception as e:
            print(f"{Fore.RED}Error loading portfolio state: {str(e)}{Fore.RESET}")
            import traceback
            traceback.print_exc()
            
            # Initialize empty state with safe defaults
            self.initial_portfolio = {
                "value": 0.00,
                "cash": 0.00,
                "positions": {}
            }

    def save_portfolio_state(self):
        """Save current portfolio state to file for recovery after crashes"""
        try:
            # Create state object with safe defaults for None values
            portfolio_value = self.portfolio_value if self.portfolio_value is not None else 0.00
            cash = self.cash if self.cash is not None else 0.00
            
            state = {
                "timestamp": self.get_datetime().strftime("%Y-%m-%d %H:%M:%S"),
                "portfolio_value": float(portfolio_value),
                "cash": float(cash),
                "positions": {}
            }
            
            # Get current positions
            positions = self.get_positions()
            for position in positions:
                symbol = position.symbol
                quantity = float(position.quantity)
                
                # Handle case where quantity might be 0 to avoid division by zero
                if quantity > 0 and hasattr(position, 'market_value') and position.market_value is not None:
                    price = float(position.market_value / quantity)
                    value = float(position.market_value)
                else:
                    price = 0.00
                    value = 0.00
                    
                state["positions"][symbol] = {
                    "quantity": quantity,
                    "price": price,
                    "value": value
                }
                
            # Save to file
            with open("portfolio_state.json", "w") as f:
                json.dump(state, f, indent=2)
                
            print(f"{Fore.GREEN}Portfolio state saved to portfolio_state.json{Fore.RESET}")
        
        except Exception as e:
            print(f"{Fore.RED}Error saving portfolio state: {str(e)}{Fore.RESET}")
            import traceback
            traceback.print_exc()

    def test_alpaca_connection(self):
        """Test connection to Alpaca API"""
        try:
            # Get account information
            account = self.broker.api.get_account()
            
            print(f"{Fore.GREEN}Successfully connected to Alpaca!{Fore.RESET}")
            print(f"Account ID: {account.id}")
            print(f"Account Status: {account.status}")
            print(f"Cash: ${float(account.cash):.2f}")
            print(f"Portfolio Value: ${float(account.portfolio_value):.2f}")
            return True
        except Exception as e:
            print(f"{Fore.RED}Error connecting to Alpaca: {str(e)}{Fore.RESET}")
            print(f"{Fore.RED}Please check your API credentials and internet connection{Fore.RESET}")
            return False

    def analyze_portfolio(self):
        """Analyze current portfolio allocation and compare to target weights"""
        portfolio_value = self.portfolio_value
        cash = self.get_cash()
        
        # Get all current positions
        positions = self.get_positions()
        
        # Calculate current allocation
        current_allocation = {}
        for position in positions:
            symbol = position.symbol
            # Calculate position value with better fallback logic
            if hasattr(position, 'market_value') and position.market_value:
                position_value = position.market_value
            else:
                # Get current price
                price = self.get_last_price(symbol)
                if price is None:
                    print(f"{Fore.RED}Could not get price for {symbol}, skipping allocation calculation{Fore.RESET}")
                    continue
                position_value = position.quantity * price
            
            # Calculate allocation percentage
            allocation = position_value / portfolio_value if portfolio_value > 0 else 0
            
            # Store in allocation dictionary
            current_allocation[symbol] = {
                'quantity': position.quantity,
                'value': position_value,
                'allocation': allocation,
                'price': position_value / position.quantity if position.quantity > 0 else 0
            }
        
        # Calculate allocation differences
        allocation_diff = {}
        for stock in self.stocks_config:
            symbol = stock['symbol']
            target = stock['weight']
            current = current_allocation.get(symbol, {}).get('allocation', 0)
            allocation_diff[symbol] = target - current
            
            # Print allocation status
            if symbol in current_allocation:
                status = "BALANCED" if abs(allocation_diff[symbol]) < 0.02 else "UNDERWEIGHT" if allocation_diff[symbol] > 0 else "OVERWEIGHT"
                color = Fore.GREEN if status == "BALANCED" else Fore.YELLOW if status == "UNDERWEIGHT" else Fore.RED
                print(f"{color}{symbol}: Target {target*100:.1f}%, Current {current*100:.1f}%, Diff {allocation_diff[symbol]*100:+.1f}% ({status}){Fore.RESET}")
            else:
                print(f"{Fore.YELLOW}{symbol}: Target {target*100:.1f}%, Current 0.0%, Diff {target*100:+.1f}% (NOT HELD){Fore.RESET}")
        
        return current_allocation, allocation_diff
    
    def rebalance_portfolio(self):
        """Rebalance portfolio to match target weights"""
        print(f"{Fore.CYAN}Starting portfolio rebalancing...{Fore.RESET}")
        
        # Get current allocation and differences
        current_allocation, allocation_diff = self.analyze_portfolio()
        
        # Store current allocation
        self.portfolio_allocation = current_allocation
        
        # Get portfolio value and cash
        portfolio_value = self.portfolio_value
        cash = self.get_cash()
        
        # First, sell overweight positions
        for stock in self.stocks_config:
            symbol = stock['symbol']
            diff = allocation_diff.get(symbol, 0)
            
            # If significantly overweight (>2%), sell to rebalance
            if diff < -0.02:
                position = self.get_position(symbol)
                if position is not None and position.quantity > 0:
                    # Calculate how much to sell
                    target_value = portfolio_value * stock['weight']
                    current_value = position.quantity * position.price
                    value_to_reduce = current_value - target_value
                    shares_to_sell = int(value_to_reduce / position.price)
                    
                    if shares_to_sell > 0:
                        order = self.create_order(
                            symbol,
                            shares_to_sell,
                            "sell",
                            order_type="market"
                        )
                        
                        print(f"{Fore.YELLOW}REBALANCE SELL for {symbol}: {shares_to_sell} shares at ${position.price:.2f} (OVERALLOCATED){Fore.RESET}")
                        self.submit_order(order)
                        
                        # Log the trade
                        self.log_trade(symbol, "rebalance_sell", shares_to_sell, position.price, "neutral", 0.5)
        
        # Next, buy underweight positions
        for stock in self.stocks_config:
            symbol = stock['symbol']
            diff = allocation_diff.get(symbol, 0)
            
            # If significantly underweight (>2%), buy to rebalance
            if diff > 0.02:
                # Get current price
                price = self.get_last_price(symbol)
                if price is None:
                    print(f"{Fore.RED}Could not get price for {symbol}, skipping rebalance{Fore.RESET}")
                    continue
                
                # Calculate quantity to buy
                target_value = portfolio_value * stock['weight']
                current_value = current_allocation.get(symbol, {}).get('value', 0)
                value_to_add = target_value - current_value
                max_investment = min(cash * self.cash_at_risk, value_to_add)
                quantity = max_investment / price
                
                if quantity > 0:
                    order = self.create_order(
                        symbol,
                        quantity,
                        "buy",
                        order_type="market"
                    )
                    
                    print(f"{Fore.YELLOW}REBALANCE BUY for {symbol}: {quantity:.2f} shares at ${price:.2f} (UNDERALLOCATED){Fore.RESET}")
                    self.submit_order(order)
                    
                    # Log the trade
                    self.log_trade(symbol, "rebalance_buy", quantity, price, "neutral", 0.5)
        
        print(f"{Fore.CYAN}Portfolio rebalancing complete.{Fore.RESET}")
    
    def position_sizing(self, symbol, price):
        """Calculate position size based on cash at risk and stock weight"""
        cash = self.get_cash()
        portfolio_value = self.portfolio_value
        
        # Find the weight for this symbol
        weight = next((stock["weight"] for stock in self.stocks_config if stock["symbol"] == symbol), 0)
        
        # Get current allocation if any
        current_allocation = self.portfolio_allocation.get(symbol, {}).get('allocation', 0)
        target_allocation = weight
        
        # Calculate allocation difference
        allocation_diff = target_allocation - current_allocation
        
        # If we're already overallocated, don't buy more
        if allocation_diff <= 0:
            print(f"{Fore.YELLOW}Already at or above target allocation for {symbol} (current: {current_allocation*100:.1f}%, target: {target_allocation*100:.1f}%){Fore.RESET}")
            return 0
        
        # Calculate quantity based on weight and cash at risk
        if price is None or price == 0:
            return 0
        
        # Calculate how much to invest to reach target allocation
        target_value = portfolio_value * target_allocation
        current_value = portfolio_value * current_allocation
        value_to_add = target_value - current_value
        
        # Limit by available cash and cash_at_risk
        max_investment = min(cash * self.cash_at_risk, value_to_add)
        
        # Calculate quantity
        quantity = max_investment / price
        
        print(f"{Fore.CYAN}Position sizing for {symbol}:{Fore.RESET}")
        print(f"  - Target allocation: {target_allocation*100:.1f}%")
        print(f"  - Current allocation: {current_allocation*100:.1f}%")
        print(f"  - Allocation difference: {allocation_diff*100:+.1f}%")
        print(f"  - Value to add: ${value_to_add:.2f}")
        print(f"  - Max investment: ${max_investment:.2f}")
        print(f"  - Calculated quantity: {quantity:.2f} shares")
        
        return int(quantity)  # Whole shares only
    
    def get_sentiment(self, symbol, company_name):
        """Get sentiment for a specific stock using Yahoo Finance news with improved analysis"""
        try:
            # Import required libraries
            import yfinance as yf
            import json
            
            # Get today's date and yesterday's date
            today = self.get_datetime()
            yesterday = today - timedelta(days=1)
            
            # Fetch news from Yahoo Finance
            ticker = yf.Ticker(symbol)
            news_data = ticker.news
            
            if not news_data:
                print(f"{Fore.YELLOW}No recent news found for {symbol}, using company name as fallback{Fore.RESET}")
                news = f"Latest news about {company_name} stock price and financial performance"
            else:
                # Combine news content from recent news
                news_texts = []
                print(f"{Fore.CYAN}Found {len(news_data)} news items for {symbol}{Fore.RESET}")
                
                # Debug the first news item's content structure
                if len(news_data) > 0:
                    first_item = news_data[0]
                    print(f"{Fore.CYAN}First news item for {symbol}:{Fore.RESET}")
                    if 'content' in first_item:
                        content_sample = str(first_item['content'])[:200]
                        print(f"{Fore.CYAN}Content sample: {content_sample}...{Fore.RESET}")
                        
                        # If content is a dictionary or complex object, print its structure
                        if isinstance(first_item['content'], (dict, list)):
                            print(f"{Fore.CYAN}Content structure: {json.dumps(first_item['content'], indent=2, default=str)[:300]}...{Fore.RESET}")
            
                for i, item in enumerate(news_data[:5]):  # Use up to 5 most recent news items
                    if 'content' in item:
                        content = item['content']
                        
                        # If content is a dictionary, try to extract text fields
                        if isinstance(content, dict):
                            if 'title' in content:
                                print(f"{Fore.CYAN}Title: {content['title']}{Fore.RESET}")
                                news_texts.append(content['title'])
                            
                            if 'summary' in content:
                                print(f"{Fore.CYAN}Summary: {content['summary'][:150]}...{Fore.RESET}")
                                news_texts.append(content['summary'])
                            
                            # Try other common field names
                            for field in ['description', 'body', 'text', 'headline']:
                                if field in content and isinstance(content[field], str):
                                    print(f"{Fore.CYAN}{field.capitalize()}: {content[field][:150]}...{Fore.RESET}")
                                    news_texts.append(content[field])
                        
                        # If content is a string, use it directly
                        elif isinstance(content, str) and len(content) > 20:
                            print(f"{Fore.CYAN}Content: {content[:150]}...{Fore.RESET}")
                            news_texts.append(content)
            
                # If we couldn't extract any news, use a fallback
                if not news_texts:
                    print(f"{Fore.YELLOW}Could not extract news content for {symbol}, using fallback{Fore.RESET}")
                    news = f"Latest news about {company_name} stock price and financial performance"
                else:
                    # Combine all news texts
                    news = " ".join(news_texts)
                    if len(news) > 1024:
                        news = news[:1024] + "..."
            
                # Print the combined news text that will be analyzed
                print(f"{Fore.CYAN}Combined news text for sentiment analysis ({len(news)} chars):{Fore.RESET}")
                print(f"{Fore.CYAN}{news[:150]}...{Fore.RESET}")
            
            # Analyze sentiment using FinBERT
            print(f"{Fore.YELLOW}Analyzing sentiment for {symbol}...{Fore.RESET}")
            probability, sentiment = estimate_sentiment([news])
            
            # Store in history
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            
            self.sentiment_history[symbol].append({
                "date": today.strftime("%Y-%m-%d %H:%M:%S"),
                "sentiment": sentiment,
                "probability": probability,
                "news_sample": news[:100] + "..." if len(news) > 100 else news
            })
            
            # Keep history to last 10 entries
            if len(self.sentiment_history[symbol]) > 10:
                self.sentiment_history[symbol] = self.sentiment_history[symbol][-10:]
                
            # Log the sentiment with more detail
            print(f"{Fore.YELLOW}{symbol} sentiment: {sentiment} (probability: {probability:.4f}){Fore.RESET}")
            
            # Log sentiment history
            print(f"{Fore.YELLOW}Recent sentiment history for {symbol}:{Fore.RESET}")
            for i, hist in enumerate(reversed(self.sentiment_history[symbol][-3:])):
                print(f"{Fore.YELLOW}  {hist['date']}: {hist['sentiment']} ({hist['probability']:.4f}){Fore.RESET}")
            
            return probability, sentiment
            
        except Exception as e:
            print(f"{Fore.RED}Error getting sentiment for {symbol}: {str(e)}{Fore.RESET}")
            import traceback
            traceback.print_exc()
            return 0.5, "neutral"
    
    def display_portfolio_summary(self):
        """Display a summary of the current portfolio with color-coded allocation status"""
        portfolio_value = self.portfolio_value
        cash = self.get_cash()
        
        # Get all positions
        positions = self.get_positions()
        
        # Calculate current allocation
        current_allocation = {}
        total_invested = 0
        
        for position in positions:
            symbol = position.symbol
            # Calculate position value
            if hasattr(position, 'market_value') and position.market_value:
                position_value = position.market_value
            else:
                price = self.get_last_price(symbol)
                if price is None:
                    continue
                position_value = position.quantity * price
            
            total_invested += position_value
            current_allocation[symbol] = {
                'quantity': position.quantity,
                'value': position_value,
                'allocation': position_value / portfolio_value if portfolio_value > 0 else 0,
                'price': position_value / position.quantity if position.quantity > 0 else 0
            }
        
        # Get target allocation
        target_allocation = {stock['symbol']: stock['weight'] for stock in self.stocks_config}
        
        # Print header
        print("\n" + "="*80)
        print(f"{Fore.CYAN}PORTFOLIO SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Fore.RESET}")
        print(f"Total Portfolio Value: ${portfolio_value:.2f}")
        print(f"Cash: ${cash:.2f} ({cash/portfolio_value*100:.1f}% of portfolio)")
        print(f"Invested: ${total_invested:.2f} ({total_invested/portfolio_value*100:.1f}% of portfolio)")
        print("="*80)
        
        # Print table header
        print(f"{'SYMBOL':<8} {'SHARES':<10} {'PRICE':<10} {'VALUE':<15} {'CURRENT %':<12} {'TARGET %':<12} {'DIFF':<10}")
        print("-"*80)
        
        # Print each position with color coding
        for symbol in sorted(target_allocation.keys()):
            target = target_allocation[symbol] * 100
            
            if symbol in current_allocation:
                pos = current_allocation[symbol]
                current = pos['allocation'] * 100
                diff = current - target
                
                # Color code based on allocation difference
                if abs(diff) < 2.0:  # Within 2% of target
                    color = Fore.GREEN
                elif abs(diff) < 5.0:  # Within 5% of target
                    color = Fore.YELLOW
                else:  # More than 5% off target
                    color = Fore.RED
                
                print(f"{symbol:<8} {pos['quantity']:<10.0f} ${pos['price']:<9.2f} ${pos['value']:<14.2f} {color}{current:<11.1f}%{Fore.RESET} {target:<11.1f}% {color}{diff:+<9.1f}%{Fore.RESET}")
            else:
                # Stock not in portfolio
                print(f"{symbol:<8} {'0':<10} {'$0.00':<10} {'$0.00':<15} {Fore.RED}{'0.0':<11}%{Fore.RESET} {target:<11.1f}% {Fore.RED}{-target:+<9.1f}%{Fore.RESET}")
        
        print("="*80)
        return current_allocation
    
    def on_trading_iteration(self):
        """Main trading logic - analyze each stock in our configuration"""
        # Display portfolio summary
        print("\n")
        print(f"{Fore.CYAN}Starting trading iteration at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Fore.RESET}")
        current_allocation = self.display_portfolio_summary()
        
        # Store current allocation
        self.portfolio_allocation = current_allocation
        
        # Save current portfolio state for recovery
        self.save_portfolio_state()
        
        # Calculate allocation differences
        allocation_diff = {}
        for stock in self.stocks_config:
            symbol = stock['symbol']
            target = stock['weight']
            current = current_allocation.get(symbol, {}).get('allocation', 0)
            allocation_diff[symbol] = target - current
        
        # Check if cash is negative or too low for trading
        cash = self.get_cash()
        if cash < 1000:  # If cash is below $1000, we need to free up some funds
            print(f"{Fore.YELLOW}Cash is low (${cash:.2f}). Looking for positions to trim to free up cash...{Fore.RESET}")
            
            # Get all positions, not just those in stocks_config
            all_positions = self.get_positions()
            
            # Find positions to sell - consider ALL positions, not just those in stocks_config
            positions_to_sell = []
            
            # First, identify positions that aren't in our target configuration at all
            target_symbols = [stock['symbol'] for stock in self.stocks_config]
            
            for position in all_positions:
                symbol = position.symbol
                
                # Skip positions with zero quantity
                if position.quantity <= 0:
                    continue
                    
                # Get current price
                price = self.get_last_price(symbol)
                if price is None:
                    continue
                    
                position_value = position.quantity * price
                
                # If position is not in our target configuration, consider selling it completely
                if symbol not in target_symbols:
                    positions_to_sell.append({
                        'symbol': symbol,
                        'position': position,
                        'price': price,
                        'value_to_reduce': position_value,
                        'priority': 1  # Highest priority - not in target config
                    })
                else:
                    # For positions in our target config, check if they're overallocated
                    target_weight = next((stock['weight'] for stock in self.stocks_config if stock['symbol'] == symbol), 0)
                    current_allocation_pct = position_value / self.portfolio_value if self.portfolio_value > 0 else 0
                    alloc_diff = current_allocation_pct - target_weight
                    
                    if alloc_diff > 0.02:  # More than 2% overallocated
                        target_value = self.portfolio_value * target_weight
                        value_to_reduce = position_value - target_value
                        
                        positions_to_sell.append({
                            'symbol': symbol,
                            'position': position,
                            'price': price,
                            'value_to_reduce': value_to_reduce,
                            'priority': 2  # Lower priority - in target config but overallocated
                        })
            
            # Sort by priority first, then by value_to_reduce (descending)
            positions_to_sell.sort(key=lambda x: (x['priority'], -x['value_to_reduce']))
            
            # Sell positions until we free up enough cash
            cash_needed = abs(cash) + 2000  # Target to free up negative cash plus $2000 buffer
            cash_freed = 0
            
            for item in positions_to_sell:
                if cash_freed >= cash_needed:
                    break
                    
                symbol = item['symbol']
                position = item['position']
                price = item['price']
                value_to_reduce = min(item['value_to_reduce'], cash_needed - cash_freed)
                
                # Calculate shares to sell (round down to ensure we don't oversell)
                shares_to_sell = int(value_to_reduce / price)
                
                # Ensure we're not selling more than we have
                shares_to_sell = min(shares_to_sell, position.quantity)
                
                if shares_to_sell > 0:
                    try:
                        order = self.create_order(
                            symbol,
                            shares_to_sell,
                            "sell",
                            order_type="market"
                        )
                        
                        reason = "NOT IN TARGET CONFIG" if item['priority'] == 1 else "OVERALLOCATED"
                        print(f"{Fore.RED}CASH MANAGEMENT SELL for {symbol}: {shares_to_sell} shares at ${price:.2f} ({reason}){Fore.RESET}")
                        self.submit_order(order)
                        
                        # Log the trade
                        self.log_trade(symbol, "cash_management_sell", shares_to_sell, price, "neutral", 0.5)
                        
                        # Update cash freed
                        cash_freed += shares_to_sell * price
                    
                    except Exception as e:
                        print(f"{Fore.RED}Error creating sell order for {symbol}: {str(e)}{Fore.RESET}")
                        import traceback
                        traceback.print_exc()
            
            if cash_freed > 0:
                print(f"{Fore.GREEN}Freed up approximately ${cash_freed:.2f} in cash by selling positions{Fore.RESET}")
            else:
                print(f"{Fore.YELLOW}Could not find suitable positions to sell to free up cash{Fore.RESET}")
        
        # Process each stock in our configuration
        for stock_config in self.stocks_config:
            symbol = stock_config["symbol"]
            company_name = stock_config["name"]
            
            # Get current price
            price = self.get_last_price(symbol)
            if price is None:
                print(f"{Fore.RED}Could not get price for {symbol}, skipping{Fore.RESET}")
                continue
                
            # Get sentiment
            probability, sentiment = self.get_sentiment(symbol, company_name)
            
            # Get current position if any
            position = self.get_position(symbol)
            
            # Trading logic - TRADITIONAL APPROACH: buy on positive sentiment
            if sentiment == "positive" and probability > 0.75:
                # Check if underallocated by >2%
                current_alloc = current_allocation.get(symbol, {}).get('allocation', 0)
                target_alloc = stock_config["weight"]
                alloc_diff = target_alloc - current_alloc
                
                if position is None or position.quantity == 0 or alloc_diff > 0.02:
                    # Check if we have enough cash
                    cash = self.get_cash()
                    if cash <= 0:
                        print(f"{Fore.YELLOW}Not enough cash (${cash:.2f}) to buy {symbol}, skipping{Fore.RESET}")
                        continue
                    
                    # Calculate quantity
                    quantity = self.position_sizing(symbol, price)
                    
                    if quantity > 0:
                        try:
                            order = self.create_order(
                                symbol,
                                quantity,
                                "buy",
                                order_type="market"
                            )
                            
                            print(f"{Fore.LIGHTMAGENTA_EX}BUY ORDER for {symbol}: {quantity} shares at ${price:.2f} (POSITIVE SENTIMENT){Fore.RESET}")
                            self.submit_order(order)
                            
                            # Log the trade
                            self.log_trade(symbol, "buy", quantity, price, sentiment, float(probability))
                        except Exception as e:
                            print(f"{Fore.RED}Error creating order for {symbol}: {str(e)}{Fore.RESET}")
                            import traceback
                            traceback.print_exc()
            
            # TRADITIONAL APPROACH: Sell on negative sentiment
            elif sentiment == "negative" and probability > 0.75:
                if position is not None and position.quantity > 0:
                    # Sell the position
                    order = self.create_order(
                        symbol,
                        position.quantity,
                        "sell",
                        order_type="market"
                    )
                    
                    print(f"{Fore.LIGHTMAGENTA_EX}SELL ORDER for {symbol}: {position.quantity} shares at ${price:.2f} (NEGATIVE SENTIMENT){Fore.RESET}")
                    self.submit_order(order)
                    
                    # Log the trade
                    self.log_trade(symbol, "sell", position.quantity, price, sentiment, float(probability))
            
            # Rebalancing logic - if significantly overallocated (>5%) and neutral sentiment, trim position
            elif sentiment == "neutral" and position is not None and position.quantity > 0:
                # Calculate allocation difference
                current_alloc = current_allocation.get(symbol, {}).get('allocation', 0)
                target_alloc = stock_config["weight"]
                alloc_diff = current_alloc - target_alloc
                
                if alloc_diff > 0.05:  # More than 5% overallocated
                    # Calculate how much to sell to get closer to target
                    target_value = self.portfolio_value * target_alloc
                    current_value = position.quantity * price
                    value_to_reduce = current_value - target_value
                    shares_to_sell = int(value_to_reduce / price)
                    
                    if shares_to_sell > 0 and shares_to_sell < position.quantity:
                        order = self.create_order(
                            symbol,
                            shares_to_sell,
                            "sell",
                            order_type="market"
                        )
                        
                        print(f"{Fore.YELLOW}REBALANCE SELL for {symbol}: {shares_to_sell} shares at ${price:.2f} (OVERALLOCATED){Fore.RESET}")
                        self.submit_order(order)
                        
                        # Log the trade
                        self.log_trade(symbol, "rebalance_sell", shares_to_sell, price, "neutral", 0.5)
            
            # Log portfolio status for this stock
            self.log_position_status(symbol, price)

    def log_position_status(self, symbol, price):
        """Log the current position status for a specific stock"""
        try:
            position = self.get_position(symbol)
            
            if position is None or position.quantity == 0:
                print(f"{Fore.BLUE}No position in {symbol}, current price: ${price:.2f}{Fore.RESET}")
            else:
                # Calculate profit/loss with better fallback logic
                # Try multiple possible attribute names used by different brokers
                if hasattr(position, 'average_entry_price') and position.average_entry_price:
                    entry_price = position.average_entry_price
                elif hasattr(position, 'price') and position.price:
                    entry_price = position.price
                elif hasattr(position, 'cost_basis') and position.cost_basis and position.quantity > 0:
                    entry_price = position.cost_basis / position.quantity
                elif hasattr(position, 'avg_entry_price') and position.avg_entry_price:
                    entry_price = position.avg_entry_price
                elif hasattr(position, 'entry_price') and position.entry_price:
                    entry_price = position.entry_price
                else:
                    # Last resort fallback - use current price
                    print(f"{Fore.YELLOW}Warning: Could not determine entry price for {symbol}, using current price{Fore.RESET}")
                    entry_price = price
                
                market_value = position.quantity * price
                profit_loss = ((price / entry_price) - 1) * 100  # Percentage
                
                # Color code based on profit/loss
                color = Fore.GREEN if profit_loss >= 0 else Fore.RED
                
                print(f"{color}Position in {symbol}: {position.quantity} shares at avg price ${entry_price:.2f}")
                print(f"{color}Current value: ${market_value:.2f}, P/L: {profit_loss:.2f}%{Fore.RESET}")
                
                # Add to sentiment history if not already there
                if symbol not in self.sentiment_history:
                    self.sentiment_history[symbol] = []
    
        except Exception as e:
            print(f"{Fore.RED}Error logging position status for {symbol}: {str(e)}{Fore.RESET}")
            print(f"{Fore.YELLOW}Position attributes: {dir(position)}{Fore.RESET}")  # Print available attributes
            import traceback
            traceback.print_exc()  # Print full traceback for debugging

    def log_trade(self, symbol, side, quantity, price, sentiment, probability):
        """Log trade details to a file for later analysis"""
        try:
            timestamp = self.get_datetime().strftime("%Y-%m-%d %H:%M:%S")
            
            # Convert any tensor values to Python native types
            if hasattr(probability, 'item'):
                probability = probability.item()  # Convert tensor to Python float
            
            # Create log entry
            log_entry = {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": side,
                "quantity": float(quantity),  # Ensure it's a native Python float
                "price": float(price),        # Ensure it's a native Python float
                "sentiment": sentiment,
                "probability": float(probability),  # Ensure it's a native Python float
                "total_value": float(quantity * price)  # Ensure it's a native Python float
            }
            
            # Print trade details
            color = Fore.GREEN if side == "buy" else Fore.RED
            print(f"{color}TRADE: {timestamp} - {side.upper()} {quantity} {symbol} @ ${price:.2f}")
            print(f"{color}REASON: {sentiment} sentiment ({probability:.4f} probability)")
            print(f"{color}VALUE: ${quantity * price:.2f}{Fore.RESET}")
            
            # Append to log file
            with open("trades_log.json", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            print(f"{Fore.RED}Error logging trade for {symbol}: {str(e)}{Fore.RESET}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging

    def save_stocks_config(self):
        """Save stocks configuration to file for dashboard access"""
        try:
            # Create a simplified version of stocks_config for the dashboard
            config_for_file = []
            for stock in self.stocks_config:
                config_for_file.append({
                    "symbol": stock["symbol"],
                    "name": stock.get("name", stock["symbol"]),
                    "weight": stock["weight"],
                    "sector": stock.get("sector", "Unknown")
                })
            
            # Save to file
            with open("stocks_config.json", "w") as f:
                json.dump(config_for_file, f, indent=2)
            
        except Exception as e:
            print(f"{Fore.RED}Error saving stocks config: {str(e)}{Fore.RESET}")

def get_top_stocks(sectors=None, count=15, min_market_cap=10e9):
    """Dynamically select top stocks based on multiple factors with better sector diversification"""
    try:
        import yfinance as yf
        import pandas as pd
        from scipy import stats
        import numpy as np
        # Set random seed for reproducibility
        np.random.seed(42)
        
        print(f"{Fore.CYAN}Dynamically selecting top stocks across multiple sectors...{Fore.RESET}")
        
        # Define sectors to include - expanded list for better diversification
        if sectors is None:
            sectors = [
                "Technology", "Consumer Cyclical", "Healthcare", 
                "Communication Services", "Financial Services",
                "Industrials", "Energy", "Consumer Defensive",
                "Utilities", "Real Estate", "Basic Materials"
            ]
        elif isinstance(sectors, str):
            sectors = [sectors]  # Convert single sector to list
            
        print(f"{Fore.CYAN}Targeting sectors: {', '.join(sectors)}{Fore.RESET}")
        
        # FORCE include Technology sector if not already included
        if "Technology" not in sectors:
            sectors.append("Technology")
            print(f"{Fore.GREEN}Added Technology sector to ensure tech representation{Fore.RESET}")
        
        # Define backup stocks by sector
        backup_stocks = {
            "Technology": ["AAPL", "MSFT", "NVDA", "ADBE", "CSCO", "ORCL", "IBM", "INTC", "AMD", "CRM"],
            "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "BKNG", "MAR"],
            "Healthcare": ["JNJ", "PFE", "UNH", "MRK", "ABT", "TMO", "ABBV", "LLY", "BMY", "AMGN"],
            "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "EA", "ATVI"],
            "Financial Services": ["JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "V", "MA", "BLK"],
            "Industrials": ["HON", "UNP", "UPS", "BA", "CAT", "GE", "MMM", "LMT", "RTX", "DE"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY", "KMI", "VLO"],
            "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "EL", "CL", "GIS"],
            "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "WEC"],
            "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "PSA", "O", "DLR", "WELL", "AVB", "EQR"],
            "Basic Materials": ["LIN", "APD", "ECL", "SHW", "NEM", "FCX", "NUE", "DOW", "DD", "VMC"]
        }
        
        # Get S&P 500 components with better error handling
        try:
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            print(f"{Fore.GREEN}Successfully retrieved S&P 500 component list ({len(sp500)} companies){Fore.RESET}")
        except Exception as e:
            print(f"{Fore.RED}Error fetching S&P 500 list: {str(e)}{Fore.RESET}")
            print(f"{Fore.YELLOW}Using backup method to get stock symbols...{Fore.RESET}")
            
            # Backup method - use predefined list of major stocks by sector
            symbols = []
            for sector in sectors:
                if sector in backup_stocks:
                    symbols.extend([(symbol, sector) for symbol in backup_stocks[sector]])
            
            # Create a dataframe similar to what we'd get from Wikipedia
            sp500 = pd.DataFrame(symbols, columns=['Symbol', 'GICS Sector'])
        
        # Filter by sector
        sp500 = sp500[sp500['GICS Sector'].isin(sectors)]
        if len(sp500) == 0:
            print(f"{Fore.RED}No stocks found in the specified sectors. Using default stocks.{Fore.RESET}")
            return get_default_stocks()
            
        print(f"{Fore.CYAN}Found {len(sp500)} stocks in the specified sectors{Fore.RESET}")
        
        # Collect data for analysis with sector diversification
        stock_data = {}
        
        # Process all stocks from all sectors to get comprehensive data
        all_symbols = sp500['Symbol'].unique().tolist()
        
        # Add backup stocks to ensure we have enough candidates
        for sector in sectors:
            if sector in backup_stocks:
                for symbol in backup_stocks[sector]:
                    if symbol not in all_symbols:
                        all_symbols.append(symbol)
        
        print(f"{Fore.CYAN}Analyzing {len(all_symbols)} potential stocks...{Fore.RESET}")
        
        # Process all symbols to get comprehensive data
        for symbol in all_symbols:
            try:
                # Get stock info
                stock = yf.Ticker(symbol)
                info = stock.info
                
                # Skip if market cap is too small or missing
                if 'marketCap' not in info or info['marketCap'] < min_market_cap:
                    continue
                
                # Get historical data for volatility calculation
                hist = stock.history(period="1y")
                if len(hist) < 100:  # Need at least 100 days of data
                    continue
                
                # Calculate metrics
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
                sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0
                
                # Calculate momentum (3-month return)
                momentum = hist['Close'].iloc[-1] / hist['Close'].iloc[-63] - 1 if len(hist) >= 63 else 0
                
                # Get EPS growth metrics if available
                eps_growth = info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0
                eps_forward = info.get('forwardEps', 0) if info.get('forwardEps') else 0
                eps_current = info.get('trailingEps', 0) if info.get('trailingEps') else 0
                
                # Calculate forward EPS growth if both values are available
                if eps_forward > 0 and eps_current > 0:
                    forward_eps_growth = (eps_forward / eps_current - 1) * 100
                else:
                    forward_eps_growth = 0
                
                # Use the maximum of reported growth and calculated growth
                eps_growth = max(eps_growth, forward_eps_growth)
                
                # Get sector information
                sector = None
                sector_row = sp500[sp500['Symbol'] == symbol]
                if not sector_row.empty:
                    sector = sector_row['GICS Sector'].iloc[0]
                else:
                    # Try to find sector in backup stocks
                    for s, symbols in backup_stocks.items():
                        if symbol in symbols:
                            sector = s
                            break
                
                if sector is None or sector not in sectors:
                    continue
                
                # Store data with additional metrics
                stock_data[symbol] = {
                    'symbol': symbol,
                    'name': info.get('shortName', symbol),
                    'market_cap': info.get('marketCap', 0),
                    'volatility': volatility,
                    'sharpe': sharpe,
                    'momentum': momentum,
                    'eps_growth': eps_growth,
                    'sector': sector
                }
                
                print(f"{Fore.GREEN}Added candidate: {symbol} ({sector}){Fore.RESET}")
                
            except Exception as e:
                print(f"{Fore.YELLOW}Error processing stock {symbol}: {str(e)}{Fore.RESET}")
                continue
        
        if len(stock_data) == 0:
            print(f"{Fore.RED}No stocks found meeting criteria. Using default stocks.{Fore.RESET}")
            return get_default_stocks()
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(stock_data).T
        
        # Create a composite score based on multiple factors
        # Normalize each factor to 0-1 range
        df['norm_volatility'] = 1 - (df['volatility'] - df['volatility'].min()) / (df['volatility'].max() - df['volatility'].min() + 1e-10)
        df['norm_eps_growth'] = (df['eps_growth'] - df['eps_growth'].min()) / (df['eps_growth'].max() - df['eps_growth'].min() + 1e-10)
        df['norm_momentum'] = (df['momentum'] - df['momentum'].min()) / (df['momentum'].max() - df['momentum'].min() + 1e-10)
        df['norm_sharpe'] = (df['sharpe'] - df['sharpe'].min()) / (df['sharpe'].max() - df['sharpe'].min() + 1e-10)
        
        # Composite score with weights - prioritize EPS growth and momentum
        df['score'] = (
            0.20 * df['norm_volatility'] +  # Lower volatility (20% weight)
            0.50 * df['norm_eps_growth'] +  # Higher EPS growth (50% weight)
            0.20 * df['norm_momentum'] +    # Recent momentum (20% weight)
            0.10 * df['norm_sharpe']        # Risk-adjusted returns (10% weight)
        )
        
        # Boost technology sector scores by 20%
        df.loc[df['sector'] == 'Technology', 'score'] *= 1.2
        
        # Ensure sector diversification
        selected_stocks = []
        
        # First, get top stocks from each sector
        for sector in sectors:
            sector_df = df[df['sector'] == sector].sort_values('score', ascending=False)
            
            # Take top 2 stocks from each sector (or fewer if not available)
            top_n = min(2, len(sector_df))
            if top_n > 0:
                selected_stocks.extend(sector_df.head(top_n).to_dict('records'))
        
        # If we need more stocks to reach count, take top scoring stocks overall
        if len(selected_stocks) < count:
            # Get symbols already selected
            selected_symbols = [stock['symbol'] for stock in selected_stocks]
            
            # Get remaining top stocks
            remaining_df = df[~df['symbol'].isin(selected_symbols)].sort_values('score', ascending=False)
            remaining_needed = min(count - len(selected_stocks), len(remaining_df))
            
            if remaining_needed > 0:
                selected_stocks.extend(remaining_df.head(remaining_needed).to_dict('records'))
        
        # If we have too many stocks, keep only the top scoring ones
        if len(selected_stocks) > count:
            selected_stocks = sorted(selected_stocks, key=lambda x: x['score'], reverse=True)[:count]
        
        # Calculate weights based on score
        total_score = sum(stock['score'] for stock in selected_stocks)
        for stock in selected_stocks:
            stock['weight'] = stock['score'] / total_score
        
        # Create final stock config list
        stocks_config = []
        for stock in selected_stocks:
            stocks_config.append({
                "symbol": stock['symbol'],
                "name": stock['name'],
                "weight": round(stock['weight'], 4),
                "volatility": round(stock['volatility'], 4),
                "sector": stock['sector'],
                "eps_growth": round(stock['eps_growth'], 2) if not pd.isna(stock['eps_growth']) else 0,
                "score": round(stock['score'], 4)
            })
        
        # Print selected stocks with sector information
        print(f"{Fore.GREEN}Selected {len(stocks_config)} stocks across {len(set(stock['sector'] for stock in stocks_config))} sectors:{Fore.RESET}")
        
        # Group by sector for better display
        by_sector = {}
        for stock in stocks_config:
            sector = stock['sector']
            if sector not in by_sector:
                by_sector[sector] = []
            by_sector[sector].append(stock)
            
        for sector, stocks in by_sector.items():
            print(f"{Fore.CYAN}{sector} ({len(stocks)} stocks):{Fore.RESET}")
            for stock in stocks:
                print(f"  - {stock['symbol']} ({stock['name']}): {stock['weight']*100:.1f}% weight, {stock['volatility']*100:.1f}% volatility, {stock['eps_growth']:.2f}% EPS growth")
        
        return stocks_config
        
    except Exception as e:
        print(f"{Fore.RED}Error in dynamic stock selection: {str(e)}{Fore.RESET}")
        import traceback
        traceback.print_exc()
        
        # Fallback to default stocks
        print(f"{Fore.YELLOW}Using default stock selection{Fore.RESET}")
        return get_default_stocks()

def get_default_stocks():
    """Return a default set of diversified stocks if dynamic selection fails"""
    return [
        {"symbol": "AAPL", "name": "Apple", "weight": 0.10, "volatility": 0.25, "sector": "Technology"},
        {"symbol": "MSFT", "name": "Microsoft", "weight": 0.10, "volatility": 0.20, "sector": "Technology"},
        {"symbol": "AMZN", "name": "Amazon", "weight": 0.10, "volatility": 0.30, "sector": "Consumer Cyclical"},
        {"symbol": "GOOGL", "name": "Alphabet", "weight": 0.10, "volatility": 0.25, "sector": "Communication Services"},
        {"symbol": "JNJ", "name": "Johnson & Johnson", "weight": 0.10, "volatility": 0.15, "sector": "Healthcare"},
        {"symbol": "JPM", "name": "JPMorgan Chase", "weight": 0.10, "volatility": 0.25, "sector": "Financial Services"},
        {"symbol": "PG", "name": "Procter & Gamble", "weight": 0.10, "volatility": 0.15, "sector": "Consumer Defensive"},
        {"symbol": "XOM", "name": "Exxon Mobil", "weight": 0.10, "volatility": 0.30, "sector": "Energy"},
        {"symbol": "NEE", "name": "NextEra Energy", "weight": 0.10, "volatility": 0.20, "sector": "Utilities"},
        {"symbol": "AMT", "name": "American Tower", "weight": 0.10, "volatility": 0.25, "sector": "Real Estate"}
    ]

def test_alpaca_connection_standalone():
    """Standalone function to test Alpaca connection"""
    from lumibot.brokers import Alpaca
    
    # Use the same credentials as in the main code
    API_KEY = "PKJ172HF65HRI6Z7R8AA"
    API_SECRET = "ZqlwAbdF8r5GaW5aJhcOHcwIMUPvnnwuZL3XGQTT"
    BASE_URL = "https://paper-api.alpaca.markets/v2"
    
    ALPACA_CREDS = {"API_KEY": API_KEY, "API_SECRET": API_SECRET, "PAPER": True}
    
    try:
        # Create a direct connection to Alpaca API
        from alpaca_trade_api import REST
        api = REST(
            key_id=API_KEY,
            secret_key=API_SECRET,
            base_url="https://paper-api.alpaca.markets"  # No trailing /v2
        )
        
        # Get account information
        account = api.get_account()
        
        print(f"{Fore.GREEN}Successfully connected to Alpaca!{Fore.RESET}")
        print(f"Account ID: {account.id}")
        print(f"Account Status: {account.status}")
        print(f"Cash: ${float(account.cash):.2f}")
        print(f"Portfolio Value: ${float(account.portfolio_value):.2f}")
        return True
    except Exception as e:
        print(f"{Fore.RED}Error connecting to Alpaca: {str(e)}{Fore.RESET}")
        print(f"{Fore.RED}Please check your API credentials and internet connection{Fore.RESET}")
        return False

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser for command line options
    parser = argparse.ArgumentParser(description='Multi-Stock Trading Bot with Sentiment Analysis')
    parser.add_argument('--test', action='store_true', help='Test Alpaca connection only')
    parser.add_argument('--risk', type=float, default=0.30, help='Percentage of cash at risk per trade (default: 0.30)')
    parser.add_argument('--backtest', action='store_true', help='Run in backtest mode instead of live trading')
    parser.add_argument('--stocks', type=int, default=15, help='Number of stocks to select (default: 15)')
    parser.add_argument('--sector', type=str, help='Specific sector to focus on (optional)')
    parser.add_argument('--min-cap', type=float, default=20, help='Minimum market cap in billions (default: 20)')
    args = parser.parse_args()
    
    # Test connection if requested
    if args.test:
        test_alpaca_connection_standalone()
        exit(0)

    # Dynamically select stocks
    sectors = [args.sector] if args.sector else None
    stocks_config = get_top_stocks(sectors=sectors, count=args.stocks, min_market_cap=args.min_cap * 1e9)
    
    if args.backtest:
        # Run backtest
        start_date = datetime(2023, 1, 1)
        end_date = datetime.now()
        
        MultiStockTrader.run_backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
            benchmark_asset="SPY",
            parameters={
                "cash_at_risk": args.risk,
                "stocks_config": stocks_config
            }
        )
    else:
        # Live trading with Alpaca
        API_KEY = "PKJ172HF65HRI6Z7R8AA"
        API_SECRET = "ZqlwAbdF8r5GaW5aJhcOHcwIMUPvnnwuZL3XGQTT"
        BASE_URL = "https://paper-api.alpaca.markets/v2"
        
        ALPACA_CREDS = {"API_KEY": API_KEY, "API_SECRET": API_SECRET, "PAPER": True}
        
        broker = Alpaca(ALPACA_CREDS)
        
        strategy = MultiStockTrader(
            name="MultiStockTrader",
            broker=broker,
            parameters={
                "cash_at_risk": args.risk,
                "stocks_config": stocks_config
            }
        )
        
        trader = Trader()
        trader.add_strategy(strategy)
        trader.run_all()
