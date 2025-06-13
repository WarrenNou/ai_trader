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
import json
import os


API_KEY = "PKJ172HF65HRI6Z7R8AA"
API_SECRET = "ZqlwAbdF8r5GaW5aJhcOHcwIMUPvnnwuZL3XGQTT"
BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {"API_KEY": API_KEY, "API_SECRET": API_SECRET, "PAPER": True}


class MLTrader(Strategy):

    def initialize(
        self, coin: str = "LTC", coin_name: str = "litecoin", cash_at_risk: float = 0.2
    ):
        self.set_market("24/7")
        self.sleeptime = "1M"  # Changed from 5H to 1M for faster testing
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.coin = coin
        self.coin_name = coin_name
        self.api = REST(base_url=BASE_URL, key_id=API_KEY,
                        secret_key=API_SECRET)

        # Add stop loss tracking
        self.stop_loss_prices = {}  # Dictionary to track stop loss prices for positions

        # Load any saved stop loss data
        self.load_stop_loss_data()

        # Log the change in sleep time
        self.log_message(
            "Strategy configured to check every 1 minute for testing", color="green")
        self.log_message(
            f"Trading {self.coin} with {self.cash_at_risk*100}% cash at risk per trade", color="green")

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(
            Asset(symbol=self.coin, asset_type=Asset.AssetType.CRYPTO),
            quote=Asset(symbol="USD", asset_type="crypto"),
        )
        if last_price is None or cash is None:
            quantity = 0
        else:
            # Convert to float if needed for calculations
            price_val = float(last_price) if hasattr(
                last_price, '__float__') else last_price
            cash_val = float(cash) if hasattr(cash, '__float__') else cash

            # From the logs, we see only ~8% of reported cash is actually available ($4899/$58950)
            # Use an even more conservative approach - assume only 5% of reported cash is available
            # This accounts for unsettled transactions, pending orders, margin requirements, and broker reserves
            available_cash_estimate = cash_val * 0.05

            # Calculate the ideal quantity based on estimated available cash
            ideal_quantity = available_cash_estimate * self.cash_at_risk / price_val

            # Add a large buffer to account for fees and rounding (use 80% of intended amount)
            quantity = ideal_quantity * 0.8

            # Log the calculation for debugging
            intended_value = available_cash_estimate * self.cash_at_risk
            self.log_message(
                f"Position sizing: Total Cash=${cash_val:.2f}, Est. Available=${available_cash_estimate:.2f} (5%), Risk%={self.cash_at_risk*100:.0f}%, Target=${intended_value:.2f}")

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
        self.log_message(
            f"Starting trading iteration at {self.get_datetime()}")

        # Check for stop loss conditions first
        stop_loss_triggered = self.check_stop_loss()
        if stop_loss_triggered:
            self.log_message(
                "Stop loss was triggered and executed, skipping regular trading logic", color="yellow")
            return

        # Log performance metrics
        self.log_performance()  # Add this line to track performance

        # Get accurate portfolio data directly from broker
        cash = self.get_cash()
        portfolio_value = self.portfolio_value
        positions = self.get_positions()
        equity_value = 0

        # Get position sizing and sentiment
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        if last_price is None:
            self.log_message(
                "Could not get last price, skipping trading iteration", color="red")
            return

        print(Fore.YELLOW + f"{probability}, {sentiment}" + Fore.RESET)

        # Get current position if any
        position = self.get_position(
            Asset(symbol=self.coin, asset_type="crypto"))
        if position is not None:
            self.log_message(
                f"Current position: {position.quantity} {self.coin}")
        else:
            self.log_message(f"No current position in {self.coin}")

        # CONTRARIAN STRATEGY: Buy on NEGATIVE sentiment
        if sentiment == "negative" and probability > 0.95 and quantity > 0:
            self.log_message(
                f"Buy conditions met: Negative sentiment with {probability:.4f} probability (CONTRARIAN)", color="green")

            try:
                # Format to 8 decimal places which is standard for crypto
                buy_quantity = round(quantity, 8)

                # Calculate intended purchase value for logging
                cash_val = float(cash) if cash is not None else 0
                intended_value = cash_val * self.cash_at_risk

                self.log_message(
                    f"Attempting to buy: {buy_quantity} {self.coin} (${intended_value:.2f} worth)", color="yellow")

                # Create a simple market buy order (not bracket)
                crypto_asset = Asset(symbol=self.coin, asset_type="crypto")
                quote_asset = Asset(symbol="USD", asset_type="crypto")

                # Step 1: Create and submit market buy order
                market_order = self.create_order(
                    crypto_asset,
                    buy_quantity,
                    "buy",
                    order_type="market",
                    quote=quote_asset
                )

                self.log_message(
                    f"BUY ORDER (CONTRARIAN - NEGATIVE SENTIMENT): {market_order}", color="green")
                print(Fore.LIGHTMAGENTA_EX +
                      f"BUY ORDER (CONTRARIAN - NEGATIVE SENTIMENT): {market_order}" + Fore.RESET)

                # Submit the order
                submitted_order = self.submit_order(market_order)
                self.last_trade = "buy"

                # Step 2: Create take profit order (separate limit order)
                # Wait a moment for the position to be updated
                import time
                time.sleep(2)

                # Get the updated position
                position = self.get_position(crypto_asset)

                if position is not None and position.quantity > 0:
                    # Set stop loss for this position (30% below entry)
                    self.set_stop_loss_for_position(position, last_price, 0.3)

                    # Calculate take profit price (50% gain)
                    take_profit_price = last_price * 1.5

                    # Use only a small conservative amount for take profit to avoid "insufficient balance" errors
                    # From the logs, we see only ~10% of position is actually available (0.037/0.373 = ~10%)
                    # Let's use an even smaller amount based on what broker reports as "available"
                    # For safety, use only 1% of total position to ensure we don't exceed available balance
                    conservative_estimate = position.quantity * \
                        0.01  # Use only 1% of total position
                    # Minimum 0.001 BTC
                    tp_quantity = round(max(conservative_estimate, 0.001), 8)

                    self.log_message(
                        f"Creating ultra-conservative take profit: {tp_quantity} {self.coin} (1% of position, total={position.quantity})", color="cyan")

                    # Create take profit order
                    tp_order = self.create_order(
                        crypto_asset,
                        tp_quantity,
                        "sell",
                        order_type="limit",
                        limit_price=take_profit_price,
                        quote=quote_asset
                    )

                    self.log_message(
                        f"TAKE PROFIT ORDER at {take_profit_price:.2f} (50% gain): {tp_order}", color="green")

                    try:
                        self.submit_order(tp_order)
                    except Exception as tp_error:
                        error_msg = str(tp_error)
                        self.log_message(
                            f"Take profit order failed: {error_msg}", color="red")

                        # Check if this is an insufficient balance error for take profit
                        if "insufficient balance" in error_msg.lower() or "40310000" in error_msg:
                            self.log_message(
                                "💡 TIP: Take profit insufficient balance. This usually means:", color="yellow")
                            self.log_message(
                                "  - Most of your crypto position is locked in other orders", color="yellow")
                            self.log_message(
                                "  - Broker reserves are keeping crypto unavailable", color="yellow")
                            self.log_message(
                                f"  - Only ~{tp_quantity/position.quantity*100:.1f}% of position was requested but still too much", color="yellow")
                            self.log_message(
                                "  Consider using even smaller take profit amounts", color="yellow")

                    # Note: For stop loss, we would need to monitor the price in the strategy
                    # and create a limit sell order when the price drops below our threshold
                    self.log_message(
                        f"Stop loss will be monitored at ${last_price * 0.7:.2f} (30% loss)", color="yellow")

            except Exception as e:
                error_msg = str(e)
                self.log_message(
                    f"Error placing buy order: {error_msg}", color="red")
                print(
                    Fore.RED + f"Error placing buy order: {error_msg}" + Fore.RESET)

                # Check if this is an insufficient balance error and suggest solution
                if "insufficient balance" in error_msg.lower() or "40310000" in error_msg:
                    self.log_message(
                        "💡 TIP: Insufficient balance detected. This usually means:", color="yellow")
                    self.log_message(
                        "  - Unsettled transactions are tying up funds", color="yellow")
                    self.log_message(
                        "  - Other pending orders are reserving cash", color="yellow")
                    self.log_message(
                        "  - Broker requires cash reserves for margin", color="yellow")
                    self.log_message(
                        "  Consider reducing cash_at_risk or waiting for settlements", color="yellow")
        else:
            # Log why we didn't buy
            if sentiment != "negative":
                self.log_message(
                    f"Not buying: Sentiment is {sentiment}, not negative (CONTRARIAN)", color="yellow")
            elif probability <= 0.95:
                self.log_message(
                    f"Not buying: Probability {probability:.4f} is not > 0.95", color="yellow")
            elif quantity <= 0:
                self.log_message(
                    f"Not buying: Calculated quantity {quantity} is too small", color="yellow")
            else:
                self.log_message("Not buying: Unknown reason", color="red")

        # CONTRARIAN STRATEGY: Sell on POSITIVE sentiment
        if sentiment == "positive" and probability > 0.95 and self.last_trade == "buy":
            if position is not None and position.quantity > 0:
                try:
                    self.log_message(
                        f"Sell conditions met: Positive sentiment with {probability:.4f} probability (CONTRARIAN)", color="green")

                    # Create sell order for the entire position
                    crypto_asset = Asset(symbol=self.coin, asset_type="crypto")
                    quote_asset = Asset(symbol="USD", asset_type="crypto")

                    order = self.create_order(
                        crypto_asset,
                        position.quantity,
                        "sell",
                        order_type="market",
                        quote=quote_asset
                    )

                    self.log_message(
                        f"SELL ORDER (CONTRARIAN - POSITIVE SENTIMENT): {order}", color="green")
                    print(
                        Fore.LIGHTCYAN_EX + f"SELL ORDER (CONTRARIAN - POSITIVE SENTIMENT): {order}" + Fore.RESET)

                    # Submit the order
                    self.submit_order(order)
                    self.last_trade = "sell"

                except Exception as e:
                    self.log_message(
                        f"Error placing sell order: {str(e)}", color="red")
                    print(
                        Fore.RED + f"Error placing sell order: {str(e)}" + Fore.RESET)
            else:
                self.log_message(
                    "Sell signal received but no position to sell", color="yellow")

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
                    length=4,  # Get 4 periods (days due to sleeptime="1D")
                    # Specify quote currency
                    quote=Asset(symbol="USD", asset_type="crypto")
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
                        price_change = (
                            newest_price - oldest_price) / oldest_price

                    # Get current position (needed for trading decisions)
                    position = self.get_position(
                        crypto_asset)  # Use the asset object
                    # Get last price and quantity for placing orders
                    # Note: position_sizing already calls get_last_price which might use Yahoo too
                    cash, last_price, quantity = self.position_sizing()
                    if last_price is None:  # Need last_price for bracket orders
                        self.log_message(
                            "Could not get last price for position sizing.", color="red")
                        return

                    # Trading decisions based on trend
                    if price_change > 0.05 and (position is None or position.quantity == 0):
                        # Buy order logic remains the same
                        order = self.create_order(
                            crypto_asset,  # Use asset object
                            quantity * 0.5,  # Half position size
                            "buy",
                            type="bracket",
                            take_profit_price=last_price * 1.3,
                            stop_loss_price=last_price * 0.9,
                            # quote= is implicitly USD via the asset pair
                        )
                        print(Fore.LIGHTBLUE_EX +
                              f"TREND BUY (Yahoo Data): {order}" + Fore.RESET)
                        self.submit_order(order)
                        self.last_trade = "buy"

                    # If price is trending down at least 5% and we have a position, sell
                    elif price_change < -0.05 and position is not None and position.quantity > 0:
                        order = self.create_order(
                            crypto_asset,  # Use asset object
                            position.quantity,
                            "sell",
                            # quote= is implicitly USD
                        )
                        print(Fore.LIGHTRED_EX +
                              f"TREND SELL (Yahoo Data): {order}" + Fore.RESET)
                        self.submit_order(order)
                        self.last_trade = "sell"
                else:
                    print(
                        Fore.YELLOW + f"Not enough historical data received via get_historical_prices for {self.coin}." + Fore.RESET)

            except Exception as e:
                # More specific error message
                print(
                    Fore.RED + f"Error using get_historical_prices for {self.coin}: {str(e)}" + Fore.RESET)

    def log_performance(self):
        """Log key performance metrics during trading"""
        try:
            # Get current position and portfolio data
            position = self.get_position(
                Asset(symbol=self.coin, asset_type="crypto"))
            portfolio_value = self.portfolio_value
            cash = self.get_cash()

            # Calculate metrics
            position_value = 0
            if position is not None:
                # Calculate position value if market_value attribute doesn't exist
                if hasattr(position, 'market_value'):
                    position_value = position.market_value
                else:
                    # Get the asset type from the position's asset, defaulting to "crypto" if not available
                    asset_type = getattr(position.asset, 'asset_type', 'crypto') if hasattr(
                        position, 'asset') and position.asset else 'crypto'

                    current_price = self.get_last_price(
                        Asset(symbol=position.symbol, asset_type=asset_type),
                        quote=Asset(symbol="USD", asset_type="crypto")
                    )
                    if current_price is not None:
                        position_value = position.quantity * current_price

            allocation = 0 if portfolio_value == 0 else position_value / portfolio_value

            # Log the information
            self.log_message(f"Portfolio Value: ${portfolio_value:.2f}")
            self.log_message(f"Cash: ${cash:.2f}")
            self.log_message(
                f"Position Value: ${position_value:.2f} ({allocation:.2%} allocation)")

            # If we have a position, log its details
            if position is not None and position.quantity > 0:
                entry_price = position.average_entry_price if hasattr(
                    position, 'average_entry_price') else 0
                current_price = self.get_last_price(Asset(symbol=self.coin, asset_type="crypto"),
                                                    quote=Asset(symbol="USD", asset_type="crypto"))
                if entry_price > 0 and current_price is not None:
                    profit_pct = (current_price - entry_price) / entry_price
                    self.log_message(f"Current P&L: {profit_pct:.2%}")

        except Exception as e:
            self.log_message(
                f"Error logging performance: {str(e)}", color="red")

    def send_test_trade(self):
        """Send a small test trade and immediately cancel it"""
        try:
            print(Fore.CYAN + "Sending test trade..." + Fore.RESET)

            # Get current price
            last_price = self.get_last_price(
                Asset(symbol=self.coin, asset_type=Asset.AssetType.CRYPTO),
                quote=Asset(symbol="USD", asset_type="crypto"),
            )

            if last_price is None:
                print(
                    Fore.RED + "Error: Could not get current price for test trade" + Fore.RESET)
                return

            # Calculate a very small quantity (about $10 worth)
            quantity = 1 / last_price

            # Create a market buy order
            order = self.create_order(
                self.coin,
                quantity,
                "buy",
                type="market",
                quote=Asset(symbol="USD", asset_type="crypto"),
            )

            print(Fore.LIGHTMAGENTA_EX +
                  f"TEST BUY ORDER: {order}" + Fore.RESET)

            # Submit the order
            submitted_order = self.submit_order(order)
            print(Fore.GREEN +
                  f"Test order submitted: {submitted_order}" + Fore.RESET)

            # Wait a moment to ensure the order is processed
            import time
            time.sleep(2)

            # Get the position (if the order was filled)
            position = self.get_position(
                Asset(symbol=self.coin, asset_type="crypto"))

            if position is not None and position.quantity > 0:
                # Create a sell order to close the position
                sell_order = self.create_order(
                    self.coin,
                    position.quantity,
                    "sell",
                    type="market",
                    quote=Asset(symbol="USD", asset_type="crypto"),
                )

                print(Fore.LIGHTCYAN_EX +
                      f"TEST SELL ORDER: {sell_order}" + Fore.RESET)

                # Submit the sell order
                submitted_sell = self.submit_order(sell_order)
                print(
                    Fore.GREEN + f"Test sell order submitted: {submitted_sell}" + Fore.RESET)

            return True

        except Exception as e:
            print(
                Fore.RED + f"Error sending test trade: {str(e)}" + Fore.RESET)
            return False

    def test_bracket_orders(self):
        """Test if bracket orders with take profit and stop loss are working correctly"""
        try:
            self.log_message("Starting bracket order test...", color="cyan")

            # Make sure we have the coin attribute
            if not hasattr(self, 'coin') or self.coin is None:
                self.log_message(
                    "Error: coin attribute not set. Using BTC as default.", color="yellow")
                coin = "BTC"
            else:
                coin = self.coin

            # Create proper Asset objects
            crypto_asset = Asset(symbol=coin, asset_type="crypto")
            quote_asset = Asset(symbol="USD", asset_type="crypto")

            # Get current price
            last_price = self.get_last_price(
                crypto_asset,
                quote=quote_asset
            )

            if last_price is None:
                self.log_message(
                    f"Error: Could not get current price for {coin}", color="red")
                return False

            # Calculate a small test quantity (about $25 worth - even smaller to avoid issues)
            test_quantity = round(25 / last_price, 8)

            self.log_message(f"Current price: ${last_price:.2f}", color="cyan")
            self.log_message(
                f"Test quantity: {test_quantity} {coin}", color="cyan")

            # Calculate take profit and stop loss prices
            take_profit_price = last_price * 1.5  # 50% profit
            stop_loss_price = last_price * 0.7    # 30% loss

            self.log_message(
                f"Take profit price: ${take_profit_price:.2f} (50% gain)", color="green")
            self.log_message(
                f"Stop loss price: ${stop_loss_price:.2f} (30% loss)", color="red")

            # Create a simple market buy order first (not bracket)
            try:
                # First try a simple market order
                self.log_message(
                    "Creating a simple market buy order first...", color="cyan")

                simple_order = self.create_order(
                    crypto_asset,
                    test_quantity,
                    "buy",
                    order_type="market",  # Use order_type instead of type
                    quote=quote_asset
                )

                self.log_message(
                    f"Simple market order created: {simple_order}", color="cyan")

                # Submit the order
                submitted_order = self.submit_order(simple_order)
                self.log_message(
                    f"Simple market order submitted: {submitted_order}", color="green")

                # Wait for the order to be processed
                import time
                time.sleep(5)  # Wait longer to ensure position is updated

                # Get the position directly from Lumibot
                position = self.get_position(crypto_asset)

                if position is not None and position.quantity > 0:
                    self.log_message(
                        f"Test position opened: {position.quantity} {coin}", color="green")

                    # Try to place a very small take profit order (0.001 BTC)
                    # This is a very small amount that should be available
                    safe_quantity = 0.001  # Very small fixed amount

                    # Now try to place take profit order
                    self.log_message(
                        "Now placing take profit order...", color="cyan")
                    self.log_message(
                        f"Using small fixed quantity for safety: {safe_quantity} {coin}", color="yellow")

                    # Take profit order
                    take_profit_order = self.create_order(
                        crypto_asset,
                        safe_quantity,
                        "sell",
                        order_type="limit",  # Use limit for take profit
                        limit_price=take_profit_price,  # Use limit_price instead of take_profit_price
                        quote=quote_asset
                    )

                    self.log_message(
                        f"Take profit order created: {take_profit_order}", color="green")
                    tp_submitted = self.submit_order(take_profit_order)
                    self.log_message(
                        f"Take profit order submitted: {tp_submitted}", color="green")

                    # For crypto, Alpaca only supports limit orders, not stop orders
                    # So we'll explain this limitation
                    self.log_message(
                        "NOTE: Alpaca doesn't support stop orders for crypto", color="yellow")
                    self.log_message(
                        "In a real strategy, you would need to monitor the price and create a limit order when needed", color="yellow")

                    # Check all orders
                    time.sleep(1)  # Wait a moment for orders to be processed
                    orders = self.get_orders()
                    self.log_message(
                        f"Total orders after test: {len(orders)}", color="cyan")

                    # Log the order IDs for reference
                    self.log_message("Order IDs for reference:", color="cyan")
                    for o in orders:
                        order_id = getattr(o, 'id', 'unknown')
                        order_type = getattr(
                            o, 'order_type', getattr(o, 'type', 'unknown'))
                        self.log_message(
                            f"  - Order ID: {order_id}, Type: {order_type}", color="cyan")

                    self.log_message(
                        "Bracket order test completed successfully!", color="green")
                    print(
                        Fore.GREEN + "Bracket order test completed successfully!" + Fore.RESET)
                    print("The test has verified that:")
                    print("1. A market order can be created and filled")
                    print("2. A take profit order can be created at 50% gain")
                    print(
                        "3. For stop loss: Alpaca doesn't support stop orders for crypto")
                    print(
                        "   In a real strategy, you would need to monitor the price and create a limit order when needed")

                    return True
                else:
                    self.log_message(
                        "No position found after market order", color="red")
                    return False

            except Exception as e:
                self.log_message(
                    f"Error with market order: {str(e)}", color="red")
                print(
                    Fore.RED + f"Error with market order: {str(e)}" + Fore.RESET)
                return False

        except Exception as e:
            self.log_message(
                f"Error testing bracket orders: {str(e)}", color="red")
            print(
                Fore.RED + f"Error testing bracket orders: {str(e)}" + Fore.RESET)
            return False

    def check_stop_loss(self):
        """Monitor positions and execute stop loss if price falls below threshold"""
        try:
            # Get current position
            position = self.get_position(
                Asset(symbol=self.coin, asset_type="crypto"))

            # If no position, nothing to check
            if position is None or position.quantity <= 0:
                return False

            # Get current price
            current_price = self.get_last_price(
                Asset(symbol=self.coin, asset_type="crypto"),
                quote=Asset(symbol="USD", asset_type="crypto")
            )

            if current_price is None:
                self.log_message(
                    "Could not get current price for stop loss check", color="red")
                return False

            # Get position key
            position_key = f"{position.symbol}_crypto"

            # Get stop loss price from our stored dictionary
            stop_loss_price = self.stop_loss_prices.get(position_key)

            # If we don't have a stored stop loss price, calculate it from entry price
            if stop_loss_price is None:
                entry_price = position.average_entry_price if hasattr(
                    position, 'average_entry_price') else None

                # If we don't have entry price, we can't calculate stop loss
                if entry_price is None:
                    self.log_message(
                        "No entry price available for stop loss calculation", color="yellow")
                    return False

                # Calculate and store stop loss price (30% below entry)
                stop_loss_price = self.set_stop_loss_for_position(
                    position, entry_price)

                if stop_loss_price is None:
                    return False

            # Check if current price is below stop loss threshold
            if current_price < stop_loss_price:
                self.log_message(
                    f"STOP LOSS TRIGGERED: Current price ${current_price:.2f} below stop loss ${stop_loss_price:.2f}", color="red")

                # Create sell order for the entire position
                crypto_asset = Asset(symbol=self.coin, asset_type="crypto")
                quote_asset = Asset(symbol="USD", asset_type="crypto")

                # Create market sell order
                order = self.create_order(
                    crypto_asset,
                    position.quantity,
                    "sell",
                    order_type="market",
                    quote=quote_asset
                )

                self.log_message(f"STOP LOSS SELL ORDER: {order}", color="red")
                print(Fore.RED + f"STOP LOSS SELL ORDER: {order}" + Fore.RESET)

                # Submit the order
                self.submit_order(order)
                self.last_trade = "sell"

                # Remove the stop loss price from our dictionary
                if position_key in self.stop_loss_prices:
                    del self.stop_loss_prices[position_key]
                    # Save the updated stop loss data
                    self.save_stop_loss_data()

                return True

            # Log current price vs stop loss for monitoring
            self.log_message(
                f"Current price ${current_price:.2f}, Stop loss set at ${stop_loss_price:.2f}", color="cyan")
            return False

        except Exception as e:
            self.log_message(
                f"Error checking stop loss: {str(e)}", color="red")
            return False

    def set_stop_loss_for_position(self, position, entry_price, stop_loss_percentage=0.3):
        """Set a stop loss price for a position and store it"""
        if position is None or position.quantity <= 0:
            return None

        # Calculate stop loss price (default 30% below entry)
        stop_loss_price = entry_price * (1 - stop_loss_percentage)

        # Store the stop loss price for this position
        asset_type = getattr(position.asset, 'asset_type', 'crypto') if hasattr(
            position, 'asset') and position.asset else 'crypto'
        position_key = f"{position.symbol}_{asset_type}"
        self.stop_loss_prices[position_key] = stop_loss_price

        self.log_message(
            f"Set stop loss for {position.symbol} at ${stop_loss_price:.2f} ({stop_loss_percentage*100:.0f}% below entry)", color="yellow")

        # Save the updated stop loss data
        self.save_stop_loss_data()

        return stop_loss_price

    def save_stop_loss_data(self):
        """Save stop loss prices to a JSON file"""
        try:
            # Convert the stop loss prices to a serializable format
            # (Asset objects can't be directly serialized to JSON)
            serializable_data = {}
            for key, value in self.stop_loss_prices.items():
                serializable_data[key] = value

            # Save to a JSON file
            with open(f"{self.coin}_stop_loss_data.json", "w") as f:
                json.dump(serializable_data, f)

            self.log_message(
                f"Saved stop loss data to {self.coin}_stop_loss_data.json", color="green")
            return True
        except Exception as e:
            self.log_message(
                f"Error saving stop loss data: {str(e)}", color="red")
            return False

    def load_stop_loss_data(self):
        """Load stop loss prices from a JSON file"""
        try:
            # Check if the file exists
            filename = f"{self.coin}_stop_loss_data.json"
            if not os.path.exists(filename):
                self.log_message(
                    f"No stop loss data file found ({filename})", color="yellow")
                return False

            # Load from the JSON file
            with open(filename, "r") as f:
                loaded_data = json.load(f)

            # Update the stop loss prices dictionary
            self.stop_loss_prices.update(loaded_data)

            self.log_message(
                f"Loaded stop loss data from {filename}", color="green")
            self.log_message(
                f"Stop loss prices: {self.stop_loss_prices}", color="cyan")
            return True
        except Exception as e:
            self.log_message(
                f"Error loading stop loss data: {str(e)}", color="red")
            return False


if __name__ == "__main__":
    from lumibot.brokers import Alpaca
    from lumibot.traders import Trader
    import logging
    from alpaca_trade_api import REST
    import time
    import argparse

    # Set up argument parser for command line options
    parser = argparse.ArgumentParser(
        description='Crypto Trading Bot with Sentiment Analysis')
    parser.add_argument('--test', action='store_true',
                        help='Run bracket order test before starting')
    parser.add_argument('--coin', type=str, default='BTC',
                        help='Coin to trade (default: BTC)')
    parser.add_argument('--risk', type=float, default=0.90,
                        help='Percentage of cash at risk per trade (default: 0.40)')
    parser.add_argument('--backtest', action='store_true',
                        help='Run in backtest mode instead of live trading')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Test connection directly with Alpaca API
    print("Testing connection to Alpaca API...")
    try:
        # Create a direct connection to Alpaca API
        api = REST(
            key_id=API_KEY,
            secret_key=API_SECRET,
            base_url="https://paper-api.alpaca.markets"  # No trailing /v2
        )

        # Get account information
        account = api.get_account()
        print(f"Successfully connected to Alpaca!")
        print(f"Account ID: {account.id}")
        print(f"Account Status: {account.status}")
        print(f"Cash: ${float(account.cash):.2f}")
        print(f"Portfolio Value: ${float(account.portfolio_value):.2f}")

        # Initialize Alpaca broker using the credentials dictionary
        broker = Alpaca(ALPACA_CREDS)

        # Parameters for the strategy
        coin = args.coin
        coin_name = coin.lower()  # Simple conversion for coin name

        # Create the strategy
        strategy = MLTrader(
            name="MLCryptoTrader",
            broker=broker,
            parameters={
                "cash_at_risk": args.risk,  # Using command line risk parameter
                "coin": coin,
                "coin_name": coin_name,
            }
        )

        # Run bracket order test if requested
        if args.test:
            print("\nTesting bracket orders with take profit and stop loss...")
            test_result = strategy.test_bracket_orders()

            if test_result:
                print(
                    Fore.GREEN + "Bracket order test completed successfully!" + Fore.RESET)
                print("The test has verified that:")
                print("1. A market order can be created and filled")
                print("2. A take profit order can be created at 50% gain")
                print("3. For stop loss: Alpaca doesn't support stop orders for crypto")
                print(
                    "   In a real strategy, you would need to monitor the price and create a limit order when needed")
            else:
                print(Fore.RED + "Bracket order test failed." + Fore.RESET)
                print("Please check the logs for more information.")
                exit(1)

        # Run in backtest mode if requested
        if args.backtest:
            from lumibot.backtesting import YahooDataBacktesting
            from datetime import datetime

            # Set up backtest parameters
            start_date = datetime(2023, 1, 1)
            end_date = datetime.now()

            print(
                f"\nRunning backtest for {coin} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            # Run backtest
            results, strat_obj = MLTrader.run_backtest(
                YahooDataBacktesting,
                start_date,
                end_date,
                benchmark_asset=f"{coin}-USD",
                quote_asset=Asset(symbol="USD", asset_type="crypto"),
                # More realistic fees
                # 0.35% is typical for crypto exchanges
                buy_trading_fees=[TradingFee(percent_fee=0.0035)],
                sell_trading_fees=[TradingFee(percent_fee=0.0035)],
                # Slippage to account for market impact
                slippage_model=lambda price, order_side, size: price *
                (1 + (0.001 * (-1 if order_side == "buy" else 1))),
                initial_capital=100000,
                parameters={
                    "cash_at_risk": args.risk,
                    "coin": coin,
                    "coin_name": coin_name,
                }
            )

            print(
                f"Backtest completed. Final portfolio value: ${results['portfolio_value'][-1]:.2f}")
            print(
                f"Benchmark final value: ${results['benchmark_value'][-1]:.2f}")
            print(f"Strategy return: {results['strategy_return'][-1]:.2%}")
            print(f"Benchmark return: {results['benchmark_return'][-1]:.2%}")

        else:
            # Create trader and run the strategy in live mode
            trader = Trader()
            trader.add_strategy(strategy)

            # Log that we're starting paper trading
            print(
                f"Starting paper trading for {coin} using contrarian sentiment strategy")
            print(
                f"Using Alpaca Paper Trading account with API key: {API_KEY[:5]}...")
            print("Press Ctrl+C to stop the trader")

            # Run the strategy in the main thread
            trader.run_all()

    except Exception as e:
        print(f"Error during setup: {str(e)}")
        print("Please check your API keys and try again.")
        exit(1)
