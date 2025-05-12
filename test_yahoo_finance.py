import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import random

def test_yahoo_finance():
    print("Testing Yahoo Finance data access...")
    
    # Test symbols - a mix of stocks, ETFs, and crypto
    symbols = ["AAPL", "SPY", "BTC-USD", "ETH-USD"]
    
    # Date range (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Test each symbol with retry mechanism
    for symbol in symbols:
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                if retry_count > 0:
                    # Add exponential backoff with jitter
                    wait_time = (2 ** retry_count) + random.uniform(0, 1)
                    print(f"  Retry {retry_count}/{max_retries} after {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                
                print(f"\nTesting symbol: {symbol}")
                
                # Get ticker info with a delay to avoid rate limiting
                ticker = yf.Ticker(symbol)
                time.sleep(1)  # Add delay between API calls
                
                # Try to get basic info first
                info = ticker.info
                print(f"  Symbol exists: {bool(info)}")
                
                if 'shortName' in info:
                    print(f"  Name: {info['shortName']}")
                
                # Add another delay before getting historical data
                time.sleep(2)
                
                # Get historical data with a smaller date range
                # Use a smaller date range to reduce data size
                shorter_start = end_date - timedelta(days=7)
                data = ticker.history(start=shorter_start, end=end_date)
                
                if data.empty:
                    print(f"  ❌ No data available for {symbol}")
                else:
                    print(f"  ✅ Data available: {len(data)} rows")
                    print(f"  Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                    print(f"  Last close price: {data['Close'].iloc[-1]:.2f}")
                    
                    # Plot the data
                    plt.figure(figsize=(10, 5))
                    plt.plot(data.index, data['Close'])
                    plt.title(f"{symbol} Price Chart")
                    plt.xlabel("Date")
                    plt.ylabel("Price")
                    plt.grid(True)
                    plt.savefig(f"{symbol}_test.png")
                    print(f"  Chart saved as {symbol}_test.png")
                
                # Mark as successful to exit retry loop
                success = True
                
                # Add delay between symbols
                if symbol != symbols[-1]:
                    delay = 5 + random.uniform(0, 2)
                    print(f"  Waiting {delay:.2f} seconds before next symbol...")
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"  ❌ Error with {symbol}: {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"  Failed after {max_retries} attempts")
    
    print("\nYahoo Finance test complete!")
    print("\nTIP: If you're still getting rate limited, try these solutions:")
    print("1. Use a VPN to change your IP address")
    print("2. Wait a few hours before trying again")
    print("3. Consider using an API key-based service like Alpha Vantage or IEX Cloud")
    print("4. For backtesting, download historical data once and save it locally")

if __name__ == "__main__":
    test_yahoo_finance()
