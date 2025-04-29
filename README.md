# Multi-Stock Trading Bot

This trading bot uses sentiment analysis from financial news to make trading decisions across multiple stocks simultaneously.

## Features

- **Multi-Stock Support**: Trade multiple stocks with customizable allocation weights
- **Sentiment Analysis**: Uses FinBERT model to analyze financial news sentiment
- **Backtesting**: Full backtesting support with Yahoo Finance data
- **Live Trading**: Ready for live trading with Alpaca

## Stocks Included

- **Apple (AAPL)**: Consumer technology giant
- **Amazon (AMZN)**: E-commerce and cloud computing leader
- **Tesla (TSLA)**: Electric vehicle and clean energy company
- **Microsoft (MSFT)**: Software and cloud services provider

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set your Alpaca API credentials in `stock_trader_multi.py`

3. Run backtesting:
   ```
   python stock_trader_multi.py
   ```

4. For live trading, uncomment the live trading section at the bottom of the file

## Configuration

You can customize the stock selection and allocation weights by modifying the `stocks_config` list:

```python
stocks_config = [
    {"symbol": "AAPL", "name": "Apple", "weight": 0.25},
    {"symbol": "AMZN", "name": "Amazon", "weight": 0.25},
    {"symbol": "TSLA", "name": "Tesla", "weight": 0.25},
    {"symbol": "MSFT", "name": "Microsoft", "weight": 0.25}
]
```

## Performance Tracking

The bot tracks all trades and portfolio performance. Results are logged during execution and can be analyzed after backtesting.

