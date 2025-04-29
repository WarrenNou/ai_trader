from flask import Flask
import os
import subprocess
import threading

app = Flask(__name__)

@app.route('/')
def home():
    return "Trading bot is running!"

@app.route('/health')
def health():
    return "OK"

@app.route('/start')
def start_bot():
    # Start the trading bot in a separate thread
    thread = threading.Thread(target=run_trading_bot)
    thread.daemon = True
    thread.start()
    return "Trading bot started!"

def run_trading_bot():
    # Use a lightweight approach
    subprocess.run(["python", "stock_trader_multi.py"])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
