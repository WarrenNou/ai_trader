import os
from flask import Flask
import threading
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return "Trading bot is running!"

@app.route('/health')
def health():
    return "OK"

def run_trading_bot():
    subprocess.run(["python", "7. dip_contra_fees.py"])

if __name__ == '__main__':
    # Start trading bot in a separate thread
    bot_thread = threading.Thread(target=run_trading_bot)
    bot_thread.daemon = True
    bot_thread.start()
    
    # Start Flask server
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)