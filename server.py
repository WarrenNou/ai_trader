from flask import Flask, jsonify, render_template
import os
import subprocess
import threading
import time

# Create Flask app with explicit template folder
app = Flask(__name__, 
            template_folder=os.path.abspath('templates'))

# Global variables to track bot state
bot_process = None
bot_running = False
bot_paused = False

@app.route('/')
def home():
    # Serve the dashboard HTML template
    return render_template('index.html')

@app.route('/health')
def health():
    return "OK"

@app.route('/start')
def start_bot():
    global bot_process, bot_running, bot_paused
    
    if bot_running and bot_paused:
        # Resume the paused bot
        bot_paused = False
        return "Trading bot resumed!"
    
    if bot_running:
        return "Trading bot is already running!"
    
    # Start the trading bot in a separate thread
    thread = threading.Thread(target=run_trading_bot)
    thread.daemon = True
    thread.start()
    return "Trading bot started!"

@app.route('/pause')
def pause_bot():
    global bot_paused, bot_running
    
    if not bot_running:
        return "No bot running to pause!"
    
    bot_paused = True
    return "Trading bot paused!"

@app.route('/stop')
def stop_bot():
    global bot_process, bot_running, bot_paused
    
    if not bot_running:
        return "No bot running to stop!"
    
    if bot_process:
        # Terminate the bot process
        bot_process.terminate()
        bot_process = None
    
    bot_running = False
    bot_paused = False
    return "Trading bot stopped!"

@app.route('/api/status')
def get_status():
    global bot_running, bot_paused
    
    status = "Running"
    if not bot_running:
        status = "Stopped"
    elif bot_paused:
        status = "Paused"
    
    # Mock data for demonstration
    return jsonify({
        "status": status,
        "last_started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "portfolio": {
            "cash": 10000.00,
            "equity": 5000.00,
            "total_value": 15000.00
        },
        "trades": [],
        "logs": []
    })

def run_trading_bot():
    global bot_process, bot_running, bot_paused
    
    bot_running = True
    
    while bot_running:
        if not bot_paused:
            # Use a lightweight approach to run the trading script
            bot_process = subprocess.Popen(["python", "7. dip_contra_fees.py"])
            bot_process.wait()
        
        # Sleep to prevent CPU overuse if paused
        time.sleep(1)
    
    bot_running = False

# Start the bot automatically when the server starts
def start_bot_on_startup():
    thread = threading.Thread(target=run_trading_bot)
    thread.daemon = True
    thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    
    # Start the bot automatically
    start_bot_on_startup()
    
    # Check if we're in a production environment
    if os.environ.get("ENVIRONMENT") == "production":
        # Use Waitress for production
        from waitress import serve
        print("Starting production server with Waitress...")
        serve(app, host="0.0.0.0", port=port, threads=4)
    else:
        # Use Flask's development server for development
        print("Starting development server...")
        app.run(host="0.0.0.0", port=port)
