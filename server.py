
import os
import subprocess
import threading
import time

# Create Flask app with explicit template folder
app = Flask(__name__, 
            template_folder=os.path.abspath('templates'))

# Global variables to track bot states
crypto_bot_process = None
stock_bot_process = None
crypto_bot_running = False
stock_bot_running = False
crypto_bot_paused = False
stock_bot_paused = False

@app.route('/')
def home():
    # Serve the dashboard HTML template
    return render_template('index.html')

@app.route('/health')
def health():
    return "OK"

@app.route('/start/<bot_type>')
def start_bot(bot_type):
    if bot_type == 'crypto':
        global crypto_bot_process, crypto_bot_running, crypto_bot_paused
        
        if crypto_bot_running and crypto_bot_paused:
            # Resume the paused bot
            crypto_bot_paused = False
            return "Crypto trading bot resumed!"
        
        if crypto_bot_running:
            return "Crypto trading bot is already running!"
        
        # Start the crypto trading bot in a separate thread
        thread = threading.Thread(target=run_crypto_bot)
        thread.daemon = True
        thread.start()
        return "Crypto trading bot started!"
    
    elif bot_type == 'stock':
        global stock_bot_process, stock_bot_running, stock_bot_paused
        
        if stock_bot_running and stock_bot_paused:
            # Resume the paused bot
            stock_bot_paused = False
            return "Stock trading bot resumed!"
        
        if stock_bot_running:
            return "Stock trading bot is already running!"
        
        # Start the stock trading bot in a separate thread
        thread = threading.Thread(target=run_stock_bot)
        thread.daemon = True
        thread.start()
        return "Stock trading bot started!"
    
    else:
        return "Invalid bot type specified!"

@app.route('/pause/<bot_type>')
def pause_bot(bot_type):
    if bot_type == 'crypto':
        global crypto_bot_paused, crypto_bot_running
        
        if not crypto_bot_running:
            return "No crypto bot running to pause!"
        
        crypto_bot_paused = True
        return "Crypto trading bot paused!"
    
    elif bot_type == 'stock':
        global stock_bot_paused, stock_bot_running
        
        if not stock_bot_running:
            return "No stock bot running to pause!"
        
        stock_bot_paused = True
        return "Stock trading bot paused!"
    
    else:
        return "Invalid bot type specified!"

@app.route('/stop/<bot_type>')
def stop_bot(bot_type):
    if bot_type == 'crypto':
        global crypto_bot_process, crypto_bot_running, crypto_bot_paused
        
        if not crypto_bot_running:
            return "No crypto bot running to stop!"
        
        if crypto_bot_process:
            # Terminate the bot process
            crypto_bot_process.terminate()
            crypto_bot_process = None
        
        crypto_bot_running = False
        crypto_bot_paused = False
        return "Crypto trading bot stopped!"
    
    elif bot_type == 'stock':
        global stock_bot_process, stock_bot_running, stock_bot_paused
        
        if not stock_bot_running:
            return "No stock bot running to stop!"
        
        if stock_bot_process:
            # Terminate the bot process
            stock_bot_process.terminate()
            stock_bot_process = None
        
        stock_bot_running = False
        stock_bot_paused = False
        return "Stock trading bot stopped!"
    
    else:
        return "Invalid bot type specified!"

@app.route('/api/status')
def get_status():
    global crypto_bot_running, crypto_bot_paused, stock_bot_running, stock_bot_paused
    
    crypto_status = "Running" if crypto_bot_running and not crypto_bot_paused else "Paused" if crypto_bot_running and crypto_bot_paused else "Stopped"
    stock_status = "Running" if stock_bot_running and not stock_bot_paused else "Paused" if stock_bot_running and stock_bot_paused else "Stopped"
    
    # Track last started times
    crypto_last_started = None
    stock_last_started = None
    
    try:
        if os.path.exists("bot_status.json"):
            with open("bot_status.json", "r") as f:
                import json
                status_data = json.load(f)
                crypto_last_started = status_data.get("crypto_last_started")
                stock_last_started = status_data.get("stock_last_started")
    except Exception as e:
        print(f"Error reading status data: {str(e)}")
    
    # Get portfolio data from file
    portfolio = {
        "cash": 10000.00,
        "equity": 5000.00,
        "total_value": 15000.00
    }
    
    try:
        if os.path.exists("portfolio_data.json"):
            with open("portfolio_data.json", "r") as f:
                import json
                portfolio_data = json.load(f)
                portfolio = {
                    "cash": portfolio_data.get("cash", 10000.00),
                    "equity": portfolio_data.get("equity", 5000.00),
                    "total_value": portfolio_data.get("total_value", 15000.00)
                }
    except Exception as e:
        print(f"Error reading portfolio data: {str(e)}")
    
    # Get logs from both bots
    logs = []
    try:
        # Get crypto bot logs
        if os.path.exists("bot_logs.txt"):
            with open("bot_logs.txt", "r") as f:
                log_lines = f.readlines()[-10:]  # Get last 10 lines
                for line in log_lines:
                    log_type = "neutral"
                    if "BUY" in line or "positive" in line:
                        log_type = "positive"
                    elif "SELL" in line or "negative" in line:
                        log_type = "negative"
                    logs.append({"message": f"[Crypto] {line.strip()}", "type": log_type})
        
        # Get stock bot logs
        if os.path.exists("stock_bot_logs.txt"):
            with open("stock_bot_logs.txt", "r") as f:
                log_lines = f.readlines()[-10:]  # Get last 10 lines
                for line in log_lines:
                    log_type = "neutral"
                    if "BUY" in line or "positive" in line:
                        log_type = "positive"
                    elif "SELL" in line or "negative" in line:
                        log_type = "negative"
                    logs.append({"message": f"[Stock] {line.strip()}", "type": log_type})
        
        # Sort logs by timestamp if possible
        logs.sort(key=lambda x: x["message"], reverse=True)
        
    except Exception as e:
        logs.append({"message": f"Error reading logs: {str(e)}", "type": "negative"})
    
    # Get trades data
    trades = []
    try:
        if os.path.exists("trades.json"):
            with open("trades.json", "r") as f:
                import json
                trades = json.load(f)
    except Exception as e:
        print(f"Error reading trades data: {str(e)}")
    
    return jsonify({
        "crypto_status": crypto_status,
        "stock_status": stock_status,
        "crypto_last_started": crypto_last_started,
        "stock_last_started": stock_last_started,
        "portfolio": portfolio,
        "trades": trades,
        "logs": logs
    })

def run_crypto_bot():
    global crypto_bot_process, crypto_bot_running, crypto_bot_paused
    
    crypto_bot_running = True
    
    # Update last started time
    try:
        import json
        status_data = {}
        if os.path.exists("bot_status.json"):
            with open("bot_status.json", "r") as f:
                status_data = json.load(f)
        
        status_data["crypto_last_started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open("bot_status.json", "w") as f:
            json.dump(status_data, f)
    except Exception as e:
        print(f"Error updating status data: {str(e)}")
    
    # Create a log file for the bot
    log_file = open("bot_logs.txt", "a")
    
    while crypto_bot_running:
        if not crypto_bot_paused:
            # Use a lightweight approach to run the crypto trading script
            # Redirect output to our log file
            crypto_bot_process = subprocess.Popen(
                ["python", "7. dip_contra_fees.py"],
                stdout=log_file,
                stderr=log_file,
                universal_newlines=True
            )
            crypto_bot_process.wait()
        
        # Sleep to prevent CPU overuse if paused
        time.sleep(1)
    
    log_file.close()
    crypto_bot_running = False

def run_stock_bot():
    global stock_bot_process, stock_bot_running, stock_bot_paused
    
    stock_bot_running = True
    
    # Update last started time
    try:
        import json
        status_data = {}
        if os.path.exists("bot_status.json"):
            with open("bot_status.json", "r") as f:
                status_data = json.load(f)
        
        status_data["stock_last_started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open("bot_status.json", "w") as f:
            json.dump(status_data, f)
    except Exception as e:
        print(f"Error updating status data: {str(e)}")
    
    # Create a log file for the bot
    log_file = open("stock_bot_logs.txt", "a")
    
    while stock_bot_running:
        if not stock_bot_paused:
            # Use a lightweight approach to run the stock trading script
            # Redirect output to our log file
            stock_bot_process = subprocess.Popen(
                ["python", "stock_trader_multi.py"],
                stdout=log_file,
                stderr=log_file,
                universal_newlines=True
            )
            stock_bot_process.wait()
        
        # Sleep to prevent CPU overuse if paused
        time.sleep(1)
    
    log_file.close()
    stock_bot_running = False

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
