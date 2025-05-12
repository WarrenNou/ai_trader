
import os
import subprocess
import threading
import time
import json
from flask import Flask, render_template, jsonify, request

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

# Configuration for bots
crypto_config = {
    "coin": "BTC",
    "risk": 0.40,
    "strategy": "Contrarian Sentiment"
}

stock_config = {
    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    "risk": 0.30,
    "strategy": "Multi-Stock Sentiment"
}

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
        
        # Log the stop event
        log_event("Crypto bot stopped by user", "neutral")
        
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
        
        # Log the stop event
        log_event("Stock bot stopped by user", "neutral")
        
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
        
        # Get server logs
        if os.path.exists("server_logs.txt"):
            with open("server_logs.txt", "r") as f:
                log_lines = f.readlines()[-5:]  # Get last 5 lines
                for line in log_lines:
                    log_type = "neutral"
                    if "error" in line.lower() or "failed" in line.lower():
                        log_type = "negative"
                    elif "success" in line.lower() or "started" in line.lower():
                        log_type = "positive"
                    logs.append({"message": f"[Server] {line.strip()}", "type": log_type})
        
        # Sort logs by timestamp if possible
        logs.sort(key=lambda x: x["message"], reverse=True)
        
    except Exception as e:
        logs.append({"message": f"Error reading logs: {str(e)}", "type": "negative"})
    
    # Get trades data
    trades = []
    try:
        if os.path.exists("trades.json"):
            with open("trades.json", "r") as f:
                trades = json.load(f)
    except Exception as e:
        print(f"Error reading trades data: {str(e)}")
    
    return jsonify({
        "crypto_status": crypto_status,
        "stock_status": stock_status,
        "crypto_last_started": crypto_last_started,
        "stock_last_started": stock_last_started,
        "crypto_config": crypto_config,
        "stock_config": stock_config,
        "portfolio": portfolio,
        "trades": trades,
        "logs": logs
    })

@app.route('/api/config', methods=['GET', 'POST'])
def bot_config():
    global crypto_config, stock_config
    
    if request.method == 'POST':
        data = request.json
        bot_type = data.get('bot_type')
        
        if bot_type == 'crypto':
            # Update crypto bot configuration
            if 'coin' in data:
                crypto_config['coin'] = data['coin']
            if 'risk' in data:
                crypto_config['risk'] = float(data['risk'])
            
            # Log the configuration change
            log_event(f"Crypto bot configuration updated: {crypto_config}", "neutral")
            
            return jsonify({"status": "success", "message": "Crypto bot configuration updated"})
            
        elif bot_type == 'stock':
            # Update stock bot configuration
            if 'symbols' in data:
                stock_config['symbols'] = data['symbols']
            if 'risk' in data:
                stock_config['risk'] = float(data['risk'])
            
            # Log the configuration change
            log_event(f"Stock bot configuration updated: {stock_config}", "neutral")
            
            return jsonify({"status": "success", "message": "Stock bot configuration updated"})
        
        else:
            return jsonify({"status": "error", "message": "Invalid bot type"})
    
    # Return current configuration
    return jsonify({
        "crypto": crypto_config,
        "stock": stock_config
    })

def log_event(message, log_type="neutral"):
    """Log an event to the server logs file"""
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open("server_logs.txt", "a") as f:
            f.write(f"{timestamp} | {message}\n")
    except Exception as e:
        print(f"Error writing to log file: {str(e)}")

def run_crypto_bot():
    global crypto_bot_process, crypto_bot_running, crypto_bot_paused
    
    crypto_bot_running = True
    
    # Update last started time
    try:
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
    
    # Log the start event
    log_event(f"Crypto bot started with coin={crypto_config['coin']}, risk={crypto_config['risk']}", "positive")
    
    while crypto_bot_running:
        if not crypto_bot_paused:
            # Use a lightweight approach to run the crypto trading script
            # Redirect output to our log file
            crypto_bot_process = subprocess.Popen(
                ["python", "7. dip_contra_fees.py", "--coin", crypto_config['coin'], "--risk", str(crypto_config['risk'])],
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
    
    # Log the start event
    log_event(f"Stock bot started with risk={stock_config['risk']}", "positive")
    
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

# Create necessary files if they don't exist
def initialize_files():
    # Create bot_status.json if it doesn't exist
    if not os.path.exists("bot_status.json"):
        with open("bot_status.json", "w") as f:
            json.dump({
                "crypto_last_started": None,
                "stock_last_started": None
            }, f)
    
    # Create portfolio_data.json if it doesn't exist
    if not os.path.exists("portfolio_data.json"):
        with open("portfolio_data.json", "w") as f:
            json.dump({
                "cash": 10000.00,
                "equity": 5000.00,
                "total_value": 15000.00
            }, f)
    
    # Create trades.json if it doesn't exist
    if not os.path.exists("trades.json"):
        with open("trades.json", "w") as f:
            json.dump([], f)
    
    # Create log files if they don't exist
    for log_file in ["bot_logs.txt", "stock_bot_logs.txt", "server_logs.txt"]:
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write(f"Log file created at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

@app.route('/api/portfolio')
def get_portfolio():
    """Get detailed portfolio information"""
    try:
        # Get portfolio data from file
        portfolio = {
            "cash": 10000.00,
            "equity": 5000.00,
            "total_value": 15000.00,
            "previous_cash": 10000.00,
            "previous_equity": 5000.00,
            "previous_total": 15000.00,
            "positions": []
        }
        
        # Try to load portfolio data from file
        if os.path.exists("portfolio_data.json"):
            with open("portfolio_data.json", "r") as f:
                portfolio_data = json.load(f)
                portfolio.update({
                    "cash": portfolio_data.get("cash", 10000.00),
                    "equity": portfolio_data.get("equity", 5000.00),
                    "total_value": portfolio_data.get("total_value", 15000.00),
                    "previous_cash": portfolio_data.get("previous_cash", portfolio_data.get("cash", 10000.00)),
                    "previous_equity": portfolio_data.get("previous_equity", portfolio_data.get("equity", 5000.00)),
                    "previous_total": portfolio_data.get("previous_total", portfolio_data.get("total_value", 15000.00))
                })
        
        # Try to load portfolio state for positions
        if os.path.exists("portfolio_state.json"):
            with open("portfolio_state.json", "r") as f:
                state_data = json.load(f)
                
                # Get positions
                if "positions" in state_data:
                    positions = []
                    for symbol, position in state_data["positions"].items():
                        # Try to get target allocation from stocks_config
                        target_allocation = 0.0
                        if os.path.exists("stocks_config.json"):
                            with open("stocks_config.json", "r") as sc_file:
                                stocks_config = json.load(sc_file)
                                for stock in stocks_config:
                                    if stock["symbol"] == symbol:
                                        target_allocation = stock["weight"]
                                        break
                        
                        # Calculate allocation
                        allocation = position["value"] / state_data["portfolio_value"] if state_data["portfolio_value"] > 0 else 0
                        
                        positions.append({
                            "symbol": symbol,
                            "quantity": position["quantity"],
                            "price": position["price"],
                            "value": position["value"],
                            "allocation": allocation,
                            "target_allocation": target_allocation
                        })
                    
                    portfolio["positions"] = positions
        
        # Force refresh from trading bot if running
        if stock_bot_running and not stock_bot_paused:
            # Log the refresh request
            log_event("Portfolio refresh requested from dashboard", "neutral")
            
            # Update portfolio_data.json with latest data
            # This would ideally trigger the trading bot to update the file
            # For now, we'll just return what we have
            
        return jsonify(portfolio)
    
    except Exception as e:
        print(f"Error getting portfolio data: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    
    # Initialize necessary files
    initialize_files()
    
    # Log server start
    log_event("Server starting up", "positive")
    
    # Check if we're in a production environment
    if os.environ.get("ENVIRONMENT") == "production":
        # Use Waitress for production
        from waitress import serve
        print("Starting production server with Waitress...")
        log_event("Starting production server with Waitress", "positive")
        serve(app, host="0.0.0.0", port=port, threads=4)
    else:
        # Use Flask's development server for development
        print("Starting development server...")
        log_event("Starting development server", "positive")
        app.run(host="0.0.0.0", port=port, debug=True)
