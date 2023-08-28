# flask_app.py

from flask import Flask, jsonify, render_template, g
from livereload import Server
import logging
import os
import pandas as pd
import psutil
from stock_analysis import HMMStockPredictor
company_name = None

app = Flask(__name__)

def kill_process_on_port(port):
    """Kill the process running on the given port."""
    for proc in psutil.process_iter():
        try:
            for conns in proc.connections(kind='inet'):
                if conns.laddr.port == port:
                    proc.kill()
                    return
        except Exception as e:
            print(f"Error killing process on port {port}: {e}")

@app.before_request
def before_request():
    local_company_name = app.config.get('COMPANY_NAME')
    if not local_company_name:
        app.logger.error("Company name is not provided or is set to None.")
        return "Error: Company name is missing", 500

    g.test_data = app.config.get('TEST_DATA')
    g.predicted_close_prices = app.config.get('PREDICTED_CLOSE_PRICES')
    
    # Initialize the predictor object
    g.predictor = HMMStockPredictor(
        company=app.config.get('COMPANY_NAME'),
        start_date=app.config.get('START_DATE'),
        end_date=app.config.get('END_DATE'),
        future_days=app.config.get('FUTURE_DAYS')
    )
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    data_df = g.test_data  # Access test_data from g object
    
    if data_df is None:
        app.logger.error("Failed to fetch test data.")
        return "Error fetching data", 500

    # Check if days_in_future is None
    if g.predictor.days_in_future is None:
        app.logger.error("days_in_future is not initialized.")
        return "Error: days_in_future is missing", 500
    
    # Generate date index for predictions
    last_date = pd.to_datetime(data_df.index[-1])
    index = pd.date_range(last_date, periods=g.predictor.days_in_future + 1, freq="D")[1:]
    if None in data_df.index:
        app.logger.error("Detected None values in data_df dates.")
        return "Error processing dates", 500

    if None in index:
        app.logger.error("Detected None values in predicted dates.")
        return "Error processing predicted dates", 500
        
    app.logger.info("Fetching data for index route.")

    # Calculate % change
    y_pred = g.predictor.predict_close_prices_for_period()
    y_pred_pct_change = (y_pred - y_pred[0]) / y_pred[0] * 100

    # Calculate actual prices using % changes
    actual_prices = []
    last_actual_close = data_df["Close"].iloc[-1]
    for pct_change in y_pred_pct_change:
        actual_close = last_actual_close * (1 + (pct_change/100))
        actual_prices.append(actual_close)

    # Convert the dates in predicted_data to the desired format
    formatted_dates = [date.strftime('%Y-%m-%d') for date in index]

    return jsonify({
        'actual_data': {
            'dates': [date.strftime('%Y-%m-%d') for date in data_df.index.tolist()],
            'values': data_df["Close"].tolist()
        },
        'predicted_data': {
            'dates': formatted_dates,
            'values': actual_prices
        }
    })
    
server_started = False
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def start_server(company_name=None, test_data=None, predicted_close_prices=None, start_date=None, end_date=None, future_days=None):
    global server_started
    
    # Set the company_name and data in the Flask app's config
    app.config['COMPANY_NAME'] = company_name
    app.config['TEST_DATA'] = test_data
    app.config['PREDICTED_CLOSE_PRICES'] = predicted_close_prices
    app.config['START_DATE'] = start_date
    app.config['END_DATE'] = end_date
    app.config['FUTURE_DAYS'] = future_days
    
    # Kill any processes using port 5000
    kill_process_on_port(5000)
    
    if not server_started:
        if os.environ.get("USE_LIVERELOAD", "False") == "True":
            try:
                server = Server(app.wsgi_app)
                server.watch('templates/*.*')
                server.watch('static/*.*')
                server.serve(port=5000, debug=False)
            except KeyboardInterrupt:
                print("\nShutting down livereload server...")
            except Exception as e:
                logging.error("Error with livereload: %s", e)
        else:
            try:
                app.run(port=5000, debug=True, use_reloader=False)
            except KeyboardInterrupt:
                print("\nShutting down Flask server...")
            except Exception as e:
                logging.error("Error starting the Flask app: %s", e)
        server_started = True

if __name__ == "__main__":
    # Start the Flask server
    start_server()