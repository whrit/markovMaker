# flask_app.py

from flask import Flask, jsonify, render_template, g
from livereload import Server
import logging
import os
import pandas as pd  # <-- Add this import

# Import necessary functions and variables from stock_analysis.py
from stock_analysis import HMMStockPredictor, company_name, start, end, future

app = Flask(__name__)

import os

def kill_process_on_port(port):
    """Kill the process running on the given port."""
    try:
        # Find the process using the port and kill it
        os.system(f"fuser -k {port}/tcp")
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")

@app.before_request
def before_request():
    print(f"Company Name: {company_name}")
    """
    This function will run before each request to initialize the HMMStockPredictor object.
    """
    g.predictor = HMMStockPredictor(
        company=company_name,
        start_date=start,
        end_date=end,
        future_days=future
    )
    g.predictor.fit()  # Ensure the model is fitted before making predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data', methods=['GET'])
def get_data():
    data_df = g.predictor.test_data  # Access predictor from g object
    if g.predictor.test_data is None:
        app.logger.error("Failed to fetch test data.")
        return "Error fetching data", 500
    
    # Generate date index for predictions
    last_date = pd.to_datetime(data_df.index[-1])
    index = pd.date_range(last_date, periods=g.predictor.days_in_future + 1, freq="D")[1:]
    if None in data_df.index:
        app.logger.error("Detected None values in data_df dates.")
        return "Error processing dates", 500

    if None in index:
        app.logger.error("Detected None values in predicted dates.")
        return "Error processing predicted dates", 500
        
    app.logger.info("Fetching data for get_data route.")

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

def start_server():
    global server_started
    
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
                # Any other cleanup code can be added here if needed
            except Exception as e:
                logging.error("Error with livereload: %s", e)
        else:
            try:
                app.run(port=5000, debug=True, use_reloader=False)
            except KeyboardInterrupt:
                print("\nShutting down Flask server...")
                # Any other cleanup code can be added here if needed
            except Exception as e:
                logging.error("Error starting the Flask app: %s", e)
        server_started = True

if __name__ == "__main__":
    # Start the Flask server
    start_server()
