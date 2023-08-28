import warnings
import logging
import itertools
import pandas as pd
import numpy as np
from pandas_datareader import data
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tqdm import tqdm
import argparse
import sys
import os
import yfinance as yf
from datetime import timedelta, datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from livereload import Server
from sklearn.preprocessing import MinMaxScaler
import json
import subprocess

# Setting up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warning in hmmlearn
warnings.filterwarnings("ignore")

def cpugpu():
    import tensorflow as tf
    # Check if GPU is available and print the list of GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"Found GPU: {gpu}")
    else:
        print("No GPU devices found.")
    
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Specify the GPU device to use (e.g., use the first GPU)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            
            # Test TensorFlow with a simple computation on the GPU
            with tf.device('/GPU:0'):
                a = tf.constant([1.0, 2.0, 3.0])
                b = tf.constant([4.0, 5.0, 6.0])
                c = a * b

            print("GPU is available and TensorFlow is using it.")
            print("Result of the computation on GPU:", c.numpy())
        except RuntimeError as e:
            print("Error while setting up GPU:", e)
    else:
        print("No GPU devices found, TensorFlow will use CPU.")

class HMMStockPredictor:
    _instance = None
    _model_trained = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(HMMStockPredictor, cls).__new__(cls)
        return cls._instance
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("TensorFlow version:", tf.__version__)

    def __init__(
        self,
        company,
        start_date,
        end_date,
        future_days,
        test_size=0.33,
        n_hidden_states=4,
        n_latency_days=10,
        n_intervals_frac_change=50,
        n_intervals_frac_high=10,
        n_intervals_frac_low=10,
    ):
        self._init_logger()
        self.company = company
        self.start_date = start_date
        self.end_date = end_date
        self.n_latency_days = n_latency_days
        self.hmm = GaussianHMM(n_components=n_hidden_states)
        self._split_train_test_data(test_size)
        self._compute_all_possible_outcomes(
            n_intervals_frac_change, n_intervals_frac_high, n_intervals_frac_low
        )
        self.predicted_close = None
        self.days_in_future = future_days
        self.predicted_close_prices = None

    def handle_nan_values(self):
        if self.test_data.isna().any().any():
            print("NaN values detected in the data!")
            # Fill NaN values with the mean of the column
            self.test_data.fillna(self.test_data.mean(), inplace=True)
    
    def _init_logger(self):
        self._logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

    def _split_train_test_data(self, test_size):
        """Downloads data and splits it into training and testing datasets."""
        # Use yfinance to load the required financial data.
        used_data = yf.download(self.company, start=self.start_date, end=self.end_date)
        if used_data.empty:
            print(f"Failed to fetch data for {self.company} from {self.start_date} to {self.end_date}.")
            sys.exit()

        # Do not shuffle the data as it is a time series
        _train_data, test_data = train_test_split(
            used_data, test_size=test_size, shuffle=False
        )
        self.train_data = _train_data
        self.test_data = test_data
        self.handle_nan_values()

        # Vectorized operation to drop columns
        self.train_data.drop(columns=["Volume", "Adj Close"], inplace=True)
        self.test_data.drop(columns=["Volume", "Adj Close"], inplace=True)

        # Set days attribute
        self.days = len(test_data)

    @staticmethod
    def _extract_features(data):
        """Extract the features using vectorized operations."""
        frac_change = (data["Close"] - data["Open"]) / data["Open"]
        frac_high = (data["High"] - data["Open"]) / data["Open"]
        frac_low = (data["Open"] - data["Low"]) / data["Open"]
        return np.column_stack((frac_change, frac_high, frac_low))

    def fit(self):
        """Fit the continuous emission Gaussian HMM."""
        self._logger.info(">>> Extracting Features")
        observations = HMMStockPredictor._extract_features(self.train_data)
        self._logger.info("Features extraction Completed <<<")
        # Fit the HMM using the fit feature of hmmlearn
        self.hmm.fit(observations)
        self._model_trained = True

    def _compute_all_possible_outcomes(
        self, n_intervals_frac_change, n_intervals_frac_high, n_intervals_frac_low
    ):
        """Creates np arrays with evenly  spaced numbers for each range."""
        frac_change_range = np.linspace(-0.1, 0.1, n_intervals_frac_change)
        frac_high_range = np.linspace(0, 0.1, n_intervals_frac_high)
        frac_low_range = np.linspace(0, 0.1, n_intervals_frac_low)

        self._possible_outcomes = np.array(
            list(itertools.product(frac_change_range, frac_high_range, frac_low_range))
        )

    def _get_most_probable_outcome(self, day_index):
        """
        Using the fitted HMM, calculate the most probable outcome for a given day (e.g. prices will rise by 0.01).
        :param day_index: Current day index
        :return: The HMM's predicted movements in frac_change, frac_high, frac_low
        """
        # Use the previous n_latency_days worth of data for predictions
        previous_data_start_index = max(0, day_index - self.n_latency_days)
        previous_data_end_index = max(0, day_index - 1)
        previous_data = self.test_data.iloc[
            previous_data_start_index:previous_data_end_index
        ]
        previous_data_features = HMMStockPredictor._extract_features(previous_data)

        outcome_score = []

        # Score all possible outcomes and select the most probable one to use for prediction
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack((previous_data_features, possible_outcome))
            outcome_score.append(self.hmm.score(total_data))

        # Get the index of the most probable outcome and return it
        most_probable_outcome = self._possible_outcomes[np.argmax(outcome_score)]

        return most_probable_outcome

    def predict_close_price(self, day_index):
        """Predict close price for a given day."""
        open_price = self.test_data.iloc[day_index]["Open"]
        (
            predicted_frac_change,
            pred_frac_high,
            pred_frac_low,
        ) = self._get_most_probable_outcome(day_index)
        return open_price * (1 + predicted_frac_change)

    def predict_close_prices_for_period(self):
        """
        Predict close prices for the testing period.
        :return: List object of predicted close prices
        """
        if not self._model_trained:
            self.fit()
        predicted_close_prices = []
        print(
            "Predicting Close prices from "
            + str(self.test_data.index[0])
            + " to "
            + str(self.test_data.index[-1])
        )
        for day_index in tqdm(range(self.days)):
            predicted_close_prices.append(self.predict_close_price(day_index))
        self.predicted_close = predicted_close_prices
        self.predicted_close_prices = predicted_close_prices

        return predicted_close_prices
    
    def real_close_prices(self):
        """ "Store and return the actual close prices."""
        actual_close_prices = self.test_data.loc[:, ["Close"]]
        return actual_close_prices

    def add_future_days(self):
        """
        Add rows to the test data dataframe for the future days being predicted with accurate days. The rows are left
        with NaN values for now as they will be populated whilst predicting.
        """
        last_day = self.test_data.index[-1] + timedelta(days=self.days_in_future)

        # Create a new df with future days x days in the future based off the -f input. Concat the new df with
        # self.test_data.
        future_dates = pd.date_range(
            self.test_data.index[-1] + pd.offsets.DateOffset(1), last_day
        )
        second_df = pd.DataFrame(
            index=future_dates, columns=["High", "Low", "Open", "Close"]
        )
        self.test_data = pd.concat([self.test_data, second_df])

        # Replace the opening price for the first day in the future with the close price of the previous day
        self.test_data.iloc[self.days]["Open"] = self.test_data.iloc[self.days - 1][
            "Close"
        ]

    def predict_close_price_fut_days(self, day_index):
        """
        Predict the close prices for the days in the future beyond the available data and populate the DF accordingly.
        :param day_index - index in DF for  current day being predicted.
        :return: Predicted close price for given day.
        """
        open_price = self.test_data.iloc[day_index]["Open"]

        # Calculate the most likely fractional changes using the trained HMM
        (
            predicted_frac_change,
            pred_frac_high,
            pred_frac_low,
        ) = self._get_most_probable_outcome(day_index)
        predicted_close_price = open_price * (1 + predicted_frac_change)

        # Fill in the dataframe based on predictions
        self.test_data.iloc[day_index]["Close"] = predicted_close_price
        self.test_data.iloc[day_index]["High"] = open_price * (1 + pred_frac_high)
        self.test_data.iloc[day_index]["Low"] = open_price * (1 - pred_frac_low)

        # After setting the predicted values
        if pd.isna(self.test_data.iloc[day_index]["Close"]) or \
        pd.isna(self.test_data.iloc[day_index]["High"]) or \
        pd.isna(self.test_data.iloc[day_index]["Low"]):
            print(f"Warning: NaN values detected for day index {day_index}")

        return predicted_close_price

    def predict_close_prices_for_future(self):
        """
        Calls the "predict_close_price_fut_days" function for each day in the future to predict future close prices.
        """
        predicted_close_prices = []
        future_indices = len(self.test_data) - self.days_in_future
        print(
            "Predicting future Close prices from "
            + str(self.test_data.index[future_indices])
            + " to "
            + str(self.test_data.index[-1])
        )
        # Handle any NaN values before processing
        self.handle_nan_values()

        # Iterate over only the final x days in the test data dataframe.
        for day_index in tqdm(range(future_indices, len(self.test_data))):
            predicted_close_prices.append(self.predict_close_price_fut_days(day_index))
            # Replace the next days Opening price (which is currently NaN) with the previous days predicted close price
            try:
                self.test_data.iloc[day_index + 1]["Open"] = self.test_data.iloc[
                    day_index
                ]["Close"]
            except IndexError:
                continue

        self.predicted_close = predicted_close_prices
        self.predicted_close_prices = predicted_close_prices
        return predicted_close_prices   

def plot_results(df, out_dir, company_name):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Actual_Close'], label='Actual Close', color='blue')
    plt.plot(df['Predicted_Close'], label='Predicted Close', color='red', linestyle='dashed')
    plt.title(f'{company_name} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(f'{out_dir}/{company_name}_HMM_Prediction_Plot.png')
    plt.show()

def check_bool(boolean):
    """
    Corrects an issue that argparser has in which it treats False inputs for a boolean argument as True.
    """
    if isinstance(boolean, bool):
        return boolean
    if boolean.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif boolean.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def calc_mse(input_df):
    """
    Calculates the Mean Squared Error between real and predicted close prices
    :param input_df: Pandas Dataframe containing the data, actual close prices, and predicted close prices
    :return: Mean Squared Error
    """
    actual_arr = (input_df.loc[:, "Actual_Close"]).values
    pred_arr = (input_df.loc[:, "Predicted_Close"]).values
    mse = mean_squared_error(actual_arr, pred_arr)
    return mse


def use_stock_predictor(company_name, start, end, future, metrics, plot, out_dir):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("TensorFlow version:", tf.__version__)

    # Correct incorrect inputs. Inputs should be of the form XXXX, but handle cases when users input 'XXXX'
    company_name = company_name.strip("'").strip('"')
    print(
        "Using continuous Hidden Markov Models to predict stock prices for "
        + str(company_name)
    )

    # Initialise HMMStockPredictor object and fit the HMM
    stock_predictor = HMMStockPredictor(
        company=company_name, start_date=start, end_date=end, future_days=future
    )
    print(
        "Training data period is from "
        + str(stock_predictor.train_data.index[0])
        + " to "
        + str(stock_predictor.train_data.index[-1])
    )
    stock_predictor.fit()

    # Get the predicted and actual stock prices and create a DF for saving if you'd like to get a metric for the model
    if metrics:
        predicted_close = stock_predictor.predict_close_prices_for_period()
        actual_close = stock_predictor.real_close_prices()
        actual_close["Predicted_Close"] = predicted_close
        output_df = actual_close.rename(columns={"Close": "Actual_Close"})

        # Calculate Mean Squared Error and save
        mse = calc_mse(output_df)
        out_name = f"{out_dir}/{company_name}_HMM_Prediction_{str(round(mse, 6))}.xlsx"
        output_df.to_excel(out_name)  # Requires openpyxl installed
        print(
            "All predictions saved. The Mean Squared Error for the "
            + str(stock_predictor.days)
            + " days considered is: "
            + str(mse)
        )

        # Plot and save results if plot is True
        if plot:
            plot_results(output_df, out_dir, company_name)

    # Predict for x days into the future
    if future:
        stock_predictor.add_future_days()
        future_pred_close = stock_predictor.predict_close_prices_for_future()

        print(
            "The predicted stock prices for the next "
            + str(future)
            + " days from "
            + str(stock_predictor.end_date)
            + " are: ",
            future_pred_close,
        )

        out_final = (
            f"{out_dir}/{company_name}_HMM_Predictions_{future}_days_in_future.xlsx"
        )
        stock_predictor.test_data.to_excel(out_final)  # Requires openpyxl installed
        print(
            "The full set of predictions has been saved, including the High, Low, Open and Close prices for "
            + str(future)
            + " days in the future."
        )
    # Return the required data instead of setting global variables
    return stock_predictor.test_data, stock_predictor.predicted_close_prices

def main():
    # Set up arg_parser to handle inputs
    arg_parser = argparse.ArgumentParser()

    # Parse console inputs
    arg_parser.add_argument(
        "-n",
        "--stock_name",
        required=True,
        type=str,
        help="Takes in the name of a stock in the form XXXX e.g. AAPL. 'AAPL' will fail.",
    )
    arg_parser.add_argument(
        "-s",
        "--start_date",
        required=True,
        type=str,
        help="Takes in the start date of the time period being evaluated. Please input dates in the"
        "following way: 'year-month-day'",
    )
    arg_parser.add_argument(
        "-e",
        "--end_date",
        required=True,
        type=str,
        help="Takes in the end date of the time period being evaluated. Please input dates in the"
        "following way: 'year-month-day'",
    )
    arg_parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save the CSV file that contains the actual stock prices along with the "
        "predictions for a given day.",
    )
    arg_parser.add_argument(
        "-p",
        "--plot",
        type=check_bool,
        nargs="?",
        const=True,
        default=False,
        help="Optional: Boolean flag specifying if the results should be plotted or not.",
    )
    arg_parser.add_argument(
        "-f",
        "--future",
        type=int,
        default=None,
        help="Optional: Value specifying how far in the future the user would like predictions.",
    )
    arg_parser.add_argument(
        "-m",
        "--metrics",
        type=check_bool,
        nargs="?",
        const=True,
        default=False,
        help="Optional: Boolean flag specifying that the user would like to see how accurate the "
        "model is at predicting prices in the testing dataset for which real data exists, i.e. "
        "dates before -e. This slows down prediction as all test days will have their close "
        "prices predicted, as opposed to just the future days, but provides a metric to score "
        "the HMM (Mean Squared Error). ",
    )
    args = arg_parser.parse_args()

    # Set variables from arguments
    company_name = args.stock_name
    start = args.start_date
    end = args.end_date
    future = args.future
    metrics = args.metrics
    plot = args.plot
    out_dir = args.out_dir if args.out_dir else os.getcwd()

    cpugpu()

    # Get the required data from the use_stock_predictor function
    test_data, predicted_close_prices = use_stock_predictor(args.stock_name, start, end, future, metrics, plot, out_dir)

    # Pass the required data to the start_server function
    from flask_app import start_server
    start_server(
        company_name=company_name,
        test_data=test_data,
        predicted_close_prices=predicted_close_prices,
        start_date=start,
        end_date=end,
        future_days=future
    )

if __name__ == "__main__":
    # Handle arguments and run predictions
    main()