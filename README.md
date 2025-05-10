# StockMind Pro: Stock Price Prediction & Analysis

This application predicts the next day's closing price of stocks using a Linear Regression model trained on historical data. It now includes competitor analysis, company information, and a modern, user-friendly interface.



## Features 

- **Predict stock closing price** for the next day using a Linear Regression model
- **Company information** from Wikipedia
- **Industry competitor analysis** with price comparisons
- **Interactive charts** for stock price visualization
- **Modern, responsive UI** for better user experience
- The prediction model uses:
  - Historical price lags
  - Volatility
  - Momentum indicators
  - Moving averages
  - Volume data

## Technologies Used

- **Flask**: Web framework for building the application
- **YFinance**: Fetch stock data from Yahoo Finance
- **Scikit-learn**: For machine learning algorithms and data preprocessing
- **Wikipedia API**: For company information
- **Chart.js**: For interactive data visualization
- **Pandas & NumPy**: For data manipulation
- **Joblib**: For saving and loading machine learning models

## Working
- **Input**: The user enters either a company name or stock ticker symbol
- **Analysis**: The application retrieves company information, historical prices, and identifies industry competitors
- **Prediction**: The Linear Regression model predicts the next day's closing price
- **Output**: Results are displayed in an intuitive interface with charts and competitor comparison

## Note
-The model has been trained once with a 9-year span of Data from Tesla, and is deployed as best_model.pkl

## Installation

Follow these steps to get your local development environment set up:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/stockmind-pro.git
    cd stockmind-pro
    ```

2. Create and activate a virtual environment:
    - For Windows:
      ```bash
      python -m venv venv
      venv\Scripts\activate
      ```
    - For macOS/Linux:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Make sure you have the trained model (`best_model.pkl`) and scaler (`scaler.pkl`) in the root directory, or train the model by running the `model.py` file.

## Running the Application

To run the Flask application, use the following comman
