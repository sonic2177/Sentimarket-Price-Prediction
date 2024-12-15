import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import requests

# Title and Sidebar Information
st.title('Stock Price Predictions with Sentiment Analysis')
st.sidebar.info('Welcome to the Sentimarket-Price-Prediction App. Choose your options below.')


# Input for Stock Symbol
option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY').upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration (days)', value=3000, step=1)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End Date', value=today)

# Load and Cache Data
@st.cache_resource
def download_data(stock_symbol, start_date, end_date):
    return yf.download(stock_symbol, start=start_date, end=end_date, progress=False)

data = download_data(option, start_date, end_date)
scaler = StandardScaler()

# Main Navigation
def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize', 'Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()

# Technical Indicators
def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    # Calculate indicators
    bb_indicator = BollingerBands(data.Close)
    bb = data.copy()
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    macd = MACD(data.Close).macd()
    rsi = RSIIndicator(data.Close).rsi()
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Close':
        st.line_chart(data.Close)
    elif option == 'BB':
        st.line_chart(bb[['Close', 'bb_h', 'bb_l']])
    elif option == 'MACD':
        st.line_chart(macd)
    elif option == 'RSI':
        st.line_chart(rsi)
    elif option == 'SMA':
        st.line_chart(sma)
    elif option == 'EMA':
        st.line_chart(ema)

# Recent Data
def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))

# Sentiment Analysis (Basic Implementation)
def fetch_sentiment(stock):
    st.header("Market Sentiment Analysis")
    try:
        url = f'https://newsapi.org/v2/everything?q={stock}&apiKey=YOUR_NEWS_API_KEY'
        response = requests.get(url)
        articles = response.json()['articles'][:5]

        sentiment_scores = []
        for article in articles:
            sentiment_scores.append(np.random.uniform(-1, 1))  # Replace with actual sentiment model

        avg_sentiment = np.mean(sentiment_scores)
        sentiment_label = "Positive" if avg_sentiment > 0 else "Negative"

        st.write(f"Average Sentiment Score: {avg_sentiment:.2f} ({sentiment_label})")
        for article in articles:
            st.write(f"- {article['title']} ({article['publishedAt']})")

    except Exception as e:
        st.error("Error fetching sentiment data. Make sure the API is set up.")

# Prediction and Model Comparison
def predict():
    st.header("Stock Price Prediction and Model Comparison")
    num = st.number_input('How many days to forecast?', value=5, step=1)
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'Extra Trees': ExtraTreesRegressor(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'XGBoost': XGBRegressor()
    }

    if st.button('Predict and Compare Models'):
        results = compare_models(models, num)
        st.subheader("Model Comparison Results")
        st.dataframe(results)

        best_model_name = results.loc[results['R² Score'].idxmax(), 'Model']
        st.success(f"The best model is: **{best_model_name}** with the highest R² Score.")

        st.subheader(f"Forecast with {best_model_name}")
        best_model = models[best_model_name]
        forecast_predictions = forecast_with_model(best_model, num)
        for i, pred in enumerate(forecast_predictions, 1):
            st.text(f"Day {i}: {pred:.2f}")

def compare_models(models, num):
    df = data[['Close']]
    df['preds'] = data.Close.shift(-num)

    x = scaler.fit_transform(df.drop(['preds'], axis=1).values)
    x_forecast = x[-num:]
    x, y = x[:-num], df['preds'].dropna().values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

    results = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)

        results.append({
            'Model': model_name,
            'R² Score': r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        })

    return pd.DataFrame(results).sort_values(by='R² Score', ascending=False)

def forecast_with_model(model, num):
    df = data[['Close']]
    df['preds'] = data.Close.shift(-num)

    x = scaler.fit_transform(df.drop(['preds'], axis=1).values)
    x_forecast = x[-num:]
    return model.predict(x_forecast)

if __name__ == '__main__':
    fetch_sentiment(option)
    main()
