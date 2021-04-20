
from pathlib import Path
from datetime import date
import joblib
import pandas as pd
import yfinance as yf
from fbprophet import Prophet


def load_data(ticker):
    START = "2019-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet(daily_seasonality=True)
    m.fit(df_train)
    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)
    return forecast

