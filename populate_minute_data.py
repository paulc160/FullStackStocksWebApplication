import config
import sqlite3
import pandas
import csv
import yfinance as yf
from datetime import datetime,timedelta
import alpaca_trade_api as tradeapi
from dateutil import tz

connection = sqlite3.connect(config.DB_FILE)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()

symbols = []
stock_ids = {}

cursor.execute("""
    SELECT * from stock
""")

stocks = cursor.fetchall()

for stock in stocks:
    symbol= stock['symbol']
    stock_ids[symbol] = stock['id']

for symbol in symbols:
    start_date = datetime(2021, 3, 20).date()
    end_date_range = datetime(2021, 3, 26).date()
    
    while start_date < end_date_range:
        end_date = start_date + timedelta(days=4)

        print(f"=== Fetching minute bars {start_date} - {end_date} for {symbol}")

        stock = yf.Ticker(symbol)
        minutes = stock.history(interval='1m',start=start_date,end=end_date)
        minutes = minutes.resample('1min').ffill()
        print(minutes)

        for index, row in minutes.iterrows():
            cursor.execute("""
                INSERT INTO stock_price_minute (stock_id, datetime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,(stock_ids[symbol], index.tz_localize(None).isoformat(), row['Open'], row['High'],row['Low'],row['Close'],row['Volume'] ))


connection.commit()