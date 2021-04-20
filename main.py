from fastapi import FastAPI, Request, Form, HTTPException
import sqlite3, config
from fastapi.templating import Jinja2Templates
from datetime import date
import datetime
from fastapi.responses import RedirectResponse
import yfinance as yf
import pandas as pd
from plotly import graph_objs as go 
from fastapi.responses import JSONResponse
from statsmodels.tsa.stattools import adfuller
import tweepy
from textblob import TextBlob
import statistics
import numpy as np
from finvizfinance.quote import finvizfinance
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob




app = FastAPI()
templates = Jinja2Templates(directory="templates")

consumer_key='4hZQpGGEiaz1MGfpwl1ZkoKHs'
consumer_secret='m1M7DOOgJcGtvI87jIFgCcToYIwEeXZYZGVstp0fjJglFP2LXq'
access_token_key='493637502-Mk3univd84qV3w15PAdgELGldT3cWaQ5QvAPMmhB'
access_token_secret='4Bd2Dt4eW26fx1o2bn6F9M4rfFWhFG8qbdweYKnIl8kzC'

@app.get("/")
def index(request: Request):
    stock_filter = request.query_params.get('filter', False)

    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    if stock_filter == 'new_closing_highs':
        cursor.execute("""
        select * from (
            select symbol, name, stock_id, max(close), date
            from stock_price join stock on stock.id = stock_price.stock_id
            group by stock_id
            order by symbol
        ) where date = (select max(date) from stock_price)
        """)
    elif stock_filter == 'new_closing_lows':
        cursor.execute("""
        select * from (
            select symbol, name, stock_id, min(close), date
            from stock_price join stock on stock.id = stock_price.stock_id
            group by stock_id
            order by symbol
        ) where date = (select max(date) from stock_price)
        """)
    elif stock_filter == 'rsi_overbought':
        cursor.execute("""
            select symbol, name, stock_id, date
            from stock_price join stock on stock.id = stock_price.stock_id
            where rsi_14 > 70
            AND date = (select max(date) from stock_price)
            order by symbol
        """)
    
    elif stock_filter == 'rsi_oversold':
        cursor.execute("""
            select symbol, name, stock_id, date
            from stock_price join stock on stock.id = stock_price.stock_id
            where rsi_14 < 30
            AND date = (select max(date) from stock_price)
            order by symbol
        """)
    elif stock_filter == 'above_sma_20':
        cursor.execute("""
            select symbol, name, stock_id, date
            from stock_price join stock on stock.id = stock_price.stock_id
            where close > sma_20
            AND date = (select max(date) from stock_price)
            order by symbol
        """)
    elif stock_filter == 'below_sma_20':
        cursor.execute("""
            select symbol, name, stock_id, date
            from stock_price join stock on stock.id = stock_price.stock_id
            where close < sma_20
            AND date=(select max(date) from stock_price)
            order by symbol
        """)

    elif stock_filter == 'above_sma_50':
        cursor.execute("""
            select symbol, name, stock_id, date
            from stock_price join stock on stock.id = stock_price.stock_id
            where close > sma_50
            AND date = (select max(date) from stock_price)
            order by symbol
        """)
    elif stock_filter == 'below_sma_50':
        cursor.execute("""
            select symbol, name, stock_id, date
            from stock_price join stock on stock.id = stock_price.stock_id
            where close < sma_50
            AND date=(select max(date) from stock_price)
            order by symbol
        """)
     
    else:
        cursor.execute("""
            SELECT id, symbol, name FROM stock ORDER BY symbol
        """)

    rows = cursor.fetchall()

    current_date = date.today().isoformat()

    cursor.execute("""
        select symbol, rsi_14, sma_20, sma_50, close
        from stock join stock_price on stock_price.stock_id = stock.id
        where date = (select max(date) from stock_price)
    """)

    indicator_rows = cursor.fetchall()

    indicator_values = {}

    for row in indicator_rows:
        indicator_values[row['symbol']] = row

    return templates.TemplateResponse("index.html", {"request": request, "stocks": rows, "indicator_values": indicator_values})

@app.get("/stock/{symbol}")
def stock_detail(request: Request, symbol):
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    cursor.execute("""
        SELECT * FROM strategy
    """)
    strategies = cursor.fetchall()

    cursor.execute("""
        SELECT id, symbol, name FROM stock WHERE symbol = ?
    """,(symbol,))
    row = cursor.fetchone()
    cursor.execute("""
        SELECT * FROM stock_price WHERE stock_id = ? ORDER BY date DESC
    """, (row['id'],))

    prices = cursor.fetchall()

    tickersearch = yf.Ticker(symbol)
    beta = tickersearch.info['beta']
    MarketCap = tickersearch.info['marketCap']
    Volume = tickersearch.info['volume']
    High52 = tickersearch.info['fiftyTwoWeekLow']
    Low52 = tickersearch.info['fiftyTwoWeekHigh']
    AverageVol = tickersearch.info['averageDailyVolume10Day']
    Trailpe = tickersearch.info['trailingPE']
    Forwardpe = tickersearch.info['forwardPE']
    sharesf = tickersearch.info['floatShares']
    sharesS = tickersearch.info['sharesShort']
    shortRatio = tickersearch.info['shortRatio']
    institutions = tickersearch.info['heldPercentInstitutions']
    ptob = tickersearch.info['priceToBook']
    peg = tickersearch.info['pegRatio']
    summary = tickersearch.info['longBusinessSummary']
    meanticker = yf.Ticker(symbol)
    meanticker_hist = meanticker.history(period="1y")
    mean = statistics.mean(meanticker_hist['Close'])
    variance = statistics.variance(meanticker_hist['Close'])
    std = statistics.stdev(meanticker_hist['Close'])

    spy = yf.Ticker("SPY")
    spy_hist = spy.history(period="1y")
    spy_hist = spy_hist.pct_change()
    meanticker_histnew = meanticker_hist.pct_change()
    correlation = spy_hist['Close'].corr(meanticker_histnew['Close'])
    return templates.TemplateResponse("stock_detail.html", {"request": request, "stock": row, "bars": prices, "strategies": strategies,"betas": beta,"MC": MarketCap,"Volume": Volume,"High52": High52,
    "Low52": Low52,"Volume10day": AverageVol,"Trailpe": Trailpe,"Forwardpe": Forwardpe,"sharesf": sharesf,"sharesS": sharesS,"shortRatio": shortRatio,"Institutions": institutions,"ptob": ptob,"peg": peg,"summary": summary,"mean": mean,"variance": variance,"std": std,"correlation": correlation})

@app.post("/apply_strategy")
def apply_strategy(strategy_id: int = Form(...), stock_id: int = Form(...)):
    connection = sqlite3.connect(config.DB_FILE)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO stock_strategy (stock_id, strategy_id) VALUES (?, ?)
    """, (stock_id, strategy_id))

    connection.commit()

    return RedirectResponse(url=f"/strategy/{strategy_id}", status_code=303)

@app.get("/orders")
def orders(request: Request):
    return templates.TemplateResponse("orders.html", {"request": request})

@app.get("/strategies")
def strategies(request: Request):
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute("""
        SELECT * FROM strategy
    """)
    strategies = cursor.fetchall()

    return templates.TemplateResponse("strategies.html", {"request": request, "strategies": strategies})

@app.get("/strategy/{strategy_id}")
def strategy(request: Request,strategy_id):
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, name
        FROM strategy
        WHERE id = ?
    """, (strategy_id,))

    strategy = cursor.fetchone()

    cursor.execute("""
        SELECT symbol, name
        FROM stock JOIN stock_strategy on stock_strategy.stock_id = stock.id
        WHERE strategy_id = ?
    """, (strategy_id,))

    stocks = cursor.fetchall()

    return templates.TemplateResponse("strategy.html", {"request": request, "stocks": stocks, "strategy": strategy})

@app.get("/wsb_tracker")
def wsb(request: Request):
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute("""
        select count(*) as num_mentions, stock_id, symbol,dt,stock.name
        from mention join stock on stock.id = mention.stock_id
        where dt between '2021-02-11T00:00:00' and '2021-02-11T23:59:59'
        group by stock_id, symbol
        order by num_mentions DESC;
    """)
    wsbs = cursor.fetchall()
    
    api = PushshiftAPI()

    start_epoch=int(datetime.datetime(2021, 3, 17).timestamp())

    submissions = list(api.search_submissions(after=start_epoch,subreddit='wallstreetbets',filter=['url','author','title','subreddit'], limit=25))

    df = pd.DataFrame(submissions)

    vader = SentimentIntensityAnalyzer()
    scores = df['title'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)

    df = df.join(scores_df, rsuffix='_right')
    new_df = df[['title', 'compound']].copy()
    new_df.rename(columns={ new_df.columns[1]: "Sentiment Score" }, inplace = True)
    new_df.rename(columns={ new_df.columns[0]: "Post" }, inplace = True)
    html2 = new_df.to_html(classes='ui striped table')

    return templates.TemplateResponse("wsb_tracker.html", {"request": request, "wsbs": wsbs,"posts": html2})

@app.get('/prophet')
def read_form(request: Request):

    return templates.TemplateResponse("prophet.html", {"request": request})

@app.post('/prophet')
def predict_species(request: Request):

    
    
    return templates.TemplateResponse("prophet.html", {"request": request})

@app.get("/correlation")
def correlation(request: Request):
    tickerp1 = ""
    tickerp2 = ""
    
    return templates.TemplateResponse("correlation.html", {"request": request,"tickerp1": tickerp1,"tickerp2": tickerp2})

@app.post("/correlation")
def correlation_form(request: Request,ticker1: str = Form(...),ticker2: str = Form(...)):
    tickerp1 = ticker1
    tickerp2 = ticker2
    corticker1 = yf.Ticker(tickerp1)
    corticker2 = yf.Ticker(tickerp2)
    corticker1_hist = corticker1.history(period="1y")
    corticker2_hist = corticker2.history(period="1y")
    corticker1_dataframe = corticker1_hist.pct_change()
    corticker2_dataframe = corticker2_hist.pct_change()
    correlation = corticker1_dataframe['Close'].corr(corticker2_dataframe['Close'])
    return templates.TemplateResponse("correlation.html", {"request": request,"tickerp1": tickerp1,"tickerp2": tickerp2,"correlation": correlation})

@app.get("/stationary")
def stationarity(request: Request):
    stationaryticker = ""
    return templates.TemplateResponse("stationarity.html", {"request": request,"stationaryticker": stationaryticker})

@app.post("/stationary")
def stationarity_form(request: Request,sticker: str = Form(...)):
    stationaryticker = sticker
    stticker1 = yf.Ticker(stationaryticker)
    stticker1_hist = stticker1.history(period="1y")
    data = adfuller(stticker1_hist['Close'])
    adfstatistic = data[0]
    pvalue = data[1]
    criticalvalue1 = data[4]['1%']
    criticalvalue5 = data[4]['5%']
    criticalvalue10 = data[4]['10%']
    return templates.TemplateResponse("stationarity.html", {"request": request,"stationaryticker": stationaryticker,"adfstatistic":adfstatistic,"pvalue":pvalue,"criticalvalue1":criticalvalue1,"criticalvalue5":criticalvalue5,"criticalvalue10":criticalvalue10})

@app.get("/sentiment")
def sentiment(request: Request):
    sentimenttickers = ""
    return templates.TemplateResponse("sentiment.html", {"request": request,"sentimenttickers": sentimenttickers})

@app.post("/sentiment")
def sentiment_form(request: Request,sentimentticker: str = Form(...)):
    sentimenttickers = sentimentticker
    stock = finvizfinance(sentimenttickers)
    news_df = stock.TickerNews()
    news_df.style.set_properties(**{'text-align': 'left'})
    html = news_df.to_html(classes='ui striped table')
    vader = SentimentIntensityAnalyzer()
    scores = news_df['Title'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    news_df = news_df.join(scores_df, rsuffix='_right')
    news_df['Date'] = pd.to_datetime(news_df.Date).dt.date
    mean_scores = news_df.groupby(['Date']).mean()
    mean_scores = mean_scores.unstack()
    mean_scores = mean_scores.xs('compound').transpose()
    mean_scores = mean_scores.to_frame()
    mean_scores = mean_scores.reset_index()
    mean_scores.rename(columns={ mean_scores.columns[1]: "Sentiment Score" }, inplace = True)
    html2 = mean_scores.to_html(classes='ui striped table')
    dates = mean_scores['Date']
    scores = list(mean_scores['Sentiment Score'])
    return templates.TemplateResponse("sentiment.html", {"request": request,"sentimenttickers": sentimenttickers,"news":html,"date_scores":html2,"dates": dates,"scores":scores})

@app.get("/mean_reversion")
def mean_reversion(request: Request):
    stocks = pd.read_csv('C:\\Users\\pconn\\OneDrive\\Desktop\\Stock Picks\\PicksWeek22Feb.csv')
    stocks.style.set_properties(**{'text-align': 'left'})
    html = stocks.to_html(classes='ui striped table')
    return templates.TemplateResponse("mean_reversion.html", {"request": request,"stocks":html})

@app.get("/momentum_stocks")
def momentum_stocks(request: Request):
    stocks = pd.read_csv('C:\\Users\\pconn\\OneDrive\\Desktop\\Stock Picks\\momentum_stocks.csv')
    stocks.style.set_properties(**{'text-align': 'left'})
    html = stocks.to_html(classes='ui striped table')
    return templates.TemplateResponse("momentum_stocks.html", {"request": request,"stocks":html})

@app.get('/arima')
def arima_model(request: Request):

    return templates.TemplateResponse("arima.html", {"request": request})

@app.get("/disclaimer")
def disclaimer(request: Request):
    return templates.TemplateResponse("disclaimer.html", {"request": request})

@app.get("/contact")
def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

@app.get("/resources")
def resource(request: Request):
    return templates.TemplateResponse("resources.html", {"request": request})

@app.get("/home")
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/twitter_sentiment")
def twitter_sentiment(request: Request):
    twittersentimenttickers = ""
    return templates.TemplateResponse("twitter_sentiment.html", {"request": request,"twittersentimenttickers": twittersentimenttickers})

@app.post("/twitter_sentiment")
def twitter_sentiment_form(request: Request,twitterticker: str = Form(...)):
    twittersentimenttickers = twitterticker
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token_key, access_token_secret)
    api = tweepy.API(auth)
    def fetch_tweets(hashtag):
        tweet_user = []
        tweet_time = []
        tweet_string = []
        for tweet in tweepy.Cursor(api.search,q=hashtag, count=2000).items(2000):
            if (not tweet.retweeted) and ("RT @" not in tweet.text):
                if tweet.lang == "en":
                    tweet_user.append(tweet.user.name)
                    tweet_time.append(tweet.created_at)
                    tweet_string.append(tweet.text)

        df = pd.DataFrame({"username":tweet_user, "time": tweet_time, "tweet": tweet_string})
        return df
    df = fetch_tweets(twittersentimenttickers)
    df["sentiment"] = df["tweet"].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
    df_pos = df[df["sentiment"] > 0.0]
    df_neg = df[df["sentiment"] < 0.0]
    new_df = df[['tweet', 'sentiment']].copy()
    positive = len(df_pos)
    negative = len(df_neg)
    new_df.rename(columns={ new_df.columns[1]: "Sentiment Score" }, inplace = True)
    new_df.rename(columns={ new_df.columns[0]: "Tweet" }, inplace = True)
    html2 = new_df.head().to_html(classes='ui striped table')
    tickersearch = yf.Ticker(twittersentimenttickers)
    Close = tickersearch.info['previousClose']
    Open = tickersearch.info['regularMarketOpen']
    High = tickersearch.info['regularMarketDayHigh']
    Low = tickersearch.info['dayLow']
    Volume = tickersearch.info['volume']
    high52 = tickersearch.info['fiftyTwoWeekHigh']
    low52 = tickersearch.info['fiftyTwoWeekLow']
    data = tickersearch.history()
    last_quote = (data.tail(1)['Close'].iloc[0])
    
    return templates.TemplateResponse("twitter_sentiment.html", {"request": request,"twittersentimenttickers": twittersentimenttickers,"positive":positive,"negative":negative,"posts":html2,"Close":Close,"Open":Open,"High":High,"Low":Low,"Volume":Volume,"high52":high52,"low52":low52,"last_quote":last_quote})

