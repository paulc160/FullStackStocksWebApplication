from psaw import PushshiftAPI
import datetime
import sqlite3, config

connection = sqlite3.connect(config.DB_FILE)

connection.row_factory = sqlite3.Row

cursor = connection.cursor()

cursor.execute("""
    SELECT * FROM stock
""")

rows = cursor.fetchall()

print(rows)

stocks = {}
for row in rows:
    stocks['$' + row['symbol']] = row['id']

print(stocks)

api = PushshiftAPI()

start_epoch=int(datetime.datetime(2021, 2, 11).timestamp())

submissions = api.search_submissions(after=start_epoch,subreddit='wallstreetbets',filter=['url','author','title','subreddit'])

for submission in submissions:
    words = submission.title.split()
    cashtags = list(set(filter(lambda word: word.lower().startswith('$'), words)))

    if len(cashtags) > 0:
        #print(cashtags)
        #print()

        for cashtag in cashtags:

            submitted_time = datetime.datetime.fromtimestamp(submission.created_utc).isoformat()

            try:
                cursor.execute("""
                    INSERT INTO mention (dt, stock_id, message, source, url)
                    VALUES (?, ?, ?, 'wallstreetbets', ?)
                """, (submitted_time, stocks[cashtag], submission.title, submission.url))
                
                connection.commit()
            except Exception as e:
                print(e)
                connection.rollback()

