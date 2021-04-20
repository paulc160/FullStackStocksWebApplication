from psaw import PushshiftAPI
import datetime

api = PushshiftAPI()

start_epoch=int(datetime.datetime(2021, 3, 1).timestamp())

submissions1 = list(api.search_submissions(after=start_epoch,subreddit='wallstreetbets',filter=['url','author','title','subreddit'], limit=10))

print(submissions1)


