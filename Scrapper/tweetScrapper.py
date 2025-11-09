import pandas as pd
import tweepy

#Read the authentication data for the twitter account
auth_token = pd.read_csv("../CSV/auth_twitter.csv")

#Pass the auth data to authenticate
auth = tweepy.Client(bearer_token=auth_token["token"][2], consumer_key=auth_token["token"][0], consumer_secret=auth_token["token"][1], access_token=auth_token["token"][3], access_token_secret=auth_token["token"][4], wait_on_rate_limit=True)

def searchTweet():
    # Set the query string (Search the tweets we need)
    query = '#emotion -is:retweet lang:en'

    #Scrap the tweets by the tweepy API
    tweet_posts = auth.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=100)
    scrap_tweets = pd.DataFrame(tweet_posts[0], columns=["text"])

    scrap_tweets.to_csv("tweets_scrapped2.csv")

searchTweet()