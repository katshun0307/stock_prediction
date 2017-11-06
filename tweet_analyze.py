""" stock_prediction / tweet_analyze.py :  """

"""
analyzes tweets for stock prediction
"""
__author__ = "Shuntaro Katsuda"

# encoding: utf-8
import json
import requests
import os
from requests_oauthlib import OAuth1

consumer_key = os.environ.get('CONSUMER_KEY')
consumer_secret = os.environ.get('CONSUMER_KEY_SECRET')
access_token = os.environ.get('ACCESS_TOKEN')
access_token_secret = os.environ.get('ACCESS_TOKEN_SECRET')


def getTimeline():
    """
    collect trump`s tweets
    :return: user's tweets in list
    """
    user_timeline_endpoint = "https://api.twitter.com/1.1/statuses/user_timeline.json"
    timeline_endpoint = "https://api.twitter.com/1.1/statuses/home_timeline.json"
    params = {'screen_name':'@realDonaldTrump', 'exclude_replies':True, 'include_rts': False}
    auth = OAuth1(consumer_key, consumer_secret, access_token, access_token_secret)
    response = requests.get(user_timeline_endpoint, params=params, auth=auth)
    return response.text


tweets = getTimeline()
tweets = json.loads(tweets)

#print(tweets[0]['text'])

