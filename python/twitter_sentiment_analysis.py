'''
	This code is an example of sentiment analysis via Twitter API.
	To get access to the API, please create an application:
	https://apps.twitter.com/app/new

Dependencies:
	tweepy (http://www.tweepy.org/)
	textblob (https://textblob.readthedocs.io/en/dev/)

References:
	https://github.com/arnauddelaunay/twitter_sentiment_challenge/blob/master/demo.py
	https://github.com/llSourcell/twitter_sentiment_challenge
'''

import tweepy
from textblob import TextBlob

# Step 1 - Authenticate

consumer_key= 'YOUR_CONSUMER_KEY'
consumer_secret= 'YOUR_CONSUMER_SECRET'

access_token='YOUR_ACCESS_TOKEN'
access_token_secret='YOUR_TOKEN_SECRET'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# List of topics to search on twitter
topics = ['Bitcoin', 'Ethereum', 'Trump', 'Worldcup']
# Period
since_date = "2017-01-01"
until_date = "2018-05-10"

# Function returning the result of analysis
def get_label(analysis, threshold = 0):
	if analysis.sentiment[0]>threshold:
		return 'Positive'
	else:
		return 'Negative'

# Retrieve Tweets and Save Them
path = '../data/tweets/'
for topic in topics:
    this_topic_polarities = []
    # Get the tweets about the topics between the dates
    this_topic_tweets = api.search(q = topic, count=100, since = since_date, until=until_date)
    # Save the tweets in csv
    with open(path+topic+'_tweets.csv', 'w') as topic_file:
        topic_file.write('tweet, sentiment_label\n')
        for tweet in this_topic_tweets:
            analysis = TextBlob(tweet.text)
            # Get the label corresponding to the sentiment analysis
            this_topic_polarities.append(analysis.sentiment[0])
            topic_file.write('{},{}\n'.format(tweet.text.encode('utf8'), get_label(analysis)))
