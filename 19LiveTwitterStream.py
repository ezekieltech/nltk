import sentiment_mod as s
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import time


#consumer key, consumer secret, access token, access secret.
ckey="M6Dp0pEqAWPFefB9bsJzswjmj"
csecret="UdR1HPtpVZKlm3IxKQBaVDa9jA5JIQ4dYySJwH7ReU7fL3isFJ"
atoken="2253337291-vyzhmxOTj1V6A5q5pDg2P147BDgNgsbOUigQeMl"
asecret="RZFKMbaDOErCDQiqkHQ4stChtRkWP4U8PEBhv8M6TrKEV"

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        
        #tweet = all_data["text"]
        tweet = ascii(all_data["text"])

        #tweet = tweet.encode('utf-8', errors ='ignore')
        sentiment_value, confidence = s.sentiment(tweet)

        print(tweet, sentiment_value, confidence)
        time.sleep(0.2)
        if confidence * 100 >= 80:
            output = open('twitter-out.txt', 'a')
            output.write(sentiment_value)
            output.write('\n')
            output.close()
        
        return True

    def on_error(self, status):
        print (status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["car"])
##maybe you can try to limit the language of tweets into English
##twitterStream.filter(languages=[&quot;en&quot;],track=[&quot;trump&quot;])﻿
