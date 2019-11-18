
# coding: utf-8

# 11-18-2019
# 
# This data originally came from Crowdflower's Data for Everyone library (https://www.figure-eight.com/data-for-everyone/)
# 
# As the original source says,
# "A sentiment analysis job about the problems of each major U.S. airline. Twitter data was scraped from February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as "late flight" or "rude service")."
# 
# The data contains 14,640 tweets covering six U.S. airlines for the period 2/16/2015 to 2/24/2015. This kernel is a simple exploration of the data and does not analyze tweet text. Sentiment labels are taken as given.

# In[416]:

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[417]:

TWITTER_PATH = '~/Desktop/datasets/'


# In[418]:

def load_twitter_data(twitter_path=TWITTER_PATH):
    csv_path = os.path.join(twitter_path, "Tweets.csv")
    return pd.read_csv(csv_path)


# In[419]:

tweets = load_twitter_data()
tweets.head(5)


# In[420]:

#quick look at distributions of sentiment, airlines, and negative reasons:

tweets['airline_sentiment'].value_counts()


# In[494]:

tweets['airline'].value_counts()


# In[421]:

tweets['negativereason'].value_counts()


# In[422]:

#quick look at null values

def print_null_values(df):
    total = df.isnull().sum().sort_values(ascending = False)
    total = total[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = total / len(df) * 100
    percent = percent[df.isnull().sum().sort_values(ascending = False) != 0]
    concat = pd.concat([total, percent], axis=1, keys=['Total','Percent'])
    print (concat)
    print ( "-------------")


# In[423]:

printNullValues(tweets)


# In[424]:

#dropping meaningless columns

nullcols = ['negativereason_gold','airline_sentiment_gold']
tweets = tweets.drop(nullcols, axis=1)


# ANALYSIS
# 
# First I want some basic comparative information. For each airline, I would like to know:
# (1) how many mentions on Twitter
# (2) distribution of positive/neutral/negative tweets
# (3) drivers of negative tweets
# 
# I will end with a look at how sentiment changes throughout the week.

# Now we are going to plot the distribution of tweet sentiment for all airlines...

# In[426]:

AIRLINE = 'US Airways'


# In[427]:

def get_airline_sentiment(airline=AIRLINE):
    airline_tweets = tweets[tweets['airline']==airline]
    airline_sentiment = airline_tweets['airline_sentiment'].value_counts() / len(airline_tweets) * 100
    airline_sentiment.set_value('airline',  airline)
    return (airline_sentiment)


# In[428]:

get_airline_sentiment()


# In[429]:

sentiments = pd.DataFrame(columns = ['airline', 'positive', 'neutral', 'negative']) 

for airline in tweets['airline'].unique():
    sentiments = sentiments.append(get_airline_sentiment(airline), ignore_index=True)

sentiments


# In[430]:

ind = np.arange(6)
negative = sentiments['negative']
neutral = sentiments['neutral']
positive = sentiments['positive']
barwidth = .85

p1 = plt.bar(ind, negative, color = '#f9bc86', edgecolor='white', width=barwidth)
p2 = plt.bar(ind, neutral, bottom=negative, color='#a3acff', edgecolor='white', width=barwidth)
p3 = plt.bar(ind, positive, bottom=[i+j for i,j in zip(negative, neutral)], color='#b5ffb9', edgecolor='white', width=barwidth)

plt.ylabel('Percentage of Tweets')
plt.title('Tweet Sentiment by Airline')
plt.xticks(ind, sentiments['airline'])
plt.legend((p1[0], p2[0], p3[0]), ('Negative', 'Neutral', 'Positive'))
plt.show()


# I wonder what people are so upset about...

# In[445]:

def get_airline_tweets(airline):
    airline_tweets = tweets[tweets['airline'] == airline]
    return airline_tweets

for airline in tweets['airline'].unique():
    airline_tweets = get_airline_tweets(airline)
    airline_neg_tweets = airline_tweets[airline_tweets['airline_sentiment'] == 'negative']
    print(airline)
    print("-------------")
    print(airline_neg_tweets['negativereason'].value_counts() / len(airline_neg_tweets) * 100)
    print("-------------")


# Not surprised to see that the primary driver of negative tweets for American is customer service issues...
# 
# Next, I think it will be fun to use the coordinates fields for some geo visualization. Note- Only ~7% of data points have coordinates data, and many of those are invalid: (0.0,0.0), so this may not work out.

# In[431]:

#split out coordinates data and drop nulls
coords = tweets['tweet_coord'].str.strip('[]')                                         .str.split(', ', expand=True)                                      .rename(columns={0:'latitude', 1:'longitude'}).dropna()
        
#replace tweet_coord with new coordinates
tweets = pd.concat([tweets[:], coords[:]], axis=1)
tweets.drop('tweet_coord', axis=1)

tweets['longitude'] = tweets['longitude'].astype(float).dropna()
tweets['latitude'] = tweets['latitude'].astype(float).dropna()


# In[446]:

"""
#having trouble importing geopandas after conda install and conda-forge install...

from shapely.geometry import Point
import geopandas
from geopandas import GeoDataFrame

geometry = [Point(xy) for xy in zip(tweets['longitude'], tweets['latitude'])]
gdf = GeoDataFrame(df, geometry=geometry)   

#this is a simple map that goes with geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15);

"""


# In[448]:

#I will use simple scatter plots since I can't use geopandas

xmin = -150
xmax = 130
ymax = 60
ymin = 0

positive_tweets = tweets[tweets['airline_sentiment'] == 'positive']
positive_tweets.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.title("Positive Tweets Around the World")

negative_tweets = tweets[tweets['airline_sentiment'] == 'negative']
negative_tweets.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.title("Negative Tweets Around the World")

plt.show()


# In[451]:

delta_tweets = get_airline_tweets('Delta')
aa_tweets = get_airline_tweets('American')

sns.lmplot( x="longitude", y="latitude", data=delta_tweets, fit_reg=False, hue='airline_sentiment', legend=False)
plt.legend(loc='lower left')
plt.title("Geography and Sentiment of Delta Airlines Tweets")

sns.lmplot( x="longitude", y="latitude", data=aa_tweets, fit_reg=False, hue='airline_sentiment', legend=False)
plt.legend(loc='lower left')
plt.title("Geography and Sentiment of American Airlines Tweets")

plt.show()


# In[459]:

#Easier to use the col argument to split the data out by airline. 
#This also keeps the x and y limits constant for easier visual comparison.

sns.lmplot( x="longitude", y="latitude", data=tweets, fit_reg=False, hue='airline_sentiment', col="airline")

plt.show()


# In[460]:

#We can do the same with sentiment. Much simpler than in 448.

sns.lmplot( x="longitude", y="latitude", data=tweets, fit_reg=False, hue='airline', col="airline_sentiment")

plt.show()


# Working with coordinates did not pan out. Because I don't have maps and the coordinates data is so sparse, I don't want to go any further.

# Next I will look at sentiment across time of day.

# In[478]:

#Let's look at tweet volume for each hour of the day (over 7 days)

tweets['tweet_created'] = pd.to_datetime(tweets['tweet_created'])

tweetsperhour = tweets['tweet_created']

res = tweetsperhour.groupby(tweets['tweet_created'].dt.hour).count().reindex(np.arange(24),fill_value=0)
res.plot(kind='bar')
plt.show()


# In[493]:

#I can show the same with sentiment distribution

col_list = ['tweet_created', 'airline_sentiment']
tweetsperhour = tweets[col_list]

negative = tweetsperhour['tweet_created'][tweetsperhour['airline_sentiment']=='negative']         .groupby(tweets['tweet_created'].dt.hour).count()
neutral = tweetsperhour['tweet_created'][tweetsperhour['airline_sentiment']=='neutral']         .groupby(tweets['tweet_created'].dt.hour).count()
positive = tweetsperhour['tweet_created'][tweetsperhour['airline_sentiment']=='positive']         .groupby(tweets['tweet_created'].dt.hour).count()

ind=np.arange(24)
barwidth=.85

p1 = plt.bar(ind, negative, color = '#f9bc86', edgecolor='white', width=barwidth)
p2 = plt.bar(ind, neutral, bottom=negative, color='#a3acff', edgecolor='white', width=barwidth)
p3 = plt.bar(ind, positive, bottom=[i+j for i,j in zip(negative, neutral)], color='#b5ffb9', edgecolor='white', width=barwidth)

plt.title('Tweet Sentiment and Volume by Hour')
plt.legend((p1[0], p2[0], p3[0]), ('Negative', 'Neutral', 'Positive'))
plt.show()


# Note that the tweet_created times for all tweets seems to be is based on a single timezone (UTC -0800 ?), so we are agnostic to the time of day in the Twitter user's location. It could be interesting to look at how sentiment changes throughout the day based on the twitter user's timezone.
# 
# To do this, we would convert user_timezone to a number of hours offset from UTC -0800, add this number to the tweet_created column, then run this same visualization. We would also need to solve for the 32% null values in user_timezone.

# In[ ]:



