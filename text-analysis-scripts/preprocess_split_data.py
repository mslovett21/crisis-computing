import os
import pandas as pd
import numpy as np
from shutil import copy
import glob
import nltk
from nltk.corpus import stopwords
from itertools import chain
nltk.download('stopwords')
import re


#access data folders for csv
out_path = r'../data/tweets_csv'
INFORMATIVE_TWEETS = out_path + "/INFORMATIVE_TWEETS/"
NONINFORMATIVE_TWEETS = out_path + "/NONINFORMATIVE_TWEETS/"

INFORMATIVE_FILES = glob.glob(INFORMATIVE_TWEETS + "/*.csv")
NONINFORMATIVE_FILES = glob.glob(NONINFORMATIVE_TWEETS + "/*.csv")

def read_data(files):
    dfs = []
    for file in files:
        df = pd.read_csv(file, error_bad_lines=False)
        dfs.append(df)
    final_df = pd.concat(dfs)
    return final_df   



def create_tweets_df():
    info_df = read_data(INFORMATIVE_FILES)
    noninfo_df = read_data(NONINFORMATIVE_FILES)
    info_df["text_info"] = 1
    noninfo_df["text_info"] = 0
    final_df = pd.concat([info_df, noninfo_df])
    return final_df



def preprocess_tweets(tweet):
    tweet = str(tweet)
    # Removing URL mentions
    tweet = ' '.join(re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet).split())
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
    # Removing stopwords
    stop  = stopwords.words('english')
    tweet =' '.join([word for word in tweet.split() if word not in (stop)])
    # Removing punctuations
    tweet = tweet.replace('[^\w\s]','') 
    tweet = tweet.lower()
    return tweet


# over all 12,762 tweets, out of this 11,407 unique tweets
# informative 8463, noninformative 4299

if __name__ == '__main__':
    try:
        tweets_df = create_tweets_df()
        tweets_df['tweet_text'] = tweets_df.apply(lambda x: preprocess_tweets(x['tweet_text']), axis= 1)
        tweets_df = tweets_df.sample(frac = 1) 
        num_tweets = len(tweets_df)
        trainset_size,valset_size, testset_size  = int(num_tweets * 0.7), int(num_tweets * 0.15),int(num_tweets * 0.15)
        train, validate, test = np.split(tweets_df, [trainset_size,valset_size])

        PATH = "../data/final_tweets/"
        os.mkdir(PATH)
        tweets_df.to_csv(PATH +"full_tweets_df.csv" ,index = False )
        train.to_csv(PATH +"train_df.csv" ,index = False )
        validate.to_csv(PATH +"validate_df.csv" ,index = False )
        test.to_csv(PATH + "test_df.csv" ,index = False )

    except Exception as e:
        print("Exception Occured::\n", e)