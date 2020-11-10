# TEXT ANALYSIS

## Create Dataset of valid ("paired") informative and noninformative tweets.
create_tweets_dataset.py

```python
python create_tweets_dataset.py

# choosing 'valid subset'
df_info = df.loc[(df['image_info'] == 'informative') & (df['text_info'] == 'informative')]
df_noninfo = df.loc[(df['image_info'] == 'not_informative') & (df['text_info'] == 'not_informative')]

```


## Preprocess Data and Split into Train, Validation and Test.
preprocess_split_data.py

This results in a dataset same as what Gautam2019 calls "CleanCrisisMMD"
* 8463 informative pairs (tweet + image)
* 4299 noninformative pairs (tweet + image)

The files that it produces:
* full_tweets_df.csv  (all of the tweets: 12,762 tweets, out of this 11,407 unique tweets )
* test_df.csv  (8933 tweets)
* validate_df.csv (1914 tweets)
* test_df.csv (1914 tweets)

```python
python preprocess_split_data.py

```


- [ ] Figure out how to get the same dataset as Olfi2020


## Preprocess the Tweets - we should follow Olfi2020

### Olfi2020
* remove stop words,
* remove non-ASCII characters,
* remove numbers,
* remove URLs,
* remove hashtag signs,
* replace all punctuation with white-space


### Gautam2019
* remove stop words,
* remove all user mentions and URLs,
* remove hashtags if the size is bigger than 8 characters,
* replace all punctuation with white-space,
* remove the tweet if the length of the tweet is less than three words

### Tasks
- [ ] Try Bidirectional LSTM (Nachiket)
- [ ] Email and request access to Crisis2Vec embeddings (Patrycja)
- [ ] Find architecture to train for text classification/ text feature extraction.
- [ ] Semi-supervised text/tweet labeling
- [ ] Contrastive learning for binary classification of tweets
