# TEXT ANALYSIS

## Create Dataset of valid ("paired") informative and noninformative tweets.
create_tweets_dataset.py

```python
python create_tweets_dataset.py

```

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
