# Olfi - Analysis of Social Media using Multimodal Deep Learning for Disaster Response

Learning from different modalities leads to more robust inference compared to learning from a single modality.

## Goals 
Use text + image data from Twitter to learn:
  * whether a tweet is informative for humanitarian aid
  * whether it contains some useful info such as report of injured or deceased people, destroyed infrastracture

## Methods

  * 2 separate learning classification tasks
  * use multimodal deep learning techniques
  * use early fusion techniques
  * learn a joint representation from two parallel depp learning architectures where oone architecture represent the text modality and the other architecture represents the image modality.

Reminder: Fusion can be performed on a feature (EARLY) or a decision level (LATE).
To learn about fusion in similar settings read (audi-visual data - Nagrani et al. 2018, audio-text Chowdhury et al. 2019). 

## Architecture
* **Images:** feature extraction - VGG16
* **Text:** feature extraction - CNN (5 hidden layers)
* **Fusion:** two feature vectors obtained are fed into shared representation followed by a dense layer before performing a prediction using softmax.

## Classification Experiments:
* only images,
* only text,
* both image + text

## Baseline:
* models trained using single modality

## Results: 
* Informativness classification task -> F1-score = 84.2
* Humanitarian classification task -> F1-score 78.3.

## Comments:
This is the first study that presents baseline results on CrisisMMD using state-of-the-art deep learning- based unimodal and multimodal approaches in one place. Guidence for future research using the CrisisMMD dataset.


## Dataset Details:
Crisis MMD dataset consists of tweets and associated images collected during seven different natural disasters and are annotated for three tasks:
1. informative vs not-informative,
2. humanitarian categories (8 classes), (merge them into 5 bigger categories)
3. demage severity (three classes),

## Data Split
* 70 % training, 15% validation, 15% test

## Metrics
1. accuracy, precision, recall and F1-score

## Challenges and Future Work

Social media data are not warranted to havve strong alignment or coupling between co-occuring text and image pairs. It is important NOT to assume the existence of strong correspondances between social media text and images, All existing classification approaches assume that there always exists a commmon label for data coming from different modalities. challenging future direction: design multimodal learning algorithm that can be also trained on heterogeneous input: tweet text and image pairs wit disagreeing labels, in which case CrisisMMD can be used in full form.




========

# Gautam - Multimodal Analysis of Disaster Tweets

## Contributions:

* Analyze the accuracy of pre-trained CNN architectures on CrisisMMD for classification of Images.
* Evaluate the accuracy of standard text features and deep learning methods for classification of tweet text in Crisis MMD.
* Combination of both text and image-based modality for identification of disaster informative tweets in the CrisisMMD.


## Labels used
* informative, non informative or cannot judge.


