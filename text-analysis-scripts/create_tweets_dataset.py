import os
import pandas as pd
from shutil import copy
import glob


# CONSTANTS

# access dataset 
RELATIVE_PATH_ANNOTATIONS = r'../data/CrisisMMD_v2.0/annotations/'
ANNOTATION_FILES = glob.glob(RELATIVE_PATH_ANNOTATIONS + "/*.tsv")

#create folder for csv
out_path = r'../data/tweets_csv'
INFORMATIVE_TWEETS = out_path + "/INFORMATIVE_TWEETS/"
NONINFORMATIVE_TWEETS = out_path + "/NONINFORMATIVE_TWEETS/"

PATHS = [out_path, INFORMATIVE_TWEETS,NONINFORMATIVE_TWEETS]



def create_dics(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s" % path)
    return


def create_directory_structure(paths):
    for p in paths:
        create_dics(p)
    return


def save_tweets(filename, df_info, df_noninfo):
    df_info[["tweet_id", "text_info_conf","tweet_text"]].to_csv(INFORMATIVE_TWEETS+filename+"_info.csv", index=False)
    df_noninfo[["tweet_id", "text_info_conf","tweet_text"]].to_csv(NONINFORMATIVE_TWEETS+filename+"_noninfo.csv", index=False)
    return



def extract_tweets(ANNOTATION_FILES ):
    """
    Extract tweets from the dataset. Save the dataframes with inf and non-inf data
    :return: None
    """
    for ann_file in ANNOTATION_FILES:
        df = pd.read_csv(ann_file, sep='\t', error_bad_lines=False)
        # Considering only those images which have text of tweet and image in the tweet marked as Informative.
        df_info = df.loc[(df['image_info'] == 'informative') & (df['text_info'] == 'informative')]
        # Separation of Non-Informative Images
        df_noninfo = df.loc[(df['image_info'] == 'not_informative') & (df['text_info'] == 'not_informative')]
        filename = ann_file.split('/')[-1].split('.')[-2]
        save_tweets(filename, df_info, df_noninfo)
    return




if __name__ == '__main__':
    try:
        create_directory_structure(PATHS)
        extract_tweets(ANNOTATION_FILES)
    except FileNotFoundError:
        print("Please check if the CrisisMMD dataset has been extracted in the same directory structure level")
    except Exception as e:
        print("Exception Occured::\n", e)