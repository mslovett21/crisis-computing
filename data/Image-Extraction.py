"""
 Following class is responsible for Image data extraction.
 Prior executing this class you will have to extract the CrisisMMD dataset on the same directory level
 as of this class.
 CrisisMMD dataset can be downloaded from : https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz
"""
import os
import os.path
import pandas as pd
import shutil
from shutil import copy
import numpy as np
from scipy import ndimage
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# CONSTANTS
ROOT = 'CrisisMMD_v2.0/'
INFORMATIVE = 'Informative'
NON_INFORMATIVE = 'Non-Informative/'
TRAINING = 'Training_data/'
TESTING = 'Testing_data/'
TRAINING_INFORMATIVE = TRAINING + INFORMATIVE
TRAINING_NON_INFORMATIVE = TRAINING + NON_INFORMATIVE
TESTING_INFORMATIVE = TESTING + INFORMATIVE
TESTING_NON_INFORMATIVE = TESTING + NON_INFORMATIVE
FIRE = INFORMATIVE + '/Fire'
FLOODS = INFORMATIVE + '/Floods'
EARTH = INFORMATIVE + '/Earthquake'
HURR = INFORMATIVE + '/Hurricane'
DONT_KNOW = '/dont-know-cant-judge'
LITTLE = '/little-or-no'
MILD = '/mild'
NAN = '/nan'
SEVERE = '/severe'
CATEGORIES_DICT = {
    'Fire': FIRE,
    'Earth-Q': EARTH,
    'Floods': FLOODS,
    'Hurricane': HURR
}
ANNOTATIONS = {
    'Fire': ['CrisisMMD_v2.0/annotations/california_wildfires_final_data.tsv'],
    'Earth-Q': ['CrisisMMD_v2.0/annotations/iraq_iran_earthquake_final_data.tsv',
                'CrisisMMD_v2.0/annotations/mexico_earthquake_final_data.tsv'],
    'Floods': ['CrisisMMD_v2.0/annotations/srilanka_floods_final_data.tsv'],
    'Hurricane': ['CrisisMMD_v2.0/annotations/hurricane_harvey_final_data.tsv',
                  'CrisisMMD_v2.0/annotations/hurricane_irma_final_data.tsv',
                  'CrisisMMD_v2.0/annotations/hurricane_maria_final_data.tsv']
}


def info_sep(category, df_informative):
    """
    Extracting the Informative Images to the specified category/class of the disaster severity
    :param category: category/class of disaster
    :param df_informative: Informative records.
    :return: None
    """
    for index, row in df_informative.iterrows():
        src = ROOT + row['image_path']
        if row['image_damage'] == 'severe_damage':
            dest = CATEGORIES_DICT[category] + SEVERE
            copy(src, dest)
        elif row['image_damage'] == 'mild_damage':
            dest = CATEGORIES_DICT[category] + MILD
            copy(src, dest)
        elif row['image_damage'] == 'dont_know_or_cant_judge':
            dest = CATEGORIES_DICT[category] + DONT_KNOW
            copy(src, dest)
        elif row['image_damage'] == 'little_or_no_damage':
            dest = CATEGORIES_DICT[category] + LITTLE
            copy(src, dest)
        else:
            dest = CATEGORIES_DICT[category] + NAN
            copy(src, dest)


def augment_images(files):
    datagen = ImageDataGenerator(rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')
    for file in files:
        img_path = ROOT + file
        if os.path.exists(img_path):
            image = np.expand_dims(ndimage.imread(img_path), 0)
            save_to = TRAINING_NON_INFORMATIVE
            datagen.fit(image)
            for x, val in zip(datagen.flow(image,  # image we chose
                                           save_to_dir=save_to,  # this is where we figure out where to save
                                           save_prefix='aug',
                                           save_format='png'), range(1)):
                pass


def non_info_sep(df_noninfo):
    """
    extract the Non-Informative Images
    :param df_noninfo: Non-Informative records
    :return: None
    """
    for index, row in df_noninfo.iterrows():
        src = ROOT + row['image_path']
        copy(src, NON_INFORMATIVE)


def create_severity_dirs(category):
    """
    Creating severity directories for the specified category
    :param category: category/class of disaster
    :return: None
    """
    os.mkdir(category + DONT_KNOW)
    os.mkdir(category + LITTLE)
    os.mkdir(category + MILD)
    os.mkdir(category + SEVERE)
    os.mkdir(category + NAN)


def create_directory_structure():
    """
    Creating File structure which will be required to store the extracted images.
    :return: None
    """
    if os.path.exists(INFORMATIVE) and os.path.isdir(INFORMATIVE):
        shutil.rmtree(INFORMATIVE)
    if os.path.exists(NON_INFORMATIVE) and os.path.isdir(NON_INFORMATIVE):
        shutil.rmtree(NON_INFORMATIVE)
    if os.path.exists(TRAINING) and os.path.isdir(TRAINING):
        shutil.rmtree(TRAINING)
    if os.path.exists(TESTING) and os.path.isdir(TESTING):
        shutil.rmtree(TESTING)
    os.mkdir(INFORMATIVE)
    os.mkdir(NON_INFORMATIVE)
    os.mkdir(TRAINING)
    os.mkdir(TESTING)

    os.mkdir(TRAINING_INFORMATIVE)
    os.mkdir(TRAINING_NON_INFORMATIVE)
    os.mkdir(TESTING_INFORMATIVE)
    os.mkdir(TESTING_NON_INFORMATIVE)

    for key, value in CATEGORIES_DICT.items():
        os.mkdir(value)
        create_severity_dirs(value)


def extract_images():
    """
    Extract images from the dataset. Further divide them into classes and severity
    :return: None
    """
    for key, values in ANNOTATIONS.items():
        print(key, '->', values)
        for value in values:
            df = pd.read_csv(value, sep='\t', error_bad_lines=False)
            # Considering only those images which have text of tweet and image in the tweet marked as Informative.
            df_informative = df.loc[(df['image_info'] == 'informative') & (df['text_info'] == 'informative')]
            # Separation of Non-Informative Images
            df_noninfo = df.loc[(df['image_info'] == 'not_informative') & (df['text_info'] == 'not_informative')]
            info_sep(category=key, df_informative=df_informative)
            non_info_sep(df_noninfo)


def get_Info_Non_Info_tweets(mainDataFrame, dataframe):
    """
    Method which will accept a dataframe and return all the informative and Non informative tweet images
    :param mainDataFrame: Main Dataset
    :param dataframe: Queried Dataset
    :return:
    """
    res = mainDataFrame[mainDataFrame['tweet_id'].isin(dataframe['tweet_id'])]
    info_res = res.loc[(res['text_info'] == 'informative') & (res['image_info'] == 'informative')]
    non_res = res.loc[(res['text_info'] == 'not_informative') & (res['image_info'] == 'not_informative')]
    return info_res['image_path'].tolist(), non_res['image_path'].tolist()


def copy_images(fnames, destination):
    """
    Method to copy the images to destination
    :param fnames: list of file names to be copied
    :param destination: path to destination folder
    :return:
    """
    Images_not_found = 0
    for path in fnames:
        src = ROOT + path
        if os.path.exists(src):
            copy(src, destination)
        else:
            Images_not_found = Images_not_found + 1
    print(Images_not_found, " Images not found for ", destination)


def create_dataset():
    """
    Method to split the dataset into training and testing w.r.t
    Informative and Non-Informative category.
    :return: None
    """
    # TODO: Code cleaning and modularity
    train = 'final_tweets/train_df.csv'
    val = 'final_tweets/validate_df.csv'
    test = 'final_tweets/test_df.csv'
    merged = 'CrisisMMD_v2.0/Merged.csv'
    merged_df = pd.read_csv(merged)

    drop_cols = ['text_info_conf', 'tweet_text']
    keep_cols = ['tweet_id', 'image_info', 'text_info', 'image_path']
    merged_df = merged_df[keep_cols]
    train_df = pd.read_csv(train)
    val_df = pd.read_csv(val)
    test_df = pd.read_csv(test)
    train_df = train_df.drop(drop_cols, axis=1)
    val_df = val_df.drop(drop_cols, axis=1)
    test_df = test_df.drop(drop_cols, axis=1)

    train_info, train_non_info = get_Info_Non_Info_tweets(merged_df, train_df)
    test_info, test_non_info = get_Info_Non_Info_tweets(merged_df, test_df)
    val_info, val_non_info = get_Info_Non_Info_tweets(merged_df, val_df)

    print('**********Training*************')
    print('Informative:', len(train_info) + len(val_info))
    print('Non-Informative:', len(train_non_info) + len(val_non_info))
    print('**********Testing*************')
    print('Informative:', len(test_info))
    print('Non-Informative:', len(test_non_info))

    copy_images(train_info, TRAINING_INFORMATIVE)
    copy_images(train_non_info, TRAINING_NON_INFORMATIVE)
    copy_images(val_info, TRAINING_INFORMATIVE)
    copy_images(val_non_info, TRAINING_NON_INFORMATIVE)
    copy_images(test_info, TESTING_INFORMATIVE)
    copy_images(test_non_info, TESTING_NON_INFORMATIVE)

    l = train_non_info[:1500] + val_non_info
    print("len of aug : ", len(l))
    augment_images(l)


if __name__ == '__main__':
    try:
        create_directory_structure()
        extract_images() # Uncomment only if you want to extract images at Granular level.
        create_dataset()
    except FileNotFoundError:
        print("Please check if the CrisisMMD dataset has been extracted in the same directory structure level")
    except Exception as e:
        print("Exception Occured::\n", e)
