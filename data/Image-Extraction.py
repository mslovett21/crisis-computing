"""
 Following class is responsible for Image data extraction.
 Prior executing this class you will have to extract the CrisisMMD dataset on the same directory level
 as of this class.
 CrisisMMD dataset can be downloaded from : https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz
"""
import os
from random import shuffle
import pandas as pd
import shutil
from shutil import copy


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
            df_noninfo = df.loc[df['image_info'] == 'not_informative']
            df_noninfo2 = df.loc[(df['image_info'] == 'informative') & (df['text_info'] == 'not_informative')]
            info_sep(category=key, df_informative=df_informative)
            non_info_sep(df_noninfo)
            non_info_sep(df_noninfo2)


def create_dataset():
    """
    Method to split the dataset into training and testing w.r.t
    Informative and Non-Informative category.
    :return: None
    """
    # TODO: Code cleaning and modularity
    info = []
    non_info = []
    for path, subdirs, files in os.walk(INFORMATIVE):
        for name in files:
            info.append(os.path.join(path, name))
    tot = len(info)
    print("Total Informative data:", tot)
    shuffle(info)
    train_split = int(tot * .90)
    train_data = info[:train_split]
    test_data = info[train_split:]
    for fname in train_data:
        copy(fname, TRAINING_INFORMATIVE)
    for fname in test_data:
        copy(fname, TESTING_INFORMATIVE)

    for path, subdirs, files in os.walk(NON_INFORMATIVE):
        for name in files:
            non_info.append(os.path.join(path, name))
    tot = len(non_info)
    print("Total Non-Informative data:", tot)
    shuffle(non_info)
    train_split = int(tot * .90)
    train_data = non_info[:train_split]
    test_data = non_info[train_split:]
    for fname in train_data:
        copy(fname, TRAINING_NON_INFORMATIVE)
    for fname in test_data:
        copy(fname, TESTING_NON_INFORMATIVE)


if __name__ == '__main__':
    try:
        create_directory_structure()
        extract_images()
        create_dataset()

    except FileNotFoundError:
        print("Please check if the CrisisMMD dataset has been extracted in the same directory structure level")
    except Exception as e:
        print("Exception Occured::\n", e)
