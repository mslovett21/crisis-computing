import sys
import os
import pandas as pd
import resnet50 as res
import torch

ROOT = r'/home/visonaries566/supcontrast/SupContrast/crisis/crisis-computing/'
PATH_TRAIN_CSV = ROOT+ 'text-analysis-scripts/additional_files/predictions_full_train_df.csv'
PATH_TEST_CSV = ROOT+'text-analysis-scripts/additional_files/predictions_test_df.csv'
PATH_MERGED = ROOT+'data/CrisisMMD_v2.0/test_merged.xlsx'
PATH_IMAGE = r'/home/visonaries566/supcontrast/SupContrast/crisis/crisis-computing/data/CrisisMMD_v2.0'


def get_files():
    """
    return train and test tweet ids along with merged.csv
    """
    train_tweet = pd.read_csv(PATH_TRAIN_CSV)
    train_tweet_id = train_tweet[['tweet_id']].copy()

    test_tweet = pd.read_csv(PATH_TEST_CSV)
    test_tweet_id = test_tweet[['tweet_id']].copy()

    merged = pd.DataFrame(pd.read_excel(PATH_MERGED))
    merged_data_df = merged[['tweet_id','image_id','image_path','image_info','text_info']].copy()

    return merged_data_df, test_tweet_id, train_tweet_id


def get_image_data(merged_df, dataset):
    """
    returns tweet id, image path and image label
    params: Maindataset: Merged.csv file 
            dataset: Queried dataset
    """
    image_df = pd.DataFrame(columns=["tweet_id", "image_id", "image_path", "actual_label", "predicted_probability", "predicted_label"])

    for ind, val in enumerate(dataset['tweet_id']):
        query_df = merged_df.loc[merged_df['tweet_id'] == val]

        for ind, rows in query_df.iterrows():
            if rows['text_info'] == rows['image_info']:
                img_id = rows['image_id']
                image_path = os.path.join(PATH_IMAGE,rows['image_path'])
                label = 0 if rows['image_info'] =='informative' else 1
                row = pd.Series([val, img_id, image_path, label,-1,-1], index=image_df.columns)
                image_df = image_df.append(row, ignore_index=True)

    return image_df

def get_image_and_labels(data_df):
    """
    returns image paths and their labels
    """
    images = []
    labels = []

    for ind, row in data_df.iterrows():
        images.append(row['image_path'])
        labels.append(row['actual_label'])
    
    return images, labels

if __name__ == "__main__":

    merged_data_df, test_tweet_id, train_tweet_id = get_files()
    
    train_df = get_image_data(merged_data_df, train_tweet_id)
    test_df = get_image_data(merged_data_df, test_tweet_id)

    print("Data successfully loaded...")
    images_test, labels_test = get_image_and_labels(test_df)
    images_train, labels_train = get_image_and_labels(train_df)

    output_prob, classes = res.run_inference(images_train, labels_train, "train")

    for ind, row in train_df.iterrows():
        train_df.iloc[ind]['predicted_label'] = 1 if classes[ind] == 0 else 0
        train_df.iloc[ind]['predicted_probability'] = 1 - output_prob[ind][0]

    train_df.to_csv('predictions_train.csv',index=False)

    print("Train CSV file created successfully.")

    output_prob,classes = res.run_inference(images_test,labels_test,"test")

    for ind, row in test_df.iterrows():
        test_df.iloc[ind]['predicted_label'] = 1 if classes[ind] == 0 else 0
        test_df.iloc[ind]['predicted_probability'] = 1-output_prob[ind][0]

    test_df.to_csv("predictions_test.csv",index=False)

    print("Test CSV file created successfully.")

