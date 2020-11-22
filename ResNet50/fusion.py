import sys
import os
import pandas as pd
from ResNet50 import resnet50 as res

ROOT = r'C:\Users\Shubham Nagarkar\Desktop\crisis-computing\data\CrisisMMD_v2.0'
PATH_TRAIN_CSV = r'C:\Users\Shubham Nagarkar\Desktop\crisis-computing\text-analysis-scripts\acc_84_BiLSTM\predictions_full_train_df.csv'
PATH_TEST_CSV = r'C:\Users\Shubham Nagarkar\Desktop\crisis-computing\text-analysis-scripts\acc_84_BiLSTM\predictions_test_df.csv'
PATH_MERGED = r'C:\Users\Shubham Nagarkar\Desktop\crisis-computing\data\CrisisMMD_v2.0\Merged.xls'

ignore_tweets = [905625009430949888, 910523364154003456, 869918891664977921, 929989361453621249]

train_tweet = pd.read_csv(PATH_TRAIN_CSV)
train_tweet_id = train_tweet[['tweet_id']].copy()

test_tweet = pd.read_csv(PATH_TEST_CSV)
test_tweet_id = test_tweet[['tweet_id']].copy()

merged = pd.read_csv(PATH_MERGED)
merged_data_df = merged[['tweet_id','image_path','image_info']].copy()

def get_image_data(merged_df, dataset):
    """
    returns tweet id, image path and image label
    params: Maindataset: Merged.csv file 
            dataset: Queried dataset
    """
    image_df = pd.DataFrame(columns=["tweet_id", "image_path", "actual label", "predicted probability", "predicted label"])

    for ind, val in enumerate(dataset['tweet_id']):
        query_df = merged_df.loc[merged_df['tweet_id']==val]
        if query_df.empty:
            continue
        image_path = os.path.join(ROOT,query_df['image_path'].values[0])
        label = 1 if query_df['image_info'].values[0] =='informative' else 0
        row = pd.Series([val, image_path, label,-1,-1], index=image_df.columns)
        image_df = image_df.append(row, ignore_index=True)
    
    return image_df

# train_df = get_image_data(merged_data_df, train_tweet_id)
test_df = get_image_data(merged_data_df, test_tweet_id)
image = ""
for row in test_df.iterrows():
    print(row)
    image = row['image_path'].values[0]
    print(image)
    break

# image =
# model = res.Resnet().to(res.DEVICE)
# model.load_state_dict(torch.load(res.FINAL_CKPT, map_location ='cuda:0'))
# output = model(image.to(DEVICE))
# print(output)