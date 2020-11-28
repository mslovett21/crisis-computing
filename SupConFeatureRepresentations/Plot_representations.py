import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE


def create_cols():
    col = 'col'
    cols = []
    for i in range(128):
        cols.append(col + str(i))
    return cols


cols = create_cols()
np.random.seed(42)

df = pd.read_csv('fake.csv')
df['features'] = df['features'].apply(lambda x: np.fromstring(x.replace("|", " "), sep=' '), 0)
cleaned_df = df[['features', 'actual label']]
x_val = df["features"].values
feature_df = pd.DataFrame(columns=cols)
targets = []
for ind, row in cleaned_df.iterrows():
    val = row['features']
    if len(val) == 128:
        a_series = pd.Series(val, index=feature_df.columns)
        feature_df = feature_df.append(a_series, ignore_index=True)
        targets.append(row['actual label'])
    else:
        # there are some images in the dataset which were grayscale or were not getting processed.
        # We exclude these images from plot.
        print("Error", val)
feature_df['target'] = targets

tsne = TSNE(n_components=2, random_state=42)
tsne_obj = tsne.fit_transform(feature_df)
tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                        'Y': tsne_obj[:, 1],
                        'digit': targets})
tsne_df.head()
sns.scatterplot(x="X", y="Y",
                hue="digit",
                palette=['orange', 'blue'],
                legend='full',
                data=tsne_df)
plt.show()
