import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
%matplotlib inline


DATA_PATH = "../data/final_tweets/"

train_df = pd.read_csv(DATA_PATH +'train_df.csv')
valid_df = pd.read_csv(DATA_PATH +'validate_df.csv')
test_df = pd.read_csv(DATA_PATH + 'test_df.csv')

X_train = train_df['tweet_text']
Y_train = train_df['text_info']

X_valid = valid_df['tweet_text']
Y_valid = valid_df['text_info']

X_test = test_df['tweet_text']
Y_test = test_df['text_info']

max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)

# train data
tok.fit_on_texts(X_train)
train_sequences = tok.texts_to_sequences(X_train)
train_sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

# validate data
valid_sequences = tok.texts_to_sequences(X_valid)
valid_sequences_matrix = sequence.pad_sequences(valid_sequences,maxlen=max_len)

# test_data
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)



le = LabelEncoder()
Y = le.fit_transform(Y)

max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)

# train data
tok.fit_on_texts(X_train)
train_sequences = tok.texts_to_sequences(X_train)
train_sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

# validate data
valid_sequences = tok.texts_to_sequences(X_valid)
valid_sequences_matrix = sequence.pad_sequences(valid_sequences,maxlen=max_len)

# test_data
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

def plot_summaries(data1, data2, title, ylabel,fname):
    plt.plot(data1)
    plt.plot(data2)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(fname+".png")

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(2*layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model




if __name__ == '__main__':
    try:
        model = RNN()
        model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
        training_history = model.fit(train_sequences_matrix,Y_train,batch_size=32,epochs= 8,
                  validation_data = (valid_sequences_matrix, Y_valid))

        accr = model.evaluate(test_sequences_matrix,Y_test)
        
        plot_summaries(training_history.history['accuracy'],training_history.history['val_accuracy'],\
              "Model Accuracy", "accuracy", "imgs/acc_notES")
        plot_summaries(training_history.history['loss'],training_history.history['val_loss'],\
              "Model Loss", "loss","imgs/loss_notES")

        model_ES = RNN()
        model_ES.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
        training_history_ES = model_ES.fit(train_sequences_matrix,Y_train,batch_size=64,epochs=10,
          validation_data = (valid_sequences_matrix, Y_valid),callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.001)])
        accr = model_ES.evaluate(test_sequences_matrix,Y_test)

        plot_summaries(training_history_ES.history['accuracy'],training_history_ES.history['val_accuracy'],\
              "Model Accuracy ES", "accuracy", "imgs/acc_ES")

        plot_summaries(training_history_ES.history['loss'],training_history_ES.history['val_loss'],\
              "Model Loss ES", "loss","imgs/loss_ES")


    except Exception as e:
        print("Exception Occured::\n", e)