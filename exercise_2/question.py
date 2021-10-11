import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

dataset = 'spam_or_not_spam/spam_or_not_spam.csv'

df = pd.read_csv(dataset)

# print(df['label'].value_counts())

# split data
x = df['email']
y = df['label']

# print(x)
# print(y)

# remove below line to see what happens
# AttributeError: 'float' object has no attribute 'lower'
# it seams that there is a number and we can lower() a number
df['email'] = df['email'].astype(str)

email_data = df['email'].values
labels = df['label'].values

# print(email_data)
# print(len(email_data))

# prepare tokenizer
t = Tokenizer()
# update the internal vocabulary based on the email data
# for example if we have -> "The kid playes" It will create a dictionary s.t. word_index["The"] = 1; word_index["kid"] = 2; and so on
# also lower integer means more frequent word
# https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/text.py
t.fit_on_texts(email_data)
# because 0 is reserved for padding go +1
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_email_data = t.texts_to_sequences(email_data)

# print(vocab_size)
# pad email data to a max length
max_length = 80
# pad sequences in order to get the same length for every line
# if one small than the max_length fill it with 0 (zeros) due to the padding='post' parameter
padded_email_data = pad_sequences(encoded_email_data, maxlen=max_length, padding='post')

# print(padded_email_data)
# print(len(padded_email_data))

embeddings_index = dict()

# loading pre-trained dictionary of word embeddings that translates each word into a 100 dimensional vector
f = open('glove.6B.100d.txt', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training email_data using the pretrained embeddings_index above
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)
