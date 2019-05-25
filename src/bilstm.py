from config import Config
SEED = Config.SEED

import numpy as np
np.random.seed(SEED)
from tensorflow import set_random_seed
set_random_seed(SEED)
import pandas as pd
import utils

from keras import backend as K
from keras.layers import Bidirectional, Embedding, Dense, Dropout, LSTM
from keras.models import Sequential 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score

train = pd.read_csv(Config.TSV_TRAIN, sep="\t", header=None, names=["idx", "class", "dummy", "text"])
dev = pd.read_csv(Config.TSV_DEV, sep="\t", header=None, names=["idx", "class", "dummy", "text"])
X_train, X_dev, y_train, y_dev = train["text"], dev["text"], train["class"], dev["class"]
X = list(X_train) + list(X_dev) + Config.MALE_NOUNS + Config.FEMALE_NOUNS + Config.PROFESSIONS
MAX_SEQUENCE_LENGTH = Config.MAX_SEQUENCE_LENGTH

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_dev = tokenizer.texts_to_sequences(X_dev)

word_index = tokenizer.word_index
print("Found " + str(len(word_index)) + " unique tokens.")

train_data = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
dev_data = pad_sequences(sequences_dev, maxlen=MAX_SEQUENCE_LENGTH)

train_labels =  to_categorical(np.array(y_train).astype(int), 2)
dev_labels = to_categorical(np.array(y_dev).astype(int), 2)


embedding_index = utils.glove2dict(Config.GLOVE_FILE)
EMBEDDING_DIM = len(next(iter(embedding_index.values())))
count = 0 
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        count = count + 1
print(str(count) + " words missing from GloVe vocabulary")
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.25))
model.add(Dense(2, activation="softmax"))
print(model.summary())

model.compile(
        loss="binary_crossentropy", 
        metrics=["accuracy"], 
        optimizer="Adam",
    )

model.fit(train_data, train_labels, validation_data=(dev_data, dev_labels), 
          epochs=3, shuffle=False,
      )

model.save(Config.LSTM_MODEL_FILE)

dev_preds = [int(item[1] >= 0.5) for item in model.predict_proba(dev_data)]
print(accuracy_score(y_dev, dev_preds))

sentences = utils.get_sentences()
sequences = tokenizer.texts_to_sequences(sentences)
test_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
preds = [item[1] for item in model.predict_proba(test_data)]
(t, prob, diff) = utils.ttest(preds)

file = Config.LSTM_FILE
np.savetxt(file, preds, header="dev accuracy: " + str(accuracy_score(y_dev, dev_preds)))