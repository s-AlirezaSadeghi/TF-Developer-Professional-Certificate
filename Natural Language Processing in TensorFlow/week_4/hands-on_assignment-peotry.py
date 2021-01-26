import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

#func definition
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()



poetry_path=r'C:\TF_professional_training\NLP\irish-lyrics-eof.txt'
data = open(poetry_path).read()

corpus = data.lower().split("\n")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)


# generate training data and labels
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words,100,input_length=max_sequence_len-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
    tf.keras.layers.Dense(total_words,activation='softmax')
])

# Create Adam optimizer with Custom learning rate
adam = Adam(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

history = model.fit(xs, ys, epochs=100, verbose=1)


# plot accuracy
plot_graphs(history, 'accuracy')


# generate 100 next word based on a seed sentence
seed_text = "I've got a bad feeling about this"
next_words = 100



for _ in range(next_words):
    # sequence the seed
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    # prepadding to match x shape as in training data
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    #predicting the next work (Arg max of possible predictions
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    # reverse tokenization to add to the seed
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)