import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


sentence = ['I love my dog',
            'I love my cat',
            'You love my dog',
            'Do you think my dog is amazing?']

tonkenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tonkenizer.fit_on_texts(sentence)
word_index = tonkenizer.word_index
print(word_index)

sentences = tonkenizer.texts_to_sequences(sentence)
padded = pad_sequences(sentences,padding='post', maxlen=5, truncating='post')
print(sentences)
print(padded)


test_data = ['I really love my dog',
             'my dog loves my manatee']

test_seq = tonkenizer.texts_to_sequences(test_data)
print(test_seq)