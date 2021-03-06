import tensorflow as tf

print(tf.__version__)

# pip install -q tensotflow-datasets

import tensorflow_datasets as tfds

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)


train_data, test_data = imdb['train'], imdb['test']


training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []


for s,l in train_data:
    training_sentences.append(str(s.tonumpy()))
    training_labels.append(l.tonumpy())


for s,l in test_data:
    testing_sentences.append(str(s.tonumpy()))
    testing_labels.append(l.tonumpy())


vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV>'
num_epochs = 10

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)


sequences = tokenizer.texts_to_sequence(training_sentences)
padded = pad_sequences(sequences,max_lenght=max_length)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_lenght=max_length),
    # tf.keras.layers.Flatten(),
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dense(6, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(padded,
          training_labels_final,
          num_epochs,
          validation_data=(testing_padded,testing_labels_final))


#visualizing Embedding
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

reverse_word_index = dict([(v,k) for (k,v) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[3]))
print(training_sentences[3])



import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()



