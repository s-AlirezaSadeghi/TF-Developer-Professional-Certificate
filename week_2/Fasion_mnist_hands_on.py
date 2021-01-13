import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt



class Mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('loss')<0.4):
            print('\nLoss is low so cancelling the training!')
            self.model.stop_training=True


callback= Mycallback()
# data object
mnist = tf.keras.datasets.fashion_mnist

# train test split from mnsit objt
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


plt.imshow(training_images[0])
# plt.show()
print(training_labels[0])
print(training_images[0].shape)

training_images  = training_images / 255.0
test_images = test_images / 255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(512,activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)])


model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images,training_labels,epochs=6,callbacks=[callback])


#
model.evaluate(test_images, test_labels)
