import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class Mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>0.99):
            print('\n Accuracy is high enough so cancelling the training!')
            self.model.stop_training=True


callback = Mycallback()

parent_dir = r'C:\TF_professional_training\cats_vs_dogs\cats_and_dogs_filtered'

train_data = os.path.join(parent_dir,'train')
validation_data = os.path.join(parent_dir,'validation')

train_dog_data_dir = os.path.join(train_data,'dogs')
train_cat_data_dir = os.path.join(train_data,'cats')


train_dog_data = os.listdir(train_dog_data_dir)
train_cat_data = os.listdir(train_cat_data_dir)




# creating instance of train an validation generators from directories,
# scale and resize them to a 150x150 plus Agumentation for Training
train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(train_data,
                                                    target_size=(150,150),
                                                    batch_size=128,
                                                    class_mode='binary')

# for test dataset we only rescale images
test_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = train_datagen.flow_from_directory(validation_data,
                                                    target_size=(150,150),
                                                    batch_size=20,
                                                    class_mode='binary')




# Create CNN Architecture
model = keras.Sequential([
    keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(128,(3,3),activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(128,(3,3),activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation=tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)])

# compile the model and get summary - using RMs prop for setting constant lr
model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.summary()



# fit the model with the generators (train and test)
history = model.fit(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=100,
                              epochs=5,
                              validation_steps=50,
                              verbose=1, callbacks=[callback])



# plotting Accuracy and Loss for  Validation and Training Dataset
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

