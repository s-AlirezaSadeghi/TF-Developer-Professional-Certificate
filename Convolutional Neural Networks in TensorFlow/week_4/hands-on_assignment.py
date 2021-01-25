import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



class Mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>0.99):
            print('\n Accuracy is high enough so cancelling the training!')
            self.model.stop_training=True


parent_dir=r'C:\TF_professional_training\rock_paper_scissor'


train_data = os.path.join(parent_dir,'train')
validation_data = os.path.join(parent_dir,'test')



callback = Mycallback()



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
                                                    class_mode='categorical')

# for test dataset we only rescale images
test_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = train_datagen.flow_from_directory(validation_data,
                                                    target_size=(150,150),
                                                    batch_size=20,
                                                    class_mode='categorical')




# Create CNN Architecture
model = keras.Sequential([
    keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Dropout(0.2)
    keras.layers.Conv2D(128,(3,3),activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Dropout(0.2)
    keras.layers.Conv2D(128,(3,3),activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation=tf.nn.relu),
    keras.layers.Dense(3,activation=tf.nn.softmax)])

# compile the model and get summary - using RMs prop for setting constant lr
model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.summary()



# fit the model with the generators (train and test)
history = model.fit(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=100,
                              epochs=1,
                              validation_steps=50,
                              verbose=1, callbacks=[callback])