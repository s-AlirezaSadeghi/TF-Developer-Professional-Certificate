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
            print('\nAccuracy is high enough so cancelling the training!')
            self.model.stop_training=True





callback = Mycallback()

parent_dir= r'C:\TF_professional_training\cats_vs_dogs\cats_and_dogs_filtered'

train_data = os.path.join(parent_dir,'train')
validation_data = os.path.join(parent_dir,'validation')

train_dog_data_dir = os.path.join(train_data,'dogs')
train_cat_data_dir = os.path.join(train_data,'cats')


train_dog_data = os.listdir(train_dog_data_dir)
train_cat_data = os.listdir(train_cat_data_dir)




# creating instance of train an validation generators from directories, scale and resize them to a 300x300
train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(train_data,
                                                    target_size=(150,150),
                                                    batch_size=128,
                                                    class_mode='binary')


test_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = train_datagen.flow_from_directory(validation_data,
                                                    target_size=(150,150),
                                                    batch_size=128,
                                                    class_mode='binary')




# Create CNN Architecture
model = keras.Sequential([
    keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(32,(3,3),activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation=tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)])

# compile the model and get summary - using RMs prop for setting constant lr
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.summary()



# fit the model with the generators (train and test)
history = model.fit(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=100,
                              epochs=2,
                              validation_steps=50,
                              verbose=1, callbacks=[callback])



### visualization of Conv NET representation ###

# Define a new Model that will take an image as input, and will output
successive_outputs = [layer.output for layer in model.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# Let's prepare a random input image from the training set.
dogs_img_files = [os.path.join(train_dog_data_dir, f) for f in train_dog_data]
cats_img_files = [os.path.join(train_cat_data_dir, f) for f in train_cat_data]
img_path = random.choice(dogs_img_files + cats_img_files)

img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Run image through our network, thus obtaining all
successive_feature_maps = visualization_model.predict(x)

# names of the layers for plot
layer_names = [layer.name for layer in model.layers[1:]]

# Display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # All layers except fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # Feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # Filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')



# visualizing Accuracy and loss
# retrive all accuracy
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']


# retrive all loss vals
loss = history.history['loss']
val_loss = history.history['val_loss']



# get epoch NO#
epochs = range(len(accuracy))

# plot train and val accuracy
plt.plot(epochs,accuracy)
plt.plot(epochs,val_accuracy)
plt.title('Training and validaiton accuracy')

plt.figure()

# plot train and val loss
plt.plot(epochs,loss)
plt.plot(epochs,val_loss)
plt.title('Training and validaiton loss')

plt.show()
