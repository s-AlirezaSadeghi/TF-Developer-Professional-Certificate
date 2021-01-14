import os
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class Mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>0.99):
            print('\nAccuracy is high enough so cancelling the training!')
            self.model.stop_training=True



# root directory for training and validation data
train_data = r'C:\TF_professional_training\Data\train'
validation_data = r'C:\TF_professional_training\Data\validation'

# training horse pictures
train_horse_dir = r'C:\TF_professional_training\Data\train\horses'

# training human pictures
train_human_dir = r'C:\TF_professional_training\Data\train\humans'




# get list picture in each directory
train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])




# set img size and conf
nrows=4
ncols=4
pic_index = 0


# show pictures
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname)
                for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname)
                for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


# instantiate call back for auto pilot

callback = Mycallback()

# creating instance of train an validation generators from directories, scale and resize them to a 300x300
train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(train_data,
                                                    target_size=(300,300),
                                                    batch_size=128,
                                                    class_mode='binary')


test_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = train_datagen.flow_from_directory(validation_data,
                                                    target_size=(300,300),
                                                    batch_size=128,
                                                    class_mode='binary')


# create CNN Architecture
model = keras.Sequential([
    keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(32,(3,3),activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation=tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)])

# compile the model and get summary - using RMs prop for setting constant lr
model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.summary()


# fit the model with the generators (train and test)
model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=8,
    verbose=2,
    callbacks=[callback])




'''' visualization of Neural network representation '''

# Define a new Model that will take an image as input, and will output
successive_outputs = [layer.output for layer in model.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# Let's prepare a random input image from the training set.
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)

img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
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

