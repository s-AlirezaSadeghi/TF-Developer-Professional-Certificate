import tensorflow as tf
import numpy as np
from tensorflow import keras

# GRADED FUNCTION: house_model
def house_model(y_new):
    xs = np.array([1,2,3,4,5,6,7,8,9],dtype=float)
    ys =  np.array([1,1.5,2,2.5,3,3.5,4,4.5,5],dtype=float)
    model = tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
    model.compile(optimizer='sgd',loss='mean_squared_error')
    model.fit(xs,ys,epochs=600)
    return model.predict(y_new)[0]


prediction = house_model([7.0])*100
print(prediction)