import tensorflow as tf
from os import path, getcwd, chdir

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)