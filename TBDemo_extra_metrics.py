'''This demonstration is based partly off of "Deep Dive Into TensorBoard" from https://neptune.ai/blog/tensorboard-tutorial.
Code is desinged to be a starting point to conduct experiements using a small set of hyperparameters.'''

import tensorflow as tf
import numpy as np
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
# Import hyper parameter tuning library
from tensorboard.plugins.hparams import api as hp

#Get MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

def get_model():
    # setup logging scheme
    MODEL = "MNIST-extra-metrics{}".format(int(time.time()))
     # setup callback to log histograms, write images, profile batch 2 only, and embeddings
    tbcallback = TensorBoard(log_dir = "logs/{}".format(MODEL),
                            histogram_freq=1,
                            write_graph=True,
                            write_images=True,
                            update_freq='epoch',
                            embeddings_freq=1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    model.fit(X_train, y_train,
            epochs=10,
            validation_split=0.2,
            callbacks=[tbcallback])
    
    return model

# build basic model and return for later use
model = get_model()