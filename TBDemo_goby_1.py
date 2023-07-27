'''This demonstration is based off the companion notebook for the book 
[Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras&a_bid=76564dff). 
This notebook was generated for TensorFlow 2.6, but is current up to 2.12.0.'''

## PART 1 ##
# Objective: Add TensorBoard to a model and train it using training and validation datasets

import tensorflow as tf
import numpy as np
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
#___________________________________________________________________________
# Step 1: Import TesorBoard

from tensorflow.keras.callbacks import TensorBoard  
#___________________________________________________________________________


# Load training and test data from imdb dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

# Helper function to vectorize sequences of words encoded as integers
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

# create train and test datasets    
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Cast as floats for training
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

# Create train/validation split
x_val = x_train[:10000]
y_val = y_train[:10000]
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]

#___________________________________________________________________________
# Step 2: Setup logging scheme

MODEL = "my-initial-model-{}".format(int(time.time()))

# initialize callback (create logs directory first or model.fit will fail)
tbcallback = TensorBoard(log_dir = "logs/{}".format(MODEL), histogram_freq=1)
#___________________________________________________________________________

# initialize a sequential fully connected network to address the ML problem
model = keras.Sequential([
    layers.Dense(16, activation="relu", name="L1_16_Dense_RELU"),
    layers.Dense(16, activation="relu", name="L2_16_Dense_RELU"),
    layers.Dense(1, activation="sigmoid", name="L3_1_Dense_SIG")
])

# Compile the network using your optmizier, loss, and metrics
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

#___________________________________________________________________________
# Step 3: Fit the model, including TensorBoard in the callback list

model.fit(partial_x_train,
          partial_y_train, 
          validation_data=(x_val, y_val),
          epochs=20, 
          batch_size=512, 
          callbacks=[tbcallback])

#___________________________________________________________________________
# Step 4: In console, from directory containing the "logs" directory.
#
# 4.1 execute> tensorboard --logdir logs
#
# 4.2 Open browser to http://localhost:6006/
