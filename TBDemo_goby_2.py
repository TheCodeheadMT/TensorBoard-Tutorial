'''This demonstration is based off the companion notebook for the book 
[Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras&a_bid=76564dff). 
This notebook was generated for TensorFlow 2.6, but is current up to 2.12.0.'''

## PART 1.5 ##
# Update the model and train it again using what we learned from TensorBoard. 


import tensorflow as tf
import numpy as np
import time
from tensorflow import keras
from sklearn import metrics
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import TensorBoard  


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

# Setup logging scheme
MODEL = "my-final-model-{}".format(int(time.time()))

# Initialize callback (create logs directory first or model.fit will fail)
tbcallback = TensorBoard(log_dir = "logs/{}".format(MODEL), histogram_freq=1)

#___________________________________________________________________________
# Make any adjustments for your final model

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Compile the network using your optmizier, loss, and metrics
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

#___________________________________________________________________________
# Refit your final model using all training data

model.fit(x_train,                              # No longer partial_x_train
          y_train,                              # No longer partial_y_train
          epochs=5,                             # Changed?
          batch_size=512, 
          callbacks=[tbcallback])

# Open browser to http://localhost:6006/


# Check how accurate the model is at predicting unseen observations
results = model.evaluate(x_test, y_test)

# tuple containing loss, accuracy
print(results)