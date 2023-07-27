'''This demonstration extends "Deep Dive Into TensorBoard" from https://neptune.ai/blog/tensorboard-tutorial.
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

# Set to 1 so training during demo fits in schedule
# log data reviwed during class used 10 epochs during param search.
EPOCHS = 1

### MNIST MODEL USED TO OPTIMIZE ### 
'''This function is called with hparams[UNITS, DROPOUT, OPTIMIZER] 
and returns the accuracy score for the model instance tested.'''
def get_mnist_model(hparams, exp_num):
      
    # setup callback to log histograms, write images, profile batch 2 only, and embeddings
    tbcallback = TensorBoard(log_dir = "logs/Exp_"+str(exp_num)+"_"+hparams[OPTIMIZER]+"_"
                        +str(hparams[UNITS])+"_"+str(hparams[DROPOUT]),
                            histogram_freq=1,
                            write_graph=True,
                            write_images=True,
                            update_freq='epoch',
                            embeddings_freq=1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(hparams[UNITS],  activation='relu'),
        tf.keras.layers.Dropout(hparams[DROPOUT]),
        tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer=hparams[OPTIMIZER],
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(X_train, y_train,
            epochs=EPOCHS,
            validation_split=0.2,
            callbacks=[tbcallback])

    loss, accuracy = model.evaluate(X_test, y_test)
    
    return accuracy
####

# Define hyper params to be included while tuning
UNITS = hp.HParam('units', hp.Discrete([300, 200, 512]))
DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2, 0.5))
OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))
METRIC_ACCURACY = 'accuracy'

# Create the logging location for each Experiment and specificy the fields recorded
with tf.summary.create_file_writer('hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[UNITS, DROPOUT, OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],)

# This function takes in the expriment directory and a set of hyper parameters 
# and executes the training and reporting cycle for the model get_mnist_model() using
# hyper parameters hparams.'''
def experiment(experiment_dir, hparams, experiment_no):

    with tf.summary.create_file_writer(experiment_dir).as_default():
        hp.hparams(hparams)
        accuracy = get_mnist_model(hparams, experiment_no)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


# Set experiment coutner
experiment_no = 0

# Iterate through all settings for units, dropout, and optmizer.
# Note there will be len(UNITS)*len(dropout)*len(optimizers) models trained,
# e.g., 3*2*3 = 18 models for this demo.
for units in UNITS.domain.values:
    for dropout_rate in (DROPOUT.domain.min_value, DROPOUT.domain.max_value):
        for optimizer in OPTIMIZER.domain.values:
            hparams = {
                UNITS: units,
                DROPOUT: dropout_rate,
                OPTIMIZER: optimizer,}

            experiment_name = f'Experiment {experiment_no}'
            print(f'Starting Experiment: {experiment_name}')
            print({h.name: hparams[h] for h in hparams})
            experiment('hparam_tuning/' + experiment_name+"_"+hparams[OPTIMIZER]+"_"
                        +str(hparams[UNITS])+"_"+str(hparams[DROPOUT]), hparams, experiment_no)

            experiment_no += 1





