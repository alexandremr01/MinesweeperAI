import os
from time import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from agents.L4MSAgent import L4MSAgent
import glob
from sklearn.model_selection import train_test_split
from keras_data_generator import DataGenerator
# Comment this line to enable training using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys

EPOCHS = 10
BATCH_SIZE = 128

# Parameters
params = {'dim': (8, 8),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

# Datasets
X_files = glob.glob('dataset/dataset_X*')
X_files = [os.path.basename(path)[9:] for path in X_files]
Y_files = glob.glob('dataset/dataset_Y*')
Y_files = [os.path.basename(path)[9:] for path in Y_files]
data_train, data_test, labels_train, labels_test = train_test_split(X_files, Y_files, test_size=0.20, random_state=42)

# Generators
training_generator = DataGenerator(data_train, **params)
validation_generator = DataGenerator(data_test, **params)

side = 8
bombs = 10
agent = L4MSAgent(side, bombs)

#agent.model.load_weights("best_model.hdf5")
checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
tensorboard = TensorBoard(log_dir="logs")
agent.model.fit(training_generator,epochs=10,
          validation_data=validation_generator,
          shuffle=True, callbacks=[tensorboard, checkpoint])

agent.model.save_weights('.h5')
