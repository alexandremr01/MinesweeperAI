import os
from time import time
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from agents.L4MSAgent import L4MSAgent

# Some PCs needed these configurations in order to run with GPU.
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

# Comment this line to enable training using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Training configuration
EPOCHS = 5
BATCH_SIZE = 128

# Loads training set
train_features = np.load('dataset/dataset_X.npy')
train_features = (train_features + 1) / 10.0
train_labels = np.load('dataset/dataset_Y.npy')

# Splits training set into train and cross-validation sets
train_features, validation_features, train_labels, validation_labels = \
    train_test_split(train_features, train_labels, test_size=0.05, random_state=3)

print('# of training set:', train_features.shape[0])
print('# of cross-validation set:', validation_features.shape[0])

print('X shape: ', train_features.shape)
print('Y shape: ', train_labels.shape)

# Minesweeper agent configuration
side = 8
bombs = 10
agent = L4MSAgent(side)

X_train, y_train = train_features, train_labels
X_validation, y_validation = validation_features, validation_labels

train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size=BATCH_SIZE)
validation_generator = ImageDataGenerator().flow(X_validation, y_validation, batch_size=BATCH_SIZE)

steps_per_epoch = X_train.shape[0] // BATCH_SIZE
validation_steps = X_validation.shape[0] // BATCH_SIZE
model = "best_model.hdf5"
if os.path.exists(model):
    print('Loading weights from previous learning session.')
    agent.load(model)
else:
    print('No weights found from previous learning session.')

checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
tensorboard = TensorBoard(log_dir="logs")
agent.model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                validation_data=validation_generator, validation_steps=validation_steps,
                shuffle=True, callbacks=[tensorboard, checkpoint])

agent.model.save_weights('final_weights.h5')
