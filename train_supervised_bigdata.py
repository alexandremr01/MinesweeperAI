import os
from time import time
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from agents.L4MSAgent import L4MSAgent
import glob
import gc
# Comment this line to enable training using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

EPOCHS = 10
BATCH_SIZE = 128

# Datasets
X_files = sorted(glob.glob('dataset/dataset_X*'))[0:110]
Y_files = sorted(glob.glob('dataset/dataset_Y*'))[0:110]
train_features_files, validation_features_files, train_labels_files, validation_labels_files = \
    train_test_split(X_files, Y_files, test_size=0.2, random_state=0)


tic = time()
train_features = np.concatenate([np.load(f) for f in train_features_files])
train_labels = np.concatenate([np.load(f) for f in train_labels_files])
_, ix, counts = np.unique(train_features, return_counts=True, axis=0, return_index=True)
train_features = train_features[ix]
train_labels = train_labels[ix]

validation_features = np.concatenate([np.load(f) for f in validation_features_files])
validation_labels = np.concatenate([np.load(f) for f in validation_labels_files])
_, ix, counts = np.unique(validation_features, return_counts=True, axis=0, return_index=True)
validation_features = validation_features[ix]
validation_labels = validation_labels[ix]

train_features = (train_features+1)/10.0
validation_features = (validation_features+1)/10.0

print(train_features[2])

print('X shape: ', train_features.shape)
print('Y shape: ', train_labels.shape)

print('X val shape: ', validation_features.shape)
print('Y val shape: ', validation_labels.shape)

toc = time()

print('Loaded data in '+str(toc-tic))

print('X shape: ', train_features.shape)
print('Y shape: ', train_labels.shape)
side = 8
bombs = 10
agent = L4MSAgent(side, bombs)

X_train, y_train = train_features, train_labels
X_validation, y_validation = validation_features, validation_labels

train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size=BATCH_SIZE)
validation_generator = ImageDataGenerator().flow(X_validation, y_validation, batch_size=BATCH_SIZE)

steps_per_epoch = X_train.shape[0] // BATCH_SIZE
validation_steps = X_validation.shape[0] // BATCH_SIZE
#agent.model.load_weights("best_model.hdf5")
checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
tensorboard = TensorBoard(log_dir="logs")
agent.model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
          validation_data=validation_generator, validation_steps=validation_steps,
          shuffle=True, callbacks=[tensorboard, checkpoint])

agent.model.save_weights('.h5')
