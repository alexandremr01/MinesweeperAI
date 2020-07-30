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

# Comment this line to enable training using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
X = np.load('dataset/dataset_X_1.npy')
c = X[14].reshape(1, 8, 8, 1)
side = 8
bombs = 10
agent = L4MSAgent(side, bombs)

agent.model.load_weights("best_model.hdf5")
print(c)
print(np.argmax(agent.model.predict(c))//side)
print(np.argmax(agent.model.predict(c))%side)