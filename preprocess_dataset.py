import glob
import numpy as np

# This file pre-processes the dataset mapping X to Y by
# concatenating all dataset files,
# removing the repeated game states in X and finally
# creating two new files for X and Y which will be used
# in training.


filesX = sorted(glob.glob("dataset/dataset_X*"))
filesY = sorted(glob.glob("dataset/dataset_Y*"))

X = np.concatenate([np.load(datafile) for datafile in filesX])
Y = np.concatenate([np.load(datafile) for datafile in filesY])
print("Original shapes:")
print(X.shape)
print(Y.shape)
_, ix, counts = np.unique(X, return_counts=True, axis=0, return_index=True)
X = X[ix]
Y = Y[ix]
np.save('dataset/dataset_X', X)
np.save('dataset/dataset_Y', Y)
print("Final shapes:")
print(X.shape)
print(Y.shape)
