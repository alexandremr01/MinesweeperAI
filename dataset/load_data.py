import numpy as np
X = np.load('dataset_X.npy')
Y = np.load('dataset_Y.npy')
Y=Y.T
data = np.array(list(zip(X, Y)))
