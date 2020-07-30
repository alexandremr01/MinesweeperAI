import glob
import numpy as np
filesX = glob.glob("original_dataset/dataset_X*")
filesY = glob.glob("original_dataset/dataset_Y*")

n = 0
batch_size = 1
j = 0

for (fileX, fileY) in zip(filesX, filesY):
    X = np.load(fileX)
    X = np.expand_dims(X, axis=0).reshape(-1, 8, 8, 1)
    Y = np.load(fileY).T
    for i in range(0, 50000//batch_size):
        Xb = X[i*batch_size: (i+1)*batch_size]
        np.save('dataset/dataset_X_'+str(n), Xb)
        Yb = Y[i*batch_size : (i+1)*batch_size]
        np.save('dataset/dataset_Y_'+str(n), Yb)
        n+=1
    j +=1 
    print(str(j)+' files of '+str(len(filesX)))