import glob
import numpy as np
filesX = sorted(glob.glob("original_dataset/dataset_X*"))
filesY = sorted(glob.glob("original_dataset/dataset_Y*"))

n = 0
batch_size = 1
j = 1

for (fileX, fileY) in zip(filesX, filesY):
    X = np.load(fileX)
    Y = np.load(fileY).T
    _, ix, counts = np.unique(X, return_counts=True, axis=0, return_index=True)
    X = X[ix]
    Y = Y[ix]
    X = np.expand_dims(X, axis=0).reshape(-1, 8, 8, 1)
    np.save('dataset/dataset_X_'+str(j), X)
    np.save('dataset/dataset_Y_'+str(j), Y)        
    j +=1 
    print(str(j)+' files of '+str(len(filesX)))