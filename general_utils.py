import numpy as np 
import cv2

def gaussian1D(arr, sigma):
    e = np.exp(np.divide(np.square(arr), np.multiply(-2,np.square(sigma))))
    d = np.multiply(np.sqrt(np.multiply(2,np.pi)), sigma)
    return np.divide(e,d)

def modxy(x, patchSize, image):
    margin = (patchSize - 1)/2
    xind = np.arange(x[1]-margin , x[1]+margin+1)
    yind = np.arange(x[0]-margin , x[0]+margin+1)
    xind = np.clip(xind, 0, image.shape[1] - 1)
    yind = np.clip(yind, 0, image.shape[0] - 1)
    x0 = x[1]*np.ones((patchSize)) 
    y0 = x[0]*np.ones((patchSize))
    x1 = np.square(np.subtract(x0, xind)).reshape((patchSize,1))
    y1 = np.square(np.subtract(y0, yind)).reshape((1,patchSize))
    # s = np.sqrt(x1+y1)
    s = (x1+y1)
    return s, xind, yind

def indMatrix(xind, yind, patchSize):
    indMat = np.vstack((xind, xind, xind, xind, xind))
    retMat = np.zeros((patchSize, patchSize, 2))
    for i in range(patchSize):
        b = yind[i]*np.ones((patchSize))
        a = np.dstack((indMat[i], b))
        retMat[i] = a
    return retMat.astype(int)