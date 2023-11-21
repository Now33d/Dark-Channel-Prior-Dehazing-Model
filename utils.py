import numpy as np 
import cv2
import os
import datetime

import general_utils
from general_utils import gaussian1D, modxy, indMatrix

class Image:
    # to: lower bound for a transmission map
    def __init__(self, path, patchSize = 3, to = 0.1):
        print('Initializing object...')
        start_time = datetime.datetime.now()
        self.image = cv2.imread(path).astype(float)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.name = os.path.basename(path)
        self.dest = os.getcwd() + '/results_0.1/' + self.name
        self.patchSize = patchSize + (not(patchSize%2))
        self.to = to 
        self.A = self.atmosphericLight()
        self.tm = self.transMat()
        end_time = datetime.datetime.now()
        print('Object initialized in {} seconds'.format(date_diff_in_Seconds(end_time, start_time)) )
        return
    
    # saves the image file in the same directory
    def save(self):
        # display = cv2.resize(self.image, (int(self.width/2),int(self.height/2)))
        cv2.imwrite('image.png',self.image)
        return

    # extract the minimum RGB value for each pixel in the patch of size nxm.
    # Return the minimum color intensity(colorInt) and color index(colorInd) for the patch.
    def minRGB(self, patch):
        colorInt = np.min(patch, axis=2)
        colorInd = np.argmin(patch, axis=-1)
        return colorInt, colorInd
    
    # given an patch it finds a matrix with minimum RGB values for each pixel 
    # then it finds the median intensity
    def medianPatchInt(self, patch):
        imgIntMatrix, _ = self.minRGB(patch)
        return np.median(imgIntMatrix)
    
    # given an patch it finds a matrix with minimum RGB values for each pixel 
    # then it finds the min intensity
    def minPatchInt(self, patch):
        imgIntMatrix, _ = self.minRGB(patch)
        return np.min(imgIntMatrix)
    
    # extracts a local patch of the given size about pixel x:(x,y)
    def localPatch(self, x):
        margin = (self.patchSize - 1)/2
        xmin = int(min(max(x[1] - margin, 0),self.image.shape[1] - 1))
        xmax = int(min(max(x[1] + margin, 0),self.image.shape[1] - 1) + 1)
        ymin = int(min(max(x[0] - margin, 0),self.image.shape[0] - 1))
        ymax = int(min(max(x[0] + margin, 0),self.image.shape[0] - 1) + 1)
        patch = self.image[ymin:ymax , xmin:xmax]
        # print('Bounding box about pixel {} is {} {} {} {}'.format(x, xmin, xmax, ymin, ymax))
        # print('Patch Size: {}'.format(patch.shape))
        return patch
    
    # returns I_dark(x) for a pixel - Eq(12)
    def intensityDark(self, x):    
        patch = self.localPatch(x)
        return self.medianPatchInt(patch)
    
    # returns atmospheric light A for an image - Eq(13) 
    def atmosphericLight(self):
        intensityDarkMat = np.zeros((self.height, self.width))
        for idx, x in np.ndenumerate(self.image):
            if idx[2] == 0:
                intensityDarkMat[idx[0]][idx[1]] = self.intensityDark(x = [idx[1],idx[0]])
        x = np.unravel_index(np.argmax(intensityDarkMat, axis=None), intensityDarkMat.shape)
        return self.image[x]
    
    # compute the transmission map about a pixel x - Eq (9)
    # look at Eq(17) and Eq(18)
    def transmissionMap(self, x):
        patch = self.localPatch(x)
        minEst = self.minPatchInt((1/self.A)*patch) 
        # t = max((1-minEst), self.to)
        t = 1 - minEst
        return t
    
    # compute the transmission matrix for the entire image during init
    def transMat(self):
        mat = np.zeros((self.height, self.width))
        for x in range(self.width):
            for y in range(self.height):
                mat[y][x] = self.transmissionMap([x,y])
        return mat
    
    # refine transmission for the entire image
    def transmissionRefinement(self, x, method = 'gaussian', sigma = 4):        
        if method == 'gaussian':
            # calculate input to gaussian function
            modVal, xind, yind = modxy(x, self.patchSize, self.image)
            # apply gaussian function
            gaus = gaussian1D(modVal, sigma)
            # get index matrix
            indxPairs = indMatrix(xind, yind, self.patchSize)
            # extract transmission values for indxPairs
            transNbhd = np.zeros((indxPairs.shape[0], indxPairs.shape[1]))
            for y in range(self.patchSize):
                for x in range(self.patchSize):
                    x0,y0 = indxPairs[y][x]
                    transNbhd[y][x] = self.tm[y0][x0]
            t = np.sum( np.multiply( gaus,transNbhd ) ) / np.sum(gaus)
            t = max(t , self.to)
            return t
            
            
    # estimate the value of the pixel without dehazing
    def estimateInt(self, x):
        t = self.transmissionRefinement(x)
        I = self.image[x[1]][x[0]]
        return (((I - self.A) / t) + self.A).astype(int)
    
    # apply de-hazing over the entire image
    def reconstructImage(self):
        reconstructedImg = np.zeros((self.height, self.width, 3))
        for x in range(self.width):
            for y in range(self.height):
                reconstructedImg[y][x] = self.estimateInt([x,y])
        maxVal = np.max(reconstructedImg)
        minVal = np.min(reconstructedImg)
        reconstructedImg = (reconstructedImg - minVal) * 255.0/maxVal
        cv2.imwrite(self.dest,reconstructedImg)
        return reconstructedImg



def date_diff_in_Seconds(dt2, dt1):
    timedelta = dt2 - dt1
    return timedelta.days * 24 * 3600 + timedelta.seconds

