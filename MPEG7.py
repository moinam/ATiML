# pip install numpy
# pip install opencv-python

import numpy as np
import cv2

image=cv2.imread('000005.jpg')

def divideBlocksAndGetDominantColor(image):
    height, width, channels = image.shape
    
    blocks = []
    h = 8
    v = 8
    for i in range(v):
        for j in range(h):
            temp = image[int((height/v) * i) : int((height/v) * (i+1)) , int((width/h) * j) : int((width/h) * (j+1))]
            blocks.append(np.array(averageColor(temp)))
    
    return getlargerImage(blocks, h, v, 1)

def averageColor(temp):
    avg_row = np.average(temp, axis=0)
    avg_color = np.average(avg_row, axis=0)
    return_img = np.ones((1,1,3), dtype=np.uint8)
    return_img[:,:] = avg_color
    return return_img

def getlargerImage(blkList, horizontal, vertical, times):
    overAllList = []
    for i in range(horizontal):
        eachRow = []
        for j in range(vertical):
            for k in range(times):
                eachRow.append(blkList[i*horizontal + j][0][0])
            temp = np.array(eachRow)
        for k in range(times):
            overAllList.append(temp)
    
    return np.array(overAllList)

def mpeg7_features(image):
    
    #Image partitioning and representative color selection
    representative_img = divideBlocksAndGetDominantColor(image)
    
    #Conversion of color space from RGB to YCbCr
    YCbCr = cv2.cvtColor(representative_img, cv2.COLOR_BGR2YCrCb)
    
    #Get the Y, Cb and Cr components
    Y, Cb, Cr = YCbCr[:,:,0]/255.0, YCbCr[:,:,1]/255.0, YCbCr[:,:,2]/255.0
    
    #DCT transformation of each component
    dctY = cv2.dct(Y)
    dctCb = cv2.dct(Cb)
    dctCr = cv2.dct(Cr)

    #Zigzag scanning of the transformed matrices\
    scannedDctY = np.concatenate([np.diagonal(dctY[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-dctY.shape[0], dctY.shape[0])])
    scannedDctCb = np.concatenate([np.diagonal(dctCb[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-dctCb.shape[0], dctCb.shape[0])])
    scannedDctCr = np.concatenate([np.diagonal(dctCr[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-dctCr.shape[0], dctCr.shape[0])])
    
    return (scannedDctY, scannedDctCb, scannedDctCr)

print(mpeg7_features(image))