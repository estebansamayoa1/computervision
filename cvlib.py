#cvlib.py
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math

def imgread(filename):
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    if img.dtype != np.uint8:
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return img


def imgview(img):
    fig = plt.imshow(img, cmap='gray',vmin=0,vmax=255)
    plt.axis('off')
    return fig
    

def hist(img):
    tabla, (foto, histo)=plt.subplots(1, 2, figsize=(12, 6))
    foto.imshow(img, cmap='gray',vmin=0,vmax=255)
    foto.axis('off')
    histo.hist(img.ravel(),256,[0,256])
    plt.show()

def imgcmp(img1, img2):
    tabla, (fig1, fig2)=plt.subplots(1, 2, figsize=(12, 6))
    fig1.imshow(img1, cmap='gray',vmin=0,vmax=255)
    fig2.imshow(img2, cmap='gray',vmin=0,vmax=255)
    fig1.axis('off')
    fig2.axis('off')
    plt.show()

def sobel_mag(sx,sy):
    sq=(sx**2)+(sy**2)
    magnitude=math.sqrt(sq)
    return magnitude

def sobel_angle(sx,sy):
    angle=np.arctan2(sx,sy)
    return angle

