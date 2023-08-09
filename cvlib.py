#cvlib.py
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def imgread(filename):
    img=cv.imread(filename, cv.IMREAD_GRAYSCALE)
    return img

def imgview(img):
    fig = plt.imshow(img,vmin=0,vmax=255)
    plt.axis('off')
    return fig
    

def hist(img):
    tabla, (foto, histo)=plt.subplots(1, 2, figsize=(12, 6))
    foto.imshow(img,vmin=0,vmax=255)
    foto.axis('off')
    histo.hist(img.ravel(),256,[0,256])
    plt.show()

def imgcmp(img1, img2):
    tabla, (fig1, fig2)=plt.subplots(1, 2, figsize=(12, 6))
    fig1.imshow(img1,vmin=0,vmax=255)
    fig2.imshow(img2,vmin=0,vmax=255)
    fig1.axis('off')
    fig2.axis('off')
    plt.show()

