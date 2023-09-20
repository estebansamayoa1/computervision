import cv2 as cv
import numpy as np
import argparse
import cvlib

parser = argparse.ArgumentParser()

parser.add_argument("--p", "--path", required=True, help="Path to the image file")

args = parser.parse_args()

img=cvlib.imgread(args.p)

if img is None:
    print("Error: Image not found or invalid image file path.")
    exit()

binary_image = cv.threshold(img, 125, 255, cv.THRESH_BINARY)[1]
binary_image = cv.bitwise_not(binary_image)


edges = cv.Canny(binary_image, 50, 150)

contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 

min_height = 60
max_height = 90 

result_image = img.copy()
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    contour_height = h
    
    if min_height <= contour_height <= max_height:
        cv.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv.imshow('Letras en las placas', result_image)
cv.waitKey(0)
cv.destroyAllWindows()


