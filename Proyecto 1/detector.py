import cv2 as cv
import numpy as np
import cvlib
import os
import glob
import argparse
import joblib


def crop_image(img):
    """
    Esta función se usa para cortar la imagen y que solo se enfoque en la placa, 
    asi no tiene que hacer contornos de cosas innecesarias.
    Hace contornos de toda la imagen de rectangulos y encuentra el rectangulo más claro
    o brillante, que sería la placa por ser blanca y retorna la imagen cortada en la placa para 
    encontrar los contornos más facilmente
    """
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_brightness = 0
    canvas = img.copy()
    for cnt in contours:
        rect = cv.boundingRect(cnt)
        x, y, w, h = rect
        if w*h > 40000:
            mask = np.zeros(img.shape, np.uint8)
            mask[y:y+h, x:x+w] = img[y:y+h, x:x+w]
            brightness = np.sum(mask)
            if brightness > max_brightness:
                brightest_rectangle = rect
                max_brightness = brightness

    if brightest_rectangle is not None:
        x, y, w, h = brightest_rectangle
    cropped_image = img[y:y + h, x:x + w]
    return cropped_image


def shadow_remove(img):
    """
    Esta es una función que permite reducir las sombras en la imagen, para que sea más claro y 
    fácil encontrar los contornos de cada letra. 
    """
    rgb_planes = cv.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv.dilate(plane, np.ones((11,11), np.uint8))
        bg_img = cv.medianBlur(dilated_img, 21)
        diff_img = 255 - cv.absdiff(plane, bg_img)
        norm_img = cv.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremov = cv.merge(result_norm_planes)
    return shadowremov

def thresholding(img):
    """
    Hace una binarización de los valores de la imagen que se le envía y luego, invierte los colores, es decir 
    los valores de 255 los vuelve 0 y viceversa.
    """
    img=shadow_remove(img)
    blur = cv.GaussianBlur(img,(5,5),0)
    ret3,binary_image = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) 
    return binary_image

def find_contours(binary_img, cropped_img):
    """
    Aquí encuentra los contornos de los objetos y los discrimina bajo un porcentaje de la imagen que en 
    este caso serían las letras de las placas, luego los guarda en un arreglo de aprovados para 
    enviarlos a una función que genera una subimagen de cada una de las letras de la placa.
    """
    edges = cv.Canny(binary_img, 50, 150)
    contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
    approved=()
    subimages = []

    height, width = cropped_img.shape
    min_height_percentage = 50
    max_height_percentage = 75
    min_width_percentage = 3.5
    max_width_percentage = 60

    min_height = (min_height_percentage / 100) * height
    max_height = (max_height_percentage / 100) * height
    min_width = (min_width_percentage / 100) * width
    max_width = (max_width_percentage / 100) * width

    result_image = cropped_img.copy()
    sorted_contours = sorted(contours, key=lambda contour: cv.boundingRect(contour)[0])

   
    for contour in sorted_contours:
        x, y, w, h = cv.boundingRect(contour)

        if min_height <= h <= max_height and min_width <= w <= max_width:
            approved += (contour,)
            cv.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    for contour in approved:
        x, y, w, h = cv.boundingRect(contour)

        subimage = cropped_img[y:y + h, x:x + w]
        subimage = cv.resize(subimage, (28, 28))
        subimage = subimage.flatten()
        subimages.append(subimage)
    return subimages

def predict_y(subimages, random_f):
    """
    Se envían las subimagenes para que pueda predecir el valor de cada una usando un Random Forest,
    ya entrenado por un dataset de letras de placas. Lo guarda en un arreglo y luego como texto para
    poder enviar a la imagen.
    """
    valores=[]
    values=""
    for image in subimages:
        image=image.reshape(1,-1)
        y_pred=random_f.predict(image)
        valores.append(y_pred)
    for i in valores:
        for j in i:
            values+=j
    return values

def process_final_img(path, values):
    """
    Esta función procesa la imagen original y le agrega el texto generado por los valores predichos
    y encierra en un rectangulo el area de la placa para mostrarla al final.
    """
    img=cvlib.imgread(path)
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    edges = cv.Canny(blurred, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_brightness = 0
    canvas = img.copy()
    for cnt in contours:
        rect = cv.boundingRect(cnt)
        x, y, w, h = rect
        if w*h > 40000:
            mask = np.zeros(img.shape, np.uint8)
            mask[y:y+h, x:x+w] = img[y:y+h, x:x+w]
            brightness = np.sum(mask)
            if brightness > max_brightness:
                brightest_rectangle = rect
                max_brightness = brightness
    if brightest_rectangle is not None:
        x, y, w, h = brightest_rectangle
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    text = values
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 240, 0)
    x, y = 40, 40
    cv.putText(img, text, (x, y), font, font_scale, font_color, thickness=2)
    return img



def main():
    """
    El main manda a llamar todas las funciones y recibe como parametro el path de la imagen,
    para luego poder procesarla enviarla a todas las funciones y generar la imagen final.
    Aquí se carga el joblib file que contiene el Random Forest ya entrenado para poder predecir los
    valores en las placas.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", "--path", required=True, help="Path to the image file")
    args = parser.parse_args()
    path=args.p
    img=cvlib.imgread(args.p)
    img=cvlib.grayscale(img)
    if img is None:
        print("Error: Image not found or invalid image file path.")
        exit()
    random_f = joblib.load("random_f.joblib")
    cropped_img=crop_image(img)
    binary_img=thresholding(cropped_img)
    subimages=find_contours(binary_img, cropped_img)
    values=predict_y(subimages, random_f)
    result_image=process_final_img(path, values)

    print(f'VALORES EN LA PLACA: {values}')
    cv.imshow('Letras en las placas', result_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


main()


