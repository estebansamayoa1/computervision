import cv2 as cv
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier
import joblib



def load_data():
    data = []
    labels = []
    data_dir="/Users/estebansamayoa/Desktop/CS UFM/8vo Semestre/Computer Vision/Proyecto 1 ES/CNN letter Dataset"
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path):
            for image_file in glob.glob(os.path.join(class_path, '*.jpg')):
                imagen = cv.imread(image_file)
                imagen = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY) 
                imagen = cv.resize(imagen, (28, 28))
                imagen = imagen.flatten()
                data.append(imagen)
                labels.append(class_folder)  
    data = np.array(data)
    labels=np.array(labels)
    return data, labels

def load_model():
    random_f = RandomForestClassifier(n_estimators = 100, max_features = 45)
    data,labels=load_data()
    random_f.fit(data, labels)
    joblib.dump(random_f, "random_f.joblib")
    if os.path.exists('random_f.joblib'):
        print("Model file 'random_f.joblib' exists.")
    else:
        print("Model file 'random_f.joblib' does not exist.")

load_model()