import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
from pathlib import Path


path='dataset\images'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# method/function to recognise image from dataset
def getImageAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    facesamples = []
    ids=[]
    for imagePath in imagePaths:
        # opening and converting it into grayscale image
        # 'L' mode maps to black and white pixels (and in between)
        PIL_image=Image.open(imagePath).convert('L')
        img_numpy=np.array(PIL_image, 'uint8')

        id=int(os.path.split(imagePath))
        # detectMultiScale() [1/3] Detects objects of different sizes 
        # in the input image. The detected objects are returned as a list of rectangles. 
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            facesamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
        
    return facesamples,ids

print("\n [INFO] Training faces...")
faces, ids=getImageAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml')

print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))
