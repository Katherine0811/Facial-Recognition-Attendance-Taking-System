import numpy as np
import cv2
import pickle
import os
from PIL import Image

# Face Detection - Haar Cascade
face_cascade = cv2.CascadeClassifier('FYP2/cascades/data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

user_name = 'Wong Kar Lok'

# Image Directory
directory = r'C:\\Users\\kathe\Desktop\\FYP\\LeeYanJie\\FYP2\\Images\\' + user_name

# Change Directorym
os.chdir(directory)

# Initialize Individual Sampling Face Count
lst = os.listdir(directory) # your directory path
count = len(lst) + 1
print(count)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor = 1.05, minNeighbors = 6, minSize = (200, 200))

    for (x, y, w, h) in faces:
        roi_gray = gray[y: y + h, x: x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the Captured Image into the Datasets Folder
        img_item = (user_name.replace(" ", "_") + "_(" + str(count) + ").jpg" )

        cv2.imwrite(img_item, roi_gray)
        cv2.imshow('image', frame)
        count += 1
        print(count)

    if count > 200:  # Take 50 Face Sample and Stop Video
        break
# Image Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get Directory File Location
directory = os.path.join(BASE_DIR, "Images")

# Change Directorym
os.chdir(directory)

cap.release()
cv2.destroyAllWindows()