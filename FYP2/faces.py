import numpy as np
import cv2
import pickle
import os
from PIL import Image

# Face Detection - Haar Cascade
face_cascade = cv2.CascadeClassifier('FYP2/cascades/data/haarcascade_frontalface_default.xml')

# Train Model - LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

# What do you do?
labels = {"person_name": 1}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}
    # print(labels)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Open Camera
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Haar Cascades works in gray image
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)

    for (x, y, w, h) in faces: # Detect Faces
        # print (x, y, w, h)
        roi_gray = gray[y: y + h, x: x + w] # Coordinates where w = width and h = height
        roi_color = frame[y: y + h, x: x + w] # [] = Location of this frame
        
        # Recognizer (Deep Learned Model Predict)
        id_, conf = recognizer.predict(roi_gray)
        if 60 <= conf <= 75: # confidence level 0 = exact match
            # print(id_)
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            
        # else:
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     name = "Unknown"

            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, str(conf), (x+w, y+h), font, 1, color, stroke, cv2.LINE_AA)

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        # Draw Rectangle
        color = (255, 0, 0) # BGR
        stroke = 2 # Thickness
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame (color)
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

