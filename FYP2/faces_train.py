import cv2
import os
import numpy as np
from PIL import Image
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Get Directory File Location
image_dir = os.path.join(BASE_DIR, "Images")

face_cascade = cv2.CascadeClassifier('FYP2/Cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0 # ID
label_ids = {} # dictionary
x_train = [] # numbers of pixel value
y_labels = [] # numbers related to label

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            # print(label, path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            
            id_ = label_ids[label]
            # print(label_ids)

            # x_train.append(path) # Verify Image, turn into a NUMPY array, colored gray
            # y_lables.append(label) # Add Number

            pil_image = Image.open(path).convert("L") # grayscale
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)

            image_array = np.array(final_image, "uint8")
            # print(image_array) # converts image into numbers
            faces = face_cascade.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 5)

            for (x, y, w, h) in faces:
                roi = image_array[y: y + h, x: x + w]
                x_train.append(roi)
                print(type(id_))
                y_labels.append(id_)

print(y_labels)
print(x_train)

with open("labels.pickle",'wb') as f: # writing byter
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")