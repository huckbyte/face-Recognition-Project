import cv2
import os
import numpy as np
from PIL import Image
import pickle


image_dir = (r"E:\Documents\Projects\face project 2\dataset")
face_cascade_path = (r"E:\Documents\Projects\face project 2\cascade\haarcascade_frontalface_alt2.xml")

img_ext = (".jpg", ".jpeg", ".jpe", ".jif", ".jfif", ".jfi", ".png", ".gif", ".webp", ".tiff", ".tif", ".psd", ".raw", ".arw", ".cr2", ".nrw",
                    ".k25", ".bmp", ".dib", ".heif", ".heic", ".ind", ".indd", ".indt", ".jp2", ".j2k", ".jpf", ".jpf", ".jpx", ".jpm", ".mj2", ".svg", ".svgz", ".ai", ".eps", ".ico")

face_cascade = cv2.CascadeClassifier(face_cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []


for root, dirs, files in os.walk(image_dir):
    
    for file in files:
        if file.endswith(img_ext):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)
            
            pil_image = Image.open(path).convert("L") # grayscale
            size = (300, 300)
            
            #?ANTIALIAS is deprecated and will be removed in Pillow 10 (2023-07-01) use Resampling.LANCZOS instead of lanczos
            # final_image = pil_image.resize(size, Image.ANTIALIAS) 
            final_image = pil_image.resize(size, Image.Resampling.LANCZOS)
            image_array = np.array(final_image, "uint8")
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            # print(faces)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
            
# print(y_labels)
# print(x_train)

with open("face_labels.npy", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("face_trainner.yml")