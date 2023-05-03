import pickle
from tkinter import *
import numpy as np
import cv2
import PIL
from PIL import Image,ImageTk


cap = cv2.VideoCapture(1)
width, height = 300, 300
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

face_trained_yml = (r"E:\Documents\Projects\face project 2\trainer\face_trainner.yml")
face_label_pickel = (r"E:\Documents\Projects\face project 2\trainer\face_labels.pickle")
face_cascade_path = (r"E:\Documents\Projects\face project 2\cascade\haarcascade_frontalface_alt2.xml")


face_cascade = cv2.CascadeClassifier(face_cascade_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(face_trained_yml)


labels = {"person_name": 0}

with open(face_label_pickel, 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}




class Camera:
    def __init__(self) -> None:
        self.window = Tk()
        self.class_itiems()
        self.show_frame()
        
        
    def run(self):
        self.window.mainloop()
        
    def class_itiems(self):
        self.window.title("camera")
        self.window.configure(width=500,height=500)
        self.window.resizable(width=False,height=False)
        
        self.cam_label = Label(self.window)
        self.cam_label.place(x=50,y=50)
        
    def show_frame(self):
        while True:
            
            success , frame = cap.read()
            print(success)
            frame  = cv2.flip(frame,1)
            frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)
            img = PIL.Image.fromarray(frame1)  # type: ignore
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk  # type: ignore
            self.cam_label.configure(image=imgtk)
            self.cam_label.after(1,self.show_frame)
            cv2.imshow("frma",frame)
            print(man)
            for (x, y, w, h) in faces:
                
                roi_gray = frame[y:y+h, x:x+w] #(ycord_start, ycord_end)
                roi_color = frame[y:y+h, x:x+w]
                color = (0, 255, 0) #BGR 0-255 
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
            
            # man = cv2.imshow("frma",frame)
            frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            img = PIL.Image.fromarray(frame1)  # type: ignore
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk  # type: ignore
            self.cam_label.configure(image=imgtk)
            self.cam_label.after(1,self.show_frame)
    
        
        


if __name__ == "__main__":
    app = Camera()
    app.run()
