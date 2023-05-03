from itertools import count
import os
from pathlib import Path
import cv2



img_path =(r"E:\Documents\Projects\face project 2\dataset")
#vedio capture
cap= cv2.VideoCapture(0)

#frame reaiding
cascad = (r"E:\Documents\Projects\face project 2\cascade\frontFace.xml")
face_detect = cv2.CascadeClassifier(cascad)

count = 0
while True:
    ret , img =  cap.read()
    print(ret)
    
    #changing camera vertically
    img = cv2.flip(img,+1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    
    faces = face_detect.detectMultiScale(gray,scaleFactor =1.3,minNeighbors = 5)
    for (x,y,w,h) in faces:
        count += 1
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
        roi_color = img[y:y+h, x:x+w]
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
        print(len(faces))
        cv2.imwrite("testz.png",roi_gray)
        
    # cv2.imshow('normal vedio',gray)
    
    cv2.imshow('gray image',gray)
    
    if cv2.waitKey(20) & 0xFF ==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

# p.mkdir()
# data = []
def file():
    
    user =input("enter dir name : ")
    pat = os.path.join(img_path,user)
    def check():
        with open(pat + "/dta.txt",mode="w") as file:
            
            dat = "kjjkhkhkjhfhjk"
            file.write(dat)
            
    data = os.listdir(img_path)
    print(data)

    if user in data:
        # print(data)
        check()
    else:
        p = Path(f"{img_path}/{user}")
        p.mkdir()
        check()
        # print(data)
        
        test = data.append(user)
        print(data)
        
    #     test = data.append
    # for dir_names in data:
    #     print(dir_names)
    #     if "cololll" in dir_names:
    #         print(dir_names)