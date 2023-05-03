from pathlib import Path
import cv2
import os

img_path =(r"E:\Documents\Projects\face project 2\dataset") 
cascad = (r"E:\Documents\Projects\face project 2\cascade\frontFace.xml")
#img = cv2.imread("Cascade/test.jpg")
cap = cv2.VideoCapture(0)

#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face_detect = cv2.CascadeClassifier(cascad)

# faces = face_detect.detectMultiScale(gray,scaleFactor =1.1,minNeighbors = 3)
user_name = input('enter your name : ')
count = 0

def dir_check():
    pat = os.path.join(img_path,user_name)
    dat = os.listdir(img_path)
    if user_name in dat:
        try:
            # cv2.imwrite( f"{pat}/{user_name}{count}.jpg",Frame[x:x+w,y:y+h])
            cv2.imwrite( f"{pat}/{user_name}{count}.jpg",roi_color)
            
        except:
            
            print("(-215:Assertion failed) !_img.empty() in function 'cv::imwrite'")
        
    else:
        try:
            p = Path(f"{img_path}/{user_name}")
            p.mkdir()
            # cv2.imwrite( f"{pat}/{user_name}{count}.jpg",Frame[x:x+w,y:y+h])
            cv2.imwrite( f"{pat}/{user_name}{count}.jpg",roi_color)
            
        except:
            
            print("(-215:Assertion failed) !_img.empty() in function 'cv::imwrite'")
        
        

while True:
    ret,Frame = cap.read()
    # Frame = cv2.resize(Frame,(00,300))
    print(ret)
    gray = cv2.cvtColor(Frame,cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray,scaleFactor =1.2,minNeighbors = 3)
    
    for (x,y,w,h) in faces:
        count += 1
        cv2.rectangle(Frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = Frame[y:y+h, x:x+w]
        # cv2.imwrite( img_path + "/" + str(user_name) +  str(count) + '.jpg' ,gray[x:x+w,y:y+h])
        dir_check()
        #cv2.imshow('detected image',img)
        cv2.imshow('detected video',Frame)
        
        
    if count == 450:
        break
    elif cv2.waitKey(1) & 0xFF == 27:
        break
    
        
print(f'number of faces found  = {len(faces)}')
cap.release()
cv2.destroyAllWindows()


