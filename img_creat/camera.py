import cv2


cam = cv2.VideoCapture(0)


while True:
    tes, img = cam.read()
    
    cv2.imshow("img",img)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    
cam.release()
cv2.destroyAllWindows()