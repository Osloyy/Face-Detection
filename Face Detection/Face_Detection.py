

import cv2

face_ai = cv2.CascadeClassifier('haarcascade_fontalface_default.xml')
video = cv2.VideoCapture(0)
while True:
    fret, img = video.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_ai.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255),2)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()