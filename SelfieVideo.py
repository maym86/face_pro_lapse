#! /usr/bin/env python
import os
import sys
import cv2, cv
import glob
import numpy as np


if __name__ == "__main__":
    imgs = glob.glob("images/*.jpg")
    
    face_cascade1 = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    face_cascade2 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faceHeight = 300
    outSize = (1280 ,720)
    centre = (outSize[0]/2 - faceHeight/2,outSize[1]/2 - faceHeight/2)
    
    for fname in imgs:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade1.detectMultiScale(gray, 1.5, 5)
        faces2 = face_cascade2.detectMultiScale(gray, 1.5, 5)

        if len(faces) > 0 and len(faces2) > 0:
            faces = np.concatenate((faces,faces2))
        elif len(faces) == 0 and len(faces2) == 0:
            continue
        elif len(faces) == 0:
            faces = faces2
        
         
        #sort to get the largest    
        (x,y,w,h) = sorted(faces, key=lambda x: x[2])[0]      

        scale = float(faceHeight) / float(h)
        scaled = cv2.resize(img, (0,0), fx=scale, fy=scale)

        rows,cols = gray.shape
 
        M = np.float32([[1,0,centre[0] - float(x)*scale],[0,1,centre[1] - float(y)*scale]])
        moved = cv2.warpAffine(scaled,M,outSize)
       
        cv2.imshow('face',moved)
        cv2.waitKey(1)


                
            
