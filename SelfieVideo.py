#! /usr/bin/env python
import os
import sys
import cv2, cv
import glob
import numpy as np


if __name__ == "__main__":
    imgs = glob.glob("images/*.jpg")
    
    face_cascade = []

    face_cascade.append(cv2.CascadeClassifier('haarcascade_frontalface_alt.xml'))
    face_cascade.append(cv2.CascadeClassifier('haarcascade_frontalface_default.xml'))
    face_cascade.append(cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml'))

    faceHeight = 300
    outSize = (1280 ,720)
    centre = (outSize[0]/2,outSize[1]/2)

    fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
    video = cv2.VideoWriter('video.avi',fourcc, 1.0, outSize)
    
    for fname in imgs:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = []
        for cascade in face_cascade:
            f = cascade.detectMultiScale(gray, 1.5, 5)
            if len(f) > 0:
                if len(faces) == 0:
                    faces = f
                else:
                    faces = np.concatenate((faces,f))    


        if len(faces) == 0:
            continue
        
         
        #sort to get the largest    
        face = sorted(faces, key=lambda x: x[3])[-1]      

        scale = float(faceHeight) / float(face[3])
        scaled = cv2.resize(img, (0,0), fx=scale, fy=scale)

        #rescale face
        (x,y,w,h) = tuple([scale*x for x in face])
        
        M = np.float32([[1,0,(centre[0] -w/2) - x],[0,1,(centre[1] -h/2) - y]])
        moved = cv2.warpAffine(scaled,M,outSize)

        video.write(moved)
        cv2.imshow('face',moved)
        cv2.waitKey(100)
        

    video.release()
    cv2.destroyAllWindows()
                
            
