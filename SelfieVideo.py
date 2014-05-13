#! /usr/bin/env python
import os
import sys
import cv2, cv
import glob
import numpy as np
import math

if __name__ == "__main__":
    imgs = glob.glob("images/*.jpg")
    
    fps = 2.0
    faceHeight = 300
    videoSize = (1280 ,720)
    centre = (videoSize[0]/2,videoSize[1]/2)
    
    face_cascade = []

    face_cascade.append(cv2.CascadeClassifier('haarcascade_frontalface_alt.xml'))
    face_cascade.append(cv2.CascadeClassifier('haarcascade_frontalface_default.xml'))
    face_cascade.append(cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml'))

    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
    video = cv2.VideoWriter('video.avi',fourcc, fps, videoSize)
    
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
        (x,y,w,h) = face
        
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            eyeAngle = math.atan((eyes[0][1] - eyes[1][1])/(eyes[0][0] - eyes[1][0]))
        else:
            eyeAngle = 0

                
        scale = float(faceHeight) / float(face[3])
        scaled = cv2.resize(img, (0,0), fx=scale, fy=scale)

        #rescale face
        (x,y,w,h) = tuple([scale*x for x in face])
        
        M = np.float32([[1,0,(centre[0] -w/2) - x],[0,1,(centre[1] -h/2) - y]])
        moved = cv2.warpAffine(scaled,M,videoSize)


        rot_mat = cv2.getRotationMatrix2D(centre,eyeAngle,1.0)
        rotated = cv2.warpAffine(moved, rot_mat, videoSize)
  
        video.write(rotated)
        cv2.imshow('face',rotated)
        cv2.waitKey(100)
        

    video.release()
    cv2.destroyAllWindows()
                
            
