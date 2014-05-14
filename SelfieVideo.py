#! /usr/bin/env python
import os
import sys
import cv2, cv
import glob
import numpy as np
import math

if __name__ == "__main__":
    imgs = glob.glob("images/*.jpg")
    
    fps = 4.0
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

        # find faces in the image using all possible Haar cascades
        faces = []
        for cascade in face_cascade:
            f = cascade.detectMultiScale(gray)
            if len(f) > 0:
                if len(faces) == 0:
                    faces = f
                else:
                    faces = np.concatenate((faces,f))    


        if len(faces) == 0:
            continue
        
        
        #Sort to get the largest face  
        face = sorted(faces, key=lambda x: x[3])[-1]      
        (x,y,w,h) = face


        #try to find eyes to get head rotation
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            eyeAngle = math.atan((eyes[0][1] - eyes[1][1])/(eyes[0][0] - eyes[1][0]))
        else:
            eyeAngle = 0


        #rescale to desired face size  
        scale = float(faceHeight) / float(face[3])
        scaled = cv2.resize(img, (0,0), fx=scale, fy=scale)

        #rescale face rect
        (x,y,w,h) = tuple([scale*x for x in face])

        #move the image so the face is centred
        M = np.float32([[1,0,(centre[0] -w/2) - x],[0,1,(centre[1] -h/2) - y]])
        out = cv2.warpAffine(scaled,M,videoSize)

        #rotate if angle eye angle is off centre
        if eyeAngle:
            rot_mat = cv2.getRotationMatrix2D(centre,eyeAngle,1.0)
            out = cv2.warpAffine(out, rot_mat, videoSize)

        #write to video file      
        video.write(out)
        cv2.imshow('face',out)
        cv2.waitKey(100)
        

    video.release()
    cv2.destroyAllWindows()
                
            
