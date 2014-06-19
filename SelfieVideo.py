#! /usr/bin/env python
# SelfieVideo.py 
# Detect largest face in an image and align to create a time lapse video
# by Michael May (http://maym86.com/) maym86@gmail.com


import os
import sys
import cv2, cv
import glob
import numpy as np
import math

box = []
drawingBox = False

def OnMouse(event, x, y, flags, params):
    global box, drawingBox
    if event == cv.CV_EVENT_LBUTTONDOWN:
        drawingBox = True
        print 'Start Mouse Position: '+str(x)+', '+str(y)
        box = (x, y , 0, 0)
    elif event == cv.CV_EVENT_LBUTTONUP:
        drawingBox = False
        print 'End Mouse Position: '+str(x)+', '+str(y)
        box = (box[0], box[1] , x - box[0] ,y - box[1])        
        print box
    elif event == cv.CV_EVENT_MOUSEMOVE and drawingBox:
        box = (box[0], box[1] , x - box[0] ,y - box[1])
        
        
def SelectFaceManually(img):
    global box
    cv2.namedWindow('Click Face')
    cv.SetMouseCallback('Click Face', OnMouse, 0)
    while(1):
        temp = img.copy()
        if box:
            cv2.rectangle(temp, (box[0],box[1]) , (box[0] + box[2], box[1] + box[3]),(0,0,255), 2)
        cv2.imshow('Click Face', temp)
        if cv2.waitKey(1) == ord('a'):
            face = box
            cv2.destroyAllWindows()
            box = []
            return face
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            box = []
            return []
        

def FindLargestFace(gray, faceCascades, manualMode):
    # find faces in the image using all possible Haar cascades
    faces = []
    for cascade in faceCascades:
        f = cascade.detectMultiScale(gray, minSize=(200,100))
        if len(f) > 0:
            if len(faces) == 0:
                faces = f
            else:
                faces = np.concatenate((faces,f))    

    #if no face is found manually select face
    if len(faces) == 0 and manualMode:
        face = SelectFaceManually(img)
    elif len(faces) == 0:
        return []
    else:
        #Sort to get the largest face  
        face = sorted(faces, key=lambda x: x[3])[-1]

    return face


def FindEyeAngle(gray, face):
    (x,y,w,h) = face
    #try to find eyes to get head rotation
    faceROIGray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(faceROIGray)
    if len(eyes) >= 2:
        eyeAngle = math.atan((eyes[0][1] - eyes[1][1])/(eyes[0][0] - eyes[1][0]))
    else:
        eyeAngle = 0        
    return eyeAngle


def HistEqualisationColour(img):
    ycrcb = cv2.cvtColor(img, cv.CV_BGR2YCrCb)
    y,Cr,Cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y = clahe.apply(y)
    #y = cv2.equalizeHist(y);
    ycrcb = cv2.merge((y,Cr,Cb))
    return cv2.cvtColor(ycrcb,cv.CV_YCrCb2BGR);

#if rect is not drawn from top left set so it is
def SetRectOrigin(rect):

    (x,y,w,h) = rect 
    if(w < 0):
        x = x + w
        w = -w
        
    if (h < 0):
        y = y + h
        h = -h

    return (x,y,w,h)

def PositionImage(img, face, eyeAngle):

    face = SetRectOrigin(face)   

    #rescale to desired face size
    scale = float(faceHeight) / float(abs(face[3]))
    scaled = cv2.resize(img, (0,0), fx=scale, fy=scale)

    #rescale face rect
    (x,y,w,h) = [scale*x for x in face]

    #move the image so the face is centred
    translate = np.float32([[1,0,(centre[0] -w/2) - x],[0,1,(centre[1] -h/2) - y]])
    translated = cv2.warpAffine(scaled,translate,videoSize)
    
    #rotate around the centre to eye angle
    if eyeAngle:
        rotate = cv2.getRotationMatrix2D(centre,eyeAngle,1.0)
        rotated = cv2.warpAffine(translated, rotate, videoSize)
        return rotated

    return translated


if __name__ == "__main__":
    imgs = glob.glob("images/*.jpg")

    #Parameters
    fps = 4.0
    faceHeight = 300
    videoSize = (1280 ,720)
    manualMode = True
    histEq = True
    
    centre = (videoSize[0]/2,videoSize[1]/2)    
    faceCascades = []

    faceCascades.append(cv2.CascadeClassifier('haarcascade_frontalface_alt.xml'))
    faceCascades.append(cv2.CascadeClassifier('haarcascade_frontalface_default.xml'))
    faceCascades.append(cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml'))

    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
    video = cv2.VideoWriter('video.avi',fourcc, fps, videoSize)
    
    for fname in imgs:
        img = cv2.imread(fname)
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        if histEq:
            img = HistEqualisationColour(img)
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face = FindLargestFace(gray, faceCascades, manualMode)
        
        if face == []:
            continue
        
        eyeAngle = FindEyeAngle(gray, face)
        out = PositionImage(img, face, eyeAngle)

        #write to video file      
        video.write(out)
        cv2.imshow('face',out)
        cv2.waitKey(1)
 
    video.release()
    cv2.destroyAllWindows()
                
            
