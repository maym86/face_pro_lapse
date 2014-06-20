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

# Handles mouse events for the manual face selection
#
# @param event		Mouse event
# @param x		X position of pointer
# @param y		Y position of pointer
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
        
# Returns the face rect selected manually.
#
# @param image		Image with face
# @returns box          Rect with face location      
def SelectFaceManually(img):
    global box
    cv2.namedWindow('Click Face')
    cv.SetMouseCallback('Click Face', OnMouse, 0)
    box = []
    while(1):
        temp = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        if box and box[2] != 0 and box[3] != 0:
            cv2.rectangle(temp, (box[0],box[1]) , (box[0] + box[2], box[1] + box[3]),(0,0,255), 2)
        
            if cv2.waitKey(1) == ord('a'):
                cv2.destroyAllWindows()
                return (box[0] * 2, box[1] * 2, box[2] * 2, box[3] * 2)

        cv2.imshow('Click Face', temp)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            return []        

# Normalises the face rect so that it is drawn from the top left
#
# @param rect		Rect with face location      
# @returns rect         Rect with face location with top left origin
def SetRectOrigin(rect):
    (x,y,w,h) = rect 
    if(w < 0):
        x = x + w
        w = -w
        
    if (h < 0):
        y = y + h
        h = -h

    return (x,y,w,h)    

# Finds the largest face in an image
#
# @param gray		    Greyscale image containing face
# @param faceCascades	    Haar cascades
# @param manualMode	    Bool to set whether or not manual mode will be used if face is not found
# @param minHaarFaceSize    The minimum size of the face in px which the Haar cascade will detect
# @returns rect             Rect with face location with top left origin
def FindLargestFace(gray, faceCascades, manualMode, minHaarFaceSize):
    # find faces in the image using all possible Haar cascades
    faces = []
    for cascade in faceCascades:
        f = cascade.detectMultiScale(gray, minSize=minHaarFaceSize)
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

# Finds the angle between x axis and a vector connecting the eyes
#
# @param gray		    Greyscale image containing face
# @param face	            Rect defining the face location
# @returns eyeAngle         Value in radians defining the angle 
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

# Positions the image so the face is located at the centre
#
# @param img		    Image containing face
# @param face	            Rect defining the face location
# @param eyeAngle           Value in radians defining the angle
# @returns img              Image with face rotated and centred in the image
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

# Resizes the image intlegently for face detection based on the height and width of the target video
#
# @param img		    Image containing face
# @param videoSize          The size of the target video
# @returns img              Scaled image
def ResizeFrame(img, videoSize):
    scl = 1
    if(img.shape[0] > img.shape[1]):
        scl = float(videoSize[0]) / float(img.shape[0])
    else:
        scl = float(videoSize[1])/ float(img.shape[1])
    img = cv2.resize(img, (0,0), fx=scl, fy=scl)
    return img

# Performs histogram equalisation on the image using the CLAHE method
#
# @param img		    Image containing face
# @param clipLimit          The CLAHE clip limit
# @returns img              Equalised image
def HistEqualisationColour(img, clipLimit):
    ycrcb = cv2.cvtColor(img, cv.CV_BGR2YCrCb)
    y,Cr,Cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    y = clahe.apply(y)
    #y = cv2.equalizeHist(y);
    ycrcb = cv2.merge((y,Cr,Cb))
    return cv2.cvtColor(ycrcb,cv.CV_YCrCb2BGR);

# Set the gamma of the image
#
# @param img		    Image containing face
# @param gamma              The gamma value to be set
# @returns img              Image with new gamma value
def SetGamma(img, gamma):
    img = img/255.0
    img = cv2.pow(img, gamma)
    return np.uint8(img*255)


if __name__ == "__main__":
    imgs = glob.glob("images/*.jpg")

    #Parameters
    fps = 8.0
    faceHeight = 400
    videoSize = (1280 ,720)
    manualMode = True
    correctColour = False #Experimental
    #min size of the face found by the Haar cascade
    minHaarFaceSize = (200,200)
    #####

    
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
        img = ResizeFrame(img, videoSize)

        #Experimental       
        if correctColour:
            img = SetGamma(img,0.8)
            img = HistEqualisationColour(img, 2.0)
                    
        #border added for faces at edges
        border = int(faceHeight/2)
        img= cv2.copyMakeBorder(img,border,border,border,border,cv2.BORDER_CONSTANT)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        face = FindLargestFace(gray, faceCascades, manualMode, minHaarFaceSize)
        
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
                
            
