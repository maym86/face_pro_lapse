#! /usr/bin/env python3
# SelfieVideo.py
# Detect largest face in an image and align to create a time lapse video
# by Michael May (http://maym86.com/) maym86@gmail.com

import cv2
import cv
import glob
import numpy as np
import math

box = []
drawing_box = False


def on_mouse(event, x, y, flags, params):
    """
    Handles mouse events for the manual face selection
    @param event		Mouse event
    @param x		X position of pointer
    @param y		Y position of pointer
    """
    global box, drawing_box
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_box = True
        print('Start Mouse Position: ' + str(x) + ', ' + str(y))
        box = (x, y, 0, 0)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing_box = False
        print('End Mouse Position: ' + str(x) + ', ' + str(y))
        box = (box[0], box[1], x - box[0], y - box[1])
        print(box)
    elif event == cv2.EVENT_MOUSEMOVE and drawing_box:
        box = (box[0], box[1], x - box[0], y - box[1])


def select_face_manually(image, border):
    """
    Returns the face rect selected manually.
    Detection Failed. Select face with mouse. Press 'a' to accept.
    @param image		Image with face
    @returns box          Rect with face location
    """
    global box
    cv2.namedWindow('Click Face')
    cv2.setMouseCallback('Click Face', on_mouse, 0)
    box = []
    while 1:
        temp = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        if box and box[2] != 0 and box[3] != 0:
            cv2.rectangle(
                temp, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 2)

            if cv2.waitKey(1) == ord('a'):
                cv2.destroyAllWindows()
                return (box[0] * 2) - border, (box[1] * 2) - border, box[2] * 2, box[3] * 2

        cv2.imshow('Click Face', temp)
        if cv2.waitKey(1) == 1:
            cv2.destroyAllWindows()
            return []


def set_rect_origin(rect):
    """
    Normalises the face rect so that it is drawn from the top left

    @param rect		Rect with face location
    @returns rect         Rect with face location with top left origin
    """
    (x, y, w, h) = rect
    if w < 0:
        x = x + w
        w = -w

    if h < 0:
        y = y + h
        h = -h

    return x, y, w, h


def find_largest_face(gray, face_cascades, manual_mode, min_haar_face_size, colour_img, border):
    """
    Finds the largest face in an image using all possible Haar cascades

    @param gray		    Greyscale image containing face
    @param faceCascades	    Haar cascades
    @param manualMode	    Bool to set whether or not manual mode will be used if face is not found
    @param minHaarFaceSize    The minimum size of the face in px which the Haar cascade will detect
    @returns rect             Rect with face location with top left origin
    """
    faces = []
    for cascade in face_cascades:
        face = cascade.detectMultiScale(gray, minSize=min_haar_face_size)
        if len(face) > 0:
            if len(faces) == 0:
                faces = face
            else:
                faces = np.concatenate((faces, face))

                # if no face is found manually select face
    if len(faces) == 0 and manual_mode:
        face = select_face_manually(colour_img, border)
        print(face)
    elif len(faces) == 0:
        return []
    else:
        # Sort to get the largest face
        face = sorted(faces, key=lambda x: x[3])[-1]

    return face


def find_eye_angle(gray, face, eye_cascade):
    """
    Finds the angle between x axis and a vector connecting the eyes

    @param gray		    Greyscale image containing face
    @param face	            Rect defining the face location
    @returns eyeAngle         Value in radians defining the angle
    """
    (x, y, w, h) = face
    # try to find eyes to get head rotation
    face_roi_gray = gray[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(face_roi_gray)
    if len(eyes) >= 2:
        eye_angle = math.atan(
            (eyes[0][1] - eyes[1][1]) / (eyes[0][0] - eyes[1][0]))
    else:
        eye_angle = 0
    return eye_angle


def position_image(img, face, eye_angle, face_height, video_centre, video_out_size):
    """
    Positions the image so the face is located at the centre

    @param img		    Image containing face
    @param face	            Rect defining the face location
    @param eyeAngle           Value in radians defining the angle
    @returns img              Image with face rotated and centred in the image
    """
    face = set_rect_origin(face)

    # rescale to desired face size
    scale = float(face_height) / float(abs(face[3]))
    scaled = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    # rescale face rect
    (x, y, w, h) = [scale * x for x in face]

    # move the image so the face is centred
    translate = np.float32(
        [[1, 0, (video_centre[0] - w / 2) - x], [0, 1, (video_centre[1] - h / 2) - y]])
    translated = cv2.warpAffine(scaled, translate, video_out_size)

    # rotate around the centre to eye angle
    if eye_angle:
        rotate = cv2.getRotationMatrix2D(video_centre, eye_angle, 1.0)
        rotated = cv2.warpAffine(translated, rotate, video_out_size)
        return rotated

    return translated


def resize_frame(img, videoSize):
    """
    Resizes the image intelligently for face detection based on the height and width of the target video

    @param img		    Image containing face
    @param videoSize          The size of the target video
    @returns img              Scaled image
    """
    scl = 1
    if img.shape[0] > img.shape[1]:
        scl = float(videoSize[0]) / float(img.shape[0])
    else:
        scl = float(videoSize[1]) / float(img.shape[1])
    img = cv2.resize(img, (0, 0), fx=scl, fy=scl)
    return img


def hist_equalisation_colour(img, clipLimit):
    """
    Performs histogram equalisation on the image using the CLAHE method

    @param img		    Image containing face
    @param clipLimit          The CLAHE clip limit
    @returns img              Equalised image
    """
    ycrcb = cv2.cvtColor(img, cv.CV_BGR2YCrCb)
    y, Cr, Cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    y = clahe.apply(y)
    # y = cv2.equalizeHist(y);
    ycrcb = cv2.merge((y, Cr, Cb))
    return cv2.cvtColor(ycrcb, cv.CV_YCrCb2BGR)


def set_gamma(img, gamma):
    """
    Set the gamma of the image

    @param img		    Image containing face
    @param gamma              The gamma value to be set
    @returns img              Image with new gamma value
    """
    img /= 255.0
    img = cv2.pow(img, gamma)
    return np.uint8(img * 255)


def main():
    images = glob.glob("images/*.jpg")

    # Parameters
    fps = 15.0
    face_height = 400
    video_out_size = (1280, 720)
    manual_mode = True
    correct_colour = False  # Experimental
    # min size of the face found by the Haar cascade
    min_haar_face_size = (200, 200)

    video_centre = (video_out_size[0] / 2, video_out_size[1] / 2)
    face_cascades = [
        cv2.CascadeClassifier('haarcascade_frontalface_alt.xml'),
        cv2.CascadeClassifier('haarcascade_frontalface_default.xml'),
        cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml'),
    ]

    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = cv2.VideoWriter('video.avi', fourcc, fps, video_out_size)

    for file_name in images:
        img = cv2.imread(file_name)
        img = resize_frame(img, video_out_size)

        # Experimental
        if correct_colour:
            img = set_gamma(img, 0.8)
            img = hist_equalisation_colour(img, 2.0)

        # border added for faces at edges
        border = int(face_height / 2)
        colour_img = cv2.copyMakeBorder(
            img, border, border, border, border, cv2.BORDER_CONSTANT)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        face = find_largest_face(
            gray, face_cascades, manual_mode, min_haar_face_size, colour_img, border)

        if face is []:
            continue

        eye_angle = find_eye_angle(gray, face, eye_cascade)
        out = position_image(img, face, eye_angle,
                             face_height, video_centre, video_out_size)

        # write to video file
        video.write(out)
        cv2.imshow('face', out)
        cv2.waitKey(1)

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
