#!/usr/bin/env python

import cv2 as cv
from umucv.stream import autoStream
from collections import deque
import numpy as np
from umucv.util import putText

#https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd

points = deque(maxlen=2)

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))

def getAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

cv.namedWindow("webcam")
cv.setMouseCallback("webcam", fun)

for key, frame in autoStream():
    dim = frame.shape
    w = dim[0]
    h = dim[1]
    centro = (int(h/2), int(w/2))
    #print(centro)
    cv.circle(frame, centro, 3,(0,0,255),-1)

    for p in points:
        cv.circle(frame, p,3,(0,0,255),-1)
    if len(points) == 2:
        cv.line(frame, points[0],points[1],(0,0,255))
        c = np.mean(points, axis=0).astype(int)
        angulo = getAngle(points[0], centro, points[1])

        f = 3.06144885e+03
        vector1 = [(points[0][0] - (frame.shape[0]/2)), (points[0][1] - (frame.shape[1]/2)), f]
        vector2 = [(points[1][0] - (frame.shape[0]/2)), (points[1][1] - (frame.shape[1]/2)), f]
        centro = [(- (frame.shape[0]/2)), (- (frame.shape[1]/2)), f]


        angulo = getAngle(vector1, centro, vector2)

        putText(frame,f'{angulo:.1f} grados',c)

    cv.imshow('webcam',frame)
    
cv.destroyAllWindows()

