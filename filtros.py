import numpy as np
import cv2 as cv

from umucv.util import ROI, putText
from umucv.stream import autoStream

def nada(v):
    pass

cv.namedWindow("input")
cv.createTrackbar("Size", "input", 5, 20, nada)
cv.moveWindow('input', 0, 0)

region = ROI("input")

for key, frame in autoStream():
    
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        if key == ord('c'):
            trozo = frame[y1:y2+1, x1:x2+1]
            cv.imshow("trozo", trozo)
        if key == ord('x'):
            region.roi = []
        if key == ord('b'):
            # FILTRO BLUR
            size = cv.getTrackbarPos("Size", "input")
            if size == 0:
                size = 1
            frame[y1:y2+1, x1:x2+1] = cv.boxFilter(frame[y1:y2+1, x1:x2+1], -1, (size,size))
        if key == ord('g'):
            # FILTRO GAUSSIAN BLUR
            size = cv.getTrackbarPos("Size", "input")
            if size == 0:
                size = 1
            frame[y1:y2+1, x1:x2+1] = cv.GaussianBlur(frame[y1:y2+1, x1:x2+1],(0,0), size)
        if key == ord('v'):
            size = cv.getTrackbarPos("Size", "input")
            if size == 0:
                size = 1
            
            kernel = np.array([[-1.0, -1.0], 
                               [2.0, 2.0],
                               [-1.0, -1.0]])

            kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)
            kernel = kernel * size

            frame[y1:y2+1, x1:x2+1] = cv.filter2D(frame[y1:y2+1, x1:x2+1], -1, kernel)
        

        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))

    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)