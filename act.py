# Selección de ROI
import numpy as np
import cv2 as cv
from umucv.util import ROI, putText
from umucv.stream import autoStream
import time

cv.namedWindow("input")
cv.moveWindow('input', 0, 0)

region = ROI("input")
background = None
detec = False
grabar = False

# Ajustes de video
start = 0
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi',fourcc, 20.0, (640,480))

for key, frame in autoStream():
    
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        if key == ord('c'):
            trozo = frame[y1:y2+1, x1:x2+1]
            cv.imshow("trozo", trozo)
        if key == ord('x'):
            region.roi = []
            background = None
        if key == ord('d'):
            detec = True
        if detec:
            frameROI = frame[y1:y2+1, x1:x2+1]
            gray = cv.cvtColor(frameROI, cv.COLOR_BGR2GRAY)
            gray = cv.GaussianBlur(gray, (21,21), 0)

            if background is None:
                background = gray

            if grabar:
                start = time.time()
                grabar = False

            if time.time() - start <= 2:
                #Guardar frame
                out.write(frame)

            cv.imshow('roi', gray)
            cv.imshow('back', background)
            
            # Deteccion de movimiento
            subtraction = cv.absdiff(background, gray)
            # Anulacion del fondo
            cv.imshow("subs", subtraction)

            threshold = cv.threshold(subtraction, 25, 255, cv.THRESH_BINARY) [1]
            threshold = cv.dilate(threshold, None, iterations=2) 
            
            countorimg = threshold.copy()
            
            im, outlines, hierarchy = cv.findContours(countorimg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for c in outlines:
                if cv.contourArea(c) < 500:
                    continue

                (x,y,w,h) = cv.boundingRect(c)
                (x,y,w,h) = (x+x1,y+y1,w+x2,h+y2)
                cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

                # Grabar 2 o 3 sec del video
                grabar = True

        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))

    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)

