#!/usr/bin/env python

# estimación de pose a partir del marcador images/ref.png
# En esta versión el objeto virtual se mueve.

# pruébalo con el vídeo de siempre

# ./pose2.py --dev=file:../images/rot4.mjpg

# con la imagen de prueba

# ./pose2.py --dev=--dir:../../images/marker.png

# o con la webcam poniéndolo en el teléfono o el monitor.

import cv2          as cv
import numpy        as np

from umucv.stream   import autoStream
from umucv.htrans   import htrans, Pose
from umucv.util     import cube, showAxes
from umucv.contours import extractContours, redu
from collections import deque


def Kfov(sz,hfovd):
    hfov = np.radians(hfovd)
    f = 1/np.tan(hfov/2)
    # print(f)
    w,h = sz
    w2 = w / 2
    h2 = h / 2
    return np.array([[f*w2, 0,    w2],
                     [0,    f*w2, h2],
                     [0,    0,    1 ]])


stream = autoStream()

HEIGHT, WIDTH = next(stream)[1].shape[:2]
size = WIDTH,HEIGHT


K = Kfov( size, 60 )


marker = np.array(
       [[0,   0,   0],
        [0,   1,   0],
        [0.5, 1,   0],
        [0.5, 0.5, 0],
        [1,   0.5, 0],
        [1,   0,   0]])

square = np.array(
       [[0,   0,   0],
        [0,   1,   0],
        [1,   1,   0],
        [1,   0,   0]])



def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if len(r) == n ]

def rots(c):
    return [np.roll(c,k,0) for k in range(len(c))]

def bestPose(K,view,model):
    poses = [ Pose(K, v.astype(float), model) for v in rots(view) ]
    return sorted(poses,key=lambda p: p.rms)[0]

points = deque(maxlen=2)
cubo = cube
nombres = ["MENOS", "MAS"]
aumentoMax = 20
aumentoActual = 10

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))

cv.namedWindow("source")
cv.setMouseCallback("source", fun)

for n, (key,frame) in enumerate(stream):

    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5, reduprec=2)

    good = polygons(cs,6,3)
    poses = []
    for g in good:
        p = bestPose(K,g,marker)
        if p.rms < 2:
            poses += [p.M]

    for M in poses:

        # capturamos el color de un punto cerca del marcador para borrarlo
        # dibujando un cuadrado encima
        x,y = htrans(M, (0.7,0.7,0) ).astype(int)
        b,g,r = frame[y,x].astype(int)
        cv.drawContours(frame,[htrans(M,square*2+(-0.5,-0.5,0)).astype(int)], -1, (255,255,255) , -1, cv.LINE_AA)
        #cv.drawContours(frame,[htrans(M,marker).astype(int)], -1, (0,0,0) , 3, cv.LINE_AA)
        #print(str(x) + " " + str(y))
        limites = htrans(M,square*2+(-0.5,-0.5,0)).astype(int)
        
        for limit in limites:
            cv.circle(frame, (limit[0],limit[1]), 1, (0,0,0), 10)
        
        contador = 0
        for point in points:
            cv.circle(frame, (point[0], point[1]), 1, (0,0,255),10)
            cv.putText(frame, nombres[contador], (point[0], point[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,255),2, cv.LINE_AA)
            contador += 1
        
        if len(points) == 2:
            if points[0][0] >= limites[1][0] and points[0][0] <= limites[0][0] and points[0][1] >= limites[2][1] and points[0][1] <= limites[0][1]:
                # Reducir altura del cubo
                if aumentoActual > 0:
                    cubo = cubo * (1,1,0.9)
                    aumentoActual -= 1
                cv.drawContours(frame, [ htrans(M, cubo ).astype(int) ], -1, (0,128,0), 3, cv.LINE_AA)

            if points[1][0] >= limites[1][0] and points[1][0] <= limites[0][0] and points[1][1] >= limites[2][1] and points[1][1] <= limites[0][1]:
                # Aumentar altura del cubo
                if aumentoActual < aumentoMax:
                    cubo = cubo * (1,1,1.1)
                    aumentoActual += 1
                cv.drawContours(frame, [ htrans(M, cubo ).astype(int) ], -1, (0,128,0), 3, cv.LINE_AA)


        # Mostramos el sistema de referencia inducido por el marcador (es una utilidad de umucv)
        # showAxes(frame, M, scale=0.5)

        # hacemos que se mueva el cubo
        
        cv.drawContours(frame, [ htrans(M, cubo).astype(int) ], -1, (0,128,0), 3, cv.LINE_AA)

    cv.imshow('source',frame)
    
