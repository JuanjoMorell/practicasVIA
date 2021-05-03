#!/usr/bin/env python

import numpy as np
import cv2 as cv

from umucv.util import ROI, putText
from umucv.stream import autoStream

cv.namedWindow("input")
cv.moveWindow('input', 0, 0)

region = ROI("input")
modelo = []

def readrgb(frame):
    return cv.cvtColor( frame, cv.COLOR_BGR2RGB) 

def rgb2yuv(x):
    return cv.cvtColor(x,cv.COLOR_RGB2YUV)   


def hconcat_resize(img_list, interpolation = cv.INTER_CUBIC):
    # take minimum hights
    h_min = min(img.shape[0] 
                for img in img_list)
    # image resizing 
    im_list_resize = [cv.resize(img,
                       (int(img.shape[1] * h_min / img.shape[0]), h_min),
                        interpolation = interpolation) 
                      for img in img_list]
    # return final image
    return cv.hconcat(im_list_resize)

def uvh(x):
    def normhist(x): return x / np.sum(x)
    
    yuv = rgb2yuv(x)
    h = cv.calcHist([yuv]     # necesario ponerlo en una lista aunque solo admite un elemento
                    ,[1,2]    # elegimos los canales U y V
                    ,None     # posible máscara
                    ,[32,32]  # las cajitas en cada dimensión
                    ,[0,256]+[0,256] # rango de interés (todo)
                   )
    return normhist(h)

for key, frame in autoStream():
    img = readrgb(frame)
    copia = frame.copy()

    if region.roi:
        [x1,y1,x2,y2] = region.roi
        if key == ord('c'):
            trozo = frame[y1:y2+1, x1:x2+1]
            cv.imshow("trozo", trozo)
        if key == ord('x'):
            region.roi = []
        if key == ord('m') and len(modelo) < 3:
            # Guardar histograma en el modelo
            modelo.append(frame[y1:y2+1, x1:x2+1])
            concatenar = hconcat_resize(modelo)
            cv.imshow("Modelo", concatenar)
        if len(modelo) == 2:
            hist = [uvh(r) for r in modelo]
            uvr = np.floor_divide( cv.cvtColor(img,cv.COLOR_RGB2YUV)[:,:,[1,2]], 8)

            u = uvr[:,:,0]
            v = uvr[:,:,1]
            med = [ np.mean(r,(0,1)) for r in modelo ]  
            lik = [ h[u,v] for h in hist ]
            lik = [ cv.GaussianBlur(l, (0,0), 10) for l in lik ]

            E = np.sum(lik, axis=0)
            p = np.array(lik) / E
            c  = np.argmax(p,axis=0)
            mp = np.max(p, axis=0)
            mp[E < 0.1] = 0

            res = np.zeros(img.shape,np.uint8)
            for k in range(len(modelo)):
                res[c==k] = med[k]

            cv.imshow("Segmentacion", res);


        
        cv.rectangle(copia, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        
    h,w,_ = frame.shape
    cv.imshow('input',copia)

