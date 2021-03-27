#!/usr/bin/env python

# ejemplo de selección de ROI

import numpy as np
import cv2 as cv

from umucv.util import ROI, putText
from umucv.stream import autoStream

cv.namedWindow("input")
cv.moveWindow('input', 0, 0)

region = ROI("input")
modelo = []

def compararHistogramas():
    return

def hconcat_resize(img_list, 
                   interpolation 
                   = cv.INTER_CUBIC):
      # take minimum hights
    h_min = min(img.shape[0] 
                for img in img_list)
      
    # image resizing 
    im_list_resize = [cv.resize(img,
                       (int(img.shape[1] * h_min / img.shape[0]),
                        h_min), interpolation
                                 = interpolation) 
                      for img in img_list]
      
    # return final image
    return cv.hconcat(im_list_resize)

for key, frame in autoStream():
    
    if region.roi:
        [x1,y1,x2,y2] = region.roi
        if key == ord('c'):
            trozo = frame[y1:y2+1, x1:x2+1]
            cv.imshow("trozo", trozo)
        if key == ord('x'):
            region.roi = []
        if key == ord('m'):
            # Guardar histograma en el modelo
            modelo.append(frame[y1:y2+1, x1:x2+1])
            concatenar = hconcat_resize(modelo)
            cv.imshow("Modelo", concatenar)

        bgr_planes = cv.split(frame[y1:y2+1, x1:x2+1])
        histSize = 256
        histRange = (0,256)
        accumulate = False

        b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
        g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
        r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
            
        hist_w = x2 - x1
        hist_h = y2 - y1
        bin_w = int(round( hist_w/histSize ))

        cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
        cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
        cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

        for i in range(1, histSize):
            cv.line(frame, ( (bin_w*(i-1))+x1, (hist_h - int(b_hist[i-1]))+y1 ),
                    ( (bin_w*(i))+x1, (hist_h - int(b_hist[i]))+y1 ),
                    ( 255, 0, 0), thickness=2)
            cv.line(frame, (( bin_w*(i-1))+x1, (hist_h - int(g_hist[i-1]))+y1 ),
                    ( (bin_w*(i))+x1, (hist_h - int(g_hist[i]))+y1 ),
                    ( 0, 255, 0), thickness=2)
            cv.line(frame, ((bin_w*(i-1))+x1, (hist_h - int(r_hist[i-1]))+y1 ),
                    ( (bin_w*(i))+x1, (hist_h - int(r_hist[i]))+y1 ),
                    ( 0, 0, 255), thickness=2)

        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))

    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)
    

