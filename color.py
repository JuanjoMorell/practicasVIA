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
modeloHist = []

def calcBRGHist(bgr_hist, hist_h):
    histSize = 256
    histRange = (0,256)
    accumulate = False
    hist_h = y2 - y1

    b_hist = cv.calcHist(bgr_hist, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv.calcHist(bgr_hist, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv.calcHist(bgr_hist, [2], None, [histSize], histRange, accumulate=accumulate)

    cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

    return b_hist, g_hist, r_hist

def dibujarHistograma(frame, bgr_hist, x1, x2, y1, y2):
    histSize = 256

    hist_w = x2 - x1
    hist_h = y2 - y1
    bin_w = int(round( hist_w/histSize ))

    b_hist, g_hist, r_hist = calcBRGHist(bgr_hist, hist_h)

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

def compararHistogramas(modelo, trozo, hist_h):
    # Histograma del trozo a comparar
    bgr_planes = cv.split(trozo)
    histSize = 256
    b_hist, g_hist, r_hist = calcBRGHist(bgr_planes, hist_h)

    best_corr = 0
    index = 0
    best_index = 0
    # Comparar con los histogramas del modelo
    for [b_hist2, g_hist2, r_hist2] in modeloHist:
        b_comp = cv.compareHist(b_hist, b_hist2, cv.HISTCMP_CORREL)
        g_comp = cv.compareHist(g_hist, g_hist2, cv.HISTCMP_CORREL)
        r_comp = cv.compareHist(r_hist, r_hist2, cv.HISTCMP_CORREL)
        #print(b_comp + g_comp + r_comp)
        if (b_comp + g_comp + r_comp) > best_corr:
            best_corr = b_comp + g_comp + r_comp
            best_index = index
        index = index + 1
    return best_index

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
            modeloHist.append(calcBRGHist(cv.split(frame[y1:y2+1, x1:x2+1]),y2 - y1))
        if key == ord('n') and modeloHist != []:
            # Evaluar el roi con el modelo de histogramas
            mejor = compararHistogramas(modeloHist, frame[y1:y2+1, x1:x2+1], y2 - y1)
            cv.imshow("detected", modelo[mejor])
        
        bgr_planes = cv.split(frame[y1:y2+1, x1:x2+1])
        dibujarHistograma(frame, bgr_planes, x1, x2, y1, y2)

        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,255), thickness=2)
        putText(frame, f'{x2-x1+1}x{y2-y1+1}', orig=(x1,y1-8))

    h,w,_ = frame.shape
    putText(frame, f'{w}x{h}')
    cv.imshow('input',frame)
