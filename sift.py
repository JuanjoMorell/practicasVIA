import cv2 as cv
import time
import os

from umucv.stream import autoStream
from umucv.util import putText

def getKeypoints(frame):
    keypoints , descriptors = sift.detectAndCompute(frame, mask=None)
    return keypoints, descriptors

def getMatches(descriptores, d0):
    matches = matcher.knnMatch(descriptors, d0, k=2)

    good = []
    for m in matches:
        if len(m) >= 2:
            best,second = m
            if best.distance < 0.75*second.distance:
                good.append(best) 
    return len(good)

def bestMatch(descriptores, modelo):
    mayorM = 0
    a = 0
    index = 0
    for [d, img] in modelo:
        nMatch = getMatches(descriptores, d)
        #print("Index: ", index, ", Matches: ", nMatch)
        if nMatch > mayorM:
            mayorM = nMatch
            a = index
        index = index + 1
    return a

sift = cv.xfeatures2d.SIFT_create(nfeatures=500)
matcher = cv.BFMatcher()

x0 = None
modelo = []
dir_modelo = "./pelis/"

for key, frame in autoStream():

    if key == ord('m'):
        # Se carga el modelo con las imagenes de Pelis
        pelis = os.listdir(dir_modelo)
        for peli in pelis:
            d = os.path.join(dir_modelo, peli)
            img = cv.imread(d)
            key, des = getKeypoints(img)
            modelo.append([des,img])
        print("* MODELO CARGADO")
    
    if key == ord('c'):
        # Almacenar nueva imagen en el modelo
        k0, d0 = getKeypoints(frame)
        modelo.append([d0,frame])
        print("* IMAGEN AÑADIDA")

    if key == ord('d'):
        # Detectar que pelicula es
        t0 = time.time()
        keypoints , descriptors = getKeypoints(frame)
        t1 = time.time()
        putText(frame, f'{len(keypoints)} pts  {1000*(t1-t0):.0f} ms')

        if modelo is []:
            continue
        else:
            t2 = time.time()
            imgIndex = bestMatch(descriptors, modelo)
            t3 = time.time()

            # mostrar imagen del objeto con mas matches
            [d, img] = modelo[imgIndex]
            cv.imshow("Match", img)

    cv.imshow("Frame", frame)
