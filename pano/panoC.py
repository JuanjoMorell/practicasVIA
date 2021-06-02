import numpy as np
import cv2   as cv

import matplotlib.pyplot as plt
import numpy.linalg      as la

from ipywidgets          import interactive
import argparse
import glob

# ----
def fig(w,h):
    plt.figure(figsize=(w,h))

def readrgb(file):
    return cv.cvtColor( cv.imread('./'+file), cv.COLOR_BGR2RGB) 

def rgb2gray(x):
    return cv.cvtColor(x,cv.COLOR_RGB2GRAY)

def imshowg(x):
    plt.imshow(x, "gray")
# ---

def desp(d):
    dx,dy = d
    return np.array([
            [1,0,dx],
            [0,1,dy],
            [0,0,1]])

def t(h,x):
    return cv.warpPerspective(x, desp((100,150)) @ h,(1000,600))

# ---

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="images")
args = vars(ap.parse_args())

# utilidad para devolver el número de correspondencias y la homografía entre dos imágenes

sift = cv.xfeatures2d.SIFT_create()
bf = cv.BFMatcher()

def match(query, model):
    x1 = query
    x2 = model
    (k1, d1) = sift.detectAndCompute(x1, None)
    (k2, d2) = sift.detectAndCompute(x2, None)
    
    matches = bf.knnMatch(d1,d2,k=2)
    # ratio test
    good = []
    for m in matches:
        if len(m) == 2:
            best, second = m
            if best.distance < 0.75*second.distance:
                good.append(best)
    
    #if len(good) < 6: return 6, None
    
    src_pts = np.array([ k2[m.trainIdx].pt for m in good ]).astype(np.float32).reshape(-1,2)
    dst_pts = np.array([ k1[m.queryIdx].pt for m in good ]).astype(np.float32).reshape(-1,2)
    
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3)
    
    return sum(mask.flatten()>0), H

pano = [readrgb(x) for x in sorted(glob.glob(args["images"] + '/*.jpg'))]

ordenParejas = sorted([(match(p,q)[0],i,j) for i,p in enumerate(pano) for j,q in enumerate(pano) if i< j],reverse=True)

# Obtener la relación con más matches y colocar la imagen en el centro
ordenFotos = [(ordenParejas[0][1], ordenParejas[0][2])]

izq = 0
der = 0
if ordenFotos[0][0] > ordenFotos[0][1]:
    izq = ordenFotos[0][1]
    der = ordenFotos[0][0]
else:
    izq = ordenFotos[0][0]
    der = ordenFotos[0][1]
next = (0,0)

h,w,_ = pano[ordenFotos[0][0]].shape
mw, mh = 100*len(pano),100
T = desp((100,float(100)))
sz = (w+2*mw,h+2*mh)
base = cv.warpPerspective(pano[ordenFotos[0][0]], T, sz)
baseImg = ordenFotos[0][0]

ordenCopia = ordenParejas[1:]
while len(ordenFotos) != (len(pano) - 1):
    for (valor, i1, i2) in ordenCopia:
        if i1 == izq or i2 == izq or i1 == der or i1 == izq: 
            next = (i1,i2)
            if izq > i1:
                izq = i1
            elif izq > i2:
                izq = i2
            elif der < i1:
                der = i1
            elif der < i2:
                der = i2
            ordenCopia.remove((valor,i1,i2))
            ordenFotos.append(next)
            break
    
print(ordenFotos)

# Tratar el resto de las imagenes obteniendo mejor matches

_,H1 = match(pano[ordenFotos[0][0]],pano[ordenFotos[0][1]])
cv.warpPerspective(pano[ordenFotos[0][1]],T@H1,sz, base, 0, cv.BORDER_TRANSPARENT)
fig(10,6)
plt.imshow(base)
plt.show()

orden = T@H1

anterior = (0,0)
for (x,y) in ordenFotos[1:]:
    if x == baseImg:
        _,H = match(pano[x], pano[y])
        orden = T@H
        cv.warpPerspective(pano[y],orden,sz, base, 0, cv.BORDER_TRANSPARENT)
        fig(10,6)
        plt.imshow(base)
        plt.show()
    elif y == baseImg:
        _,H = match(pano[y], pano[x])
        orden = T@H
        cv.warpPerspective(pano[x],orden,sz, base, 0, cv.BORDER_TRANSPARENT)
        fig(10,6)
        plt.imshow(base)
        plt.show()
    elif anterior[0] == y:
        _,H = match(pano[y], pano[x])
        orden = orden@H
        cv.warpPerspective(pano[x],orden,sz, base, 0, cv.BORDER_TRANSPARENT)
        fig(10,6)
        plt.imshow(base)
        plt.show()
    else:
        _,H = match(pano[x], pano[y])
        orden = H@orden
        cv.warpPerspective(pano[y],orden,sz, base, 0, cv.BORDER_TRANSPARENT)
        fig(10,6)
        plt.imshow(base)
        plt.show()
    anterior = (x,y)
