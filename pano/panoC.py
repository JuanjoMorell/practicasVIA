import numpy as np
import cv2   as cv

import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
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

# Obtener imagen base (imagen con mas matches)
matches = [0] * len(pano)
for (m, im1, im2) in ordenParejas:
    matches[im1] += m
    matches[im2] += m

mostMatch = 0
imgBase = 0
contador = 0
for m in matches:
    if m > mostMatch:
        mostMatch = m
        imgBase = contador
    contador += 1

print("Imagen base: " + str(imgBase) + ", con un total de " + str(mostMatch) + " matches")

# Buscar pareja con base que mas matches tenga
pareja = (0,0,0)
for (m, im1, im2) in ordenParejas:
    if im1 == imgBase or im2 == imgBase:
        pareja = (m, im1, im2)
        break

# Obtener la relación con más matches y colocar la imagen en el centro
ordenFotos = [(pareja[1], pareja[2])]

izq = 0
der = 0
if ordenFotos[0][0] > ordenFotos[0][1]:
    izq = ordenFotos[0][1]
    der = ordenFotos[0][0]
else:
    izq = ordenFotos[0][0]
    der = ordenFotos[0][1]
next = (0,0)

ordenParejas.remove(pareja)
while len(ordenFotos) != (len(pano) - 1):
    for (valor, i1, i2) in ordenParejas:
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
            ordenParejas.remove((valor,i1,i2))
            ordenFotos.append(next)
            break
    
print(ordenFotos)

h,w,_ = pano[ordenFotos[0][0]].shape
mw, mh = 500,500
T = desp((0,float(0)))
sz = (w+2*mw,h+2*mh)
base = cv.warpPerspective(pano[ordenFotos[0][0]], T, sz)
baseImg = ordenFotos[0][0]

# Tratar el resto de las imagenes obteniendo mejor matches
_,H1 = match(pano[ordenFotos[0][0]],pano[ordenFotos[0][1]])

orden = H1

despX = 0
despY = 0
tamX = 0
tamY = 0
anterior = (0,0)
for (x,y) in ordenFotos[1:]:
    if x == baseImg:
        _,H = match(pano[x], pano[y])
        orden = T@H
        if despX > orden.astype(int)[0][2]:
            despX = orden.astype(int)[0][2]
        if despY > orden.astype(int)[1][2]:
            despY = orden.astype(int)[1][2]
        tamX += abs(orden.astype(int)[0][2])
        tamY += abs(orden.astype(int)[1][2])
    elif y == baseImg:
        _,H = match(pano[y], pano[x])
        orden = T@H
        if despX > orden.astype(int)[0][2]:
            despX = orden.astype(int)[0][2]
        if despY > orden.astype(int)[1][2]:
            despY = orden.astype(int)[1][2]
        tamX += abs(orden.astype(int)[0][2])
        tamY += abs(orden.astype(int)[1][2])
    elif anterior[0] == y:
        _,H = match(pano[y], pano[x])
        orden = orden@H
        if despX > orden.astype(int)[0][2]:
            despX = orden.astype(int)[0][2]
        if despY > orden.astype(int)[1][2]:
            despY = orden.astype(int)[1][2]
        tamX += abs(orden.astype(int)[0][2])
        tamY += abs(orden.astype(int)[1][2])
    else:
        _,H = match(pano[x], pano[y])
        orden = H@orden
        if despX > orden.astype(int)[0][2]:
            despX = orden.astype(int)[0][2]
        if despY > orden.astype(int)[1][2]:
            despY = orden.astype(int)[1][2]
        tamX += abs(orden.astype(int)[0][2])
        tamY += abs(orden.astype(int)[1][2])
    anterior = (x,y)

tamX = abs(tamX - despX)
tamY = abs(tamY - despY)
print("Desplazamiento de la imagen base: " + str(abs(despX)) + ", " + str(abs(despY)))
print("Tamaño total imagen: " + str(abs(tamX)) + ", " + str(abs(tamY)))

h,w,_ = pano[ordenFotos[0][0]].shape
T = desp((abs(despX),float(abs(despY))))
sz = (tamX+pano[imgBase].shape[0], tamY+pano[imgBase].shape[1])
base = cv.warpPerspective(pano[ordenFotos[0][0]], T, sz)
baseImg = ordenFotos[0][0]


cv.warpPerspective(pano[ordenFotos[0][1]],T@H1,sz, base, 0, cv.BORDER_TRANSPARENT)

orden = T@H1

for (x,y) in ordenFotos[1:]:
    if x == baseImg:
        _,H = match(pano[x], pano[y])
        orden = T@H
        
        cv.warpPerspective(pano[y],orden,sz, base, 0, cv.BORDER_TRANSPARENT)
    elif y == baseImg:
        _,H = match(pano[y], pano[x])
        orden = T@H
        cv.warpPerspective(pano[x],orden,sz, base, 0, cv.BORDER_TRANSPARENT)
    elif anterior[0] == y:
        _,H = match(pano[y], pano[x])
        orden = orden@H
        cv.warpPerspective(pano[x],orden,sz, base, 0, cv.BORDER_TRANSPARENT)
    else:
        _,H = match(pano[x], pano[y])
        orden = H@orden
        cv.warpPerspective(pano[y],orden,sz, base, 0, cv.BORDER_TRANSPARENT)
    anterior = (x,y)

fig(10,6)
plt.imshow(base)
plt.show()