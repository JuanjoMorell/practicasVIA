import numpy as np
import cv2   as cv

import matplotlib.pyplot as plt
import numpy.linalg      as la

from ipywidgets          import interactive

from collections import deque
from umucv.util import putText
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="image")
ap.add_argument("-v", "--view", required=True,
	help="view")
ap.add_argument("-r", "--real", required=True,
	help="real")
ap.add_argument("-m", "--mm", type=int, required=True, help="mm")
ap.add_argument("-p", "--px", type=int, required=True, help="pixels")
args = vars(ap.parse_args())

def fig(w,h):
    plt.figure(figsize=(w,h))

def readrgb(file):
    return cv.cvtColor( cv.imread('./'+file), cv.COLOR_BGR2RGB) 

def rgb2gray(x):
    return cv.cvtColor(x,cv.COLOR_RGB2GRAY)

def imshowg(x):
    plt.imshow(x, "gray")

# para imprimir arrays con el número de decimales deseados
import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)

# imprime un array con pocos decimales
def sharr(a, prec=3):
    with printoptions(precision=prec, suppress=True):
        print(a)
    
# dibuja un polígono cuyos nodos son las filas de un array 2D
def shcont(c, color='blue', nodes=True):
    x = c[:,0]
    y = c[:,1]
    x = np.append(x,x[0])
    y = np.append(y,y[0])
    plt.plot(x,y,color)
    if nodes: plt.plot(x,y,'.',color=color, markerSize=11)

# crea un vector (array 1D)
# (no es imprescindible, numpy admite tuplas o listas en muchas funciones,
# pero a esta función es útil a veces para aplicar operaciones aritméticas)
def vec(*argn):
    return np.array(argn)

# convierte un conjunto de puntos ordinarios (almacenados como filas de la matriz de entrada)
# en coordenas homogéneas (añadimos una columna de 1)
def homog(x):
    ax = np.array(x)
    uc = np.ones(ax.shape[:-1]+(1,))
    return np.append(ax,uc,axis=-1)

# convierte en coordenadas tradicionales
def inhomog(x):
    ax = np.array(x)
    return ax[..., :-1] / ax[...,[-1]]

# aplica una transformación homogénea h a un conjunto
# de puntos ordinarios, almacenados como filas
def htrans(h,x):
    return inhomog(homog(x) @ h.T)

def desp(d):
    dx,dy = d
    return np.array([
            [1,0,dx],
            [0,1,dy],
            [0,0,1]])

def scale(s):
    sx,sy = s
    return np.array([
            [sx,0,0],
            [0,sy,0],
            [0,0,1]])

# rotación eje "vertical" del plano
def rot3(a):
    c = np.cos(a)
    s = np.sin(a)
    return np.array([
            [c,-s,0],
            [s, c,0],
            [0, 0,1]])

pi = np.pi
degree = pi/180

img = readrgb(args["image"])

view = np.loadtxt(args["view"], dtype=float)

plt.imshow(img)
shcont(view)
plt.show()

real = np.loadtxt(args["real"], dtype=float)
print(real)
H,_ = cv.findHomography(view, real)
rec = cv.warpPerspective(img,H,(800,1000))

points = deque(maxlen=2)
frame = rec.copy()

def fun(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))

cv.namedWindow("medir")
cv.setMouseCallback("medir", fun)

while True:
    for p in points:
        cv.circle(frame, p,3,(0,0,255),-1)
    if len(points) == 2:
        cv.line(frame, points[0],points[1],(0,0,255))
        c = np.mean(points, axis=0).astype(int)
        d = np.linalg.norm(np.array(points[1])-points[0])
        # pixels a cm
        cm = d * args["mm"] / (args["px"] *10)
        putText(frame,f'{cm:.1f} cm',c)

    cv.imshow('medir', frame)
    if cv.waitKey(10) == 27:
        break

cv.destroyAllWindows()
'''
'''