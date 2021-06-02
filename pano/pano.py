import numpy as np
import cv2   as cv

import matplotlib.pyplot as plt
import numpy.linalg      as la

from ipywidgets          import interactive
import argparse

def fig(w,h):
    plt.figure(figsize=(w,h))

def readrgb(file):
    return cv.cvtColor( cv.imread('./'+file), cv.COLOR_BGR2RGB) 

def rgb2gray(x):
    return cv.cvtColor(x,cv.COLOR_RGB2GRAY)

def imshowg(x):
    plt.imshow(x, "gray")

#----------------------------------------    

ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--image1", required=True,
	help="image1")
ap.add_argument("-i2", "--image2", required=True,
	help="image2")
args = vars(ap.parse_args())

rgb1 = readrgb(args["image1"])
rgb2 = readrgb(args["image2"])

x1 = rgb2gray(rgb1)
x2 = rgb2gray(rgb2)

sift = cv.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(x1, None)
(kps2, descs2) = sift.detectAndCompute(x2, None)
print("# (img1) kps: {}, descriptors: {}".format(len(kps), descs.shape))
print("# (img2) kps: {}, descriptors: {}".format(len(kps2), descs2.shape))

bf = cv.BFMatcher()

matches = bf.knnMatch(descs2,descs,k=2)

print('Matches: {}'.format(len(matches)))

good = []
for mt in matches:
    if len(mt) == 2:
        best, second = mt
        if best.distance < 0.75*second.distance:
            good.append(best) 

print('Buenos matches: {}'.format(len(good)))
img3 = cv.drawMatches(x2,kps2,
                      x1,kps,
                      good,
                      flags=2,outImg=None,
                      matchColor=(128,0,0))
fig(12,4)
plt.imshow(img3)
plt.show()

src_pts = np.array([ kps [m.trainIdx].pt for m in good ]).astype(np.float32).reshape(-1,2)
dst_pts = np.array([ kps2[m.queryIdx].pt for m in good ]).astype(np.float32).reshape(-1,2)
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 3)

def desp(d):
    dx,dy = d
    return np.array([
            [1,0,dx],
            [0,1,dy],
            [0,0,1]])

def t(h,x):
    return cv.warpPerspective(x, desp((100,150)) @ h,(1000,600))

fig(15,8)
imshowg( t(np.eye(3),x2) )
plt.show()

fig(15,8)
imshowg( t(H,x1) )
plt.show()

fig(15,8)
imshowg( np.maximum( t(np.eye(3),x2), t(H,x1) ) );
plt.show()

fig(15,8)
image = np.maximum( t(np.eye(3),rgb2) , t(H,rgb1) )
plt.imshow( image );
plt.show()



