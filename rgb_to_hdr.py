# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:54:33 2016

@author: hera
"""
import matplotlib.pyplot as plt
import numpy as np
"""
# Calibrating RGB -> HDR

# simple plot of colour channel correlation
img = cam.color_image()
f, ax = plt.subplots(1,3)
for i, a in enumerate(ax):
    a.plot(img[:,:,i].flatten(), img[:,:,(i+1)%3].flatten(),'.')

# cut off silly values
img = cam.color_image()
f, axes = plt.subplots(1,3)
fits = []
for i, ax in enumerate(axes):
    a = img[:,:,i].flatten()
    b = img[:,:,(i+1)%3].flatten()
    r = np.logical_and(np.logical_and(a>20, a<200), np.logical_and(b>20, b<200))

    if np.sum(r) > 100: 
        ax.plot(a[r], b[r],'.')
        p = np.polyfit(a[r], b[r], 1)
        fits.append(p)
        xa, xb = ax.get_xlim()
        ax.plot([xa, xb], [p[1]+p[0]*xa, p[1]+p[0]*xb])
    else:
        fits.append(None)
"""
# we assume blue is the brightest channel, then red, then green.
#blue = channel * slopes[c] + offsets[c]
#b = r*p0 + p1
#r = g*q0 + q1
#so b = (g*q0 + q1)*p0 + p1
#img = cam.color_image()


"""  
hdr = hdr_from_rgb(offsets, slopes, img)
f, ax = plt.subplots(1,1)
for y in [200,220,240,260,280,300]:
    sec = img[y,:,:]
    for i, col in enumerate(['red','green','blue']):
        ignore = ax.plot(sec[:,i]*slopes[i] + offsets[i], color=col)
        ignore = ax.plot(hdr[y,:],color='black')
"""