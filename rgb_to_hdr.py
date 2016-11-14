# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:54:33 2016

@author: rwb27

Utilities to calibrate a camera to generate HDR images from RGB ones, under
the assumption that we're dealing with monochromatic laser light.
"""
import matplotlib.pyplot as plt
import numpy as np

# we assume blue is the brightest channel, then red, then green.
#blue = channel * slopes[c] + offsets[c]
#b = r*p0 + p1
#r = g*q0 + q1
#so b = (g*q0 + q1)*p0 + p1
#img = cam.color_image()

def calibrate_hdr_from_rgb(img):
        """Calibrate the colour channels in an image for HDR
        
        We assume there's one true colour (blue) and that it leaks into
        the other colour channels (red, then green) as it gets more intense.
        
        Here, we correlate channels against each other, so we can retrieve
        a nice HDR image from an RGB image.
        
        It returns a length-3 vector of offsets, and slopes such that
        blue = channel * slope[c] + offset[c]
        """
        def fit_channels(img,i,j):
            a = img[:,:,i].flatten()
            b = img[:,:,j].flatten()
            r = np.logical_and(np.logical_and(a>20, a<200), np.logical_and(b>20, b<200))
            return np.polyfit(a[r], b[r], 1)
        
        offsets = [0,0,0]
        slopes = [0,0,1]
        p = fit_channels(img, 0,2) #calibrate red to blue
        offsets[0] = p[1]
        slopes[0] = p[0]
        q = fit_channels(img, 1,0) #calibrate green to red
        offsets[1] = p[1] + q[1]*p[0]
        slopes[1] = p[0]*q[0]
        offsets = np.array(offsets)
        slopes = np.array(slopes)
        return offsets, slopes

def hdr_from_rgb(offsets, slopes, rgb):
        """Convert RGB to an HDR image
        
        We assume, as above, that the laser is blue, and leaks into red then
        green.
        """
        rgbt = rgb.copy()
        rgbt[np.logical_and(rgb>200, np.array([1,0,1])[np.newaxis, np.newaxis,:])] = 0
        rgbt[np.logical_and(rgb<10, np.array([1,1,0])[np.newaxis, np.newaxis,:])] = 0
        brgt = rgbt[:,:,(2,0,1)] # Order them so they prefer B
        pick = np.argmax(brgt, axis=2) # Pick the brightest unsaturated channel
        pick = np.array([2,0,1])[pick] # fix indices so they are rgb
        hdr = rgbt.max(axis=2).astype(np.float) * slopes[pick] + offsets[pick]
        return hdr








"""
# Calibrating RGB -> HDR
img = cam.color_image()
assert np.any(img.min(axis=2)==255), "You should have some saturated pixels for best calibration..."
o,s = calibrate_hdr_from_rgb(img)
hdr = hdr_from_rgb(o,s,cam.color_image())
for y in (np.arange(5)-2)*20 + 240:
    plt.plot(hdr[y,:])

"""
