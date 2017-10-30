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
def fit_channels(img,i,j, plot=False, amin=5, amax=240, bmin=5, bmax=240):
    """Compare channels of an image, looking for linear relationships.
    
    img is the (RGB) image
    i, j are the A and B channels    
    amax and amin are the max/min values of the first channel allowed
    bmax and bmin are the max/min values of the second channel allowed
    
    The return value is a 2-element list of the slope (0) and intercept (1)
    """
    a = img[:,:,i].flatten()
    b = img[:,:,j].flatten()
    r = np.logical_and(np.logical_and(a>amin, a<amax), np.logical_and(b>bmin, b<bmax))
    coefficients = np.polyfit(a[r], b[r], 1) # m[0] is slope and m[1] is intercept
    print "fitting channel {} to {}: got {}".format(j, i, coefficients)
    if plot:
        plt.figure()
        plt.plot(a, b, '.')
        x = np.arange(256)
        plt.plot(x, x*coefficients[0] + coefficients[1], 'r-')
    return coefficients
    
    
def calibrate_hdr_from_rgb(img):
        """Calibrate the colour channels in an image for HDR
        
        We assume there's one true colour (blue) and that it leaks into
        the other colour channels (red, then green) as it gets more intense.
        
        Here, we correlate channels against each other, so we can retrieve
        a nice HDR image from an RGB image.
        
        It returns a length-3 vector of offsets, and slopes such that
        blue = channel * slope[c] + offset[c]
        """

        
        offsets = [0,0,0]
        slopes = [0,0,1]
        p = fit_channels(img, 0,2, plot=True) #calibrate red to blue
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
        
        We assume, as above, that the laser is blue, and leaks into red. We
        ignore the green channel.
        """
        hdr = rgb[:,:,2].astype(np.float) #start with blue
        saturated_pixels = hdr>200
        hdr[saturated_pixels] = rgb[:,:,0][saturated_pixels].astype(np.float) * slopes[0] + offsets[0] #fill in with red
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
