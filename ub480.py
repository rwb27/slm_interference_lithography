# -*- coding: utf-8 -*-
"""
Created on Wed Jul 04 12:16:27 2018

@author: kh302
"""

import uc480
import pylab as pl

# create instance and connect to library
cam = uc480.uc480()

# connect to first available camera
cam.connect()

# take a single image
img = cam.acquire()

# clean up
cam.disconnect()

pl.imshow(img)
pl.show()