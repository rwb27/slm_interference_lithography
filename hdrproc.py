# -*- coding: utf-8 -*-
"""
Created on Tue Oct 09 13:22:46 2018

@author: kh302
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.ndimage
from rgb_to_hdr import calibrate_hdr_from_rgb, hdr_from_rgb, fit_channels
import nplab.utils.gui
import nplab


# Loading exposure images into a list
#address C:\local\dev\slm_interference_lithography\Hamatsudata\hdrimagedata\testspot1

img_fn = ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"]
img_list = [cv2.imread(fn) for fn in img_fn]
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)