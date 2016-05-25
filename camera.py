# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:42:24 2016

@author: hera
"""

import nplab
from nplab.instrument.camera.opencv import OpenCVCamera

if __name__ == '__main__':
    cam = OpenCVCamera()
    cam.live_view = True
    ui = cam.show_gui(blocking=False)
    