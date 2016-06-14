# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:26:16 2016

@author: hera
"""

from nplab.instrument import Instrument
from nplab.instrument.light_sources.ondax_laser import OndaxLaser
from nplab.instrument.shutter.southampton_custom import ILShutter
from nplab.instrument.camera.opencv import OpenCVCamera
from slm_interference_lithography import VeryCleverBeamsplitter
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.ndimage

def beam_profile_on_SLM(slm, cam, spot, N, overlap=0.0):
    """Scan a spot over the SLM, recording intensities as we go"""
    intensities = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            slm.make_spots([spot + [float(i-N/2.0)/N, float(j-N/2.0)/N, 
                                (0.5 + overlap/2.0)/N, 0]])
            time.sleep(0.1);
            cam.color_image()
            time.sleep(0.1);
            intensities[i,j] = cam.color_image()[:,:,2].astype(np.float).sum()
    return intensities

    
def optimise_aberration_correction(slm, cam, zernike_coefficients, merit_function, dz=1):
    """Tweak the Zernike coefficients incrementally to optimise them."""

    def test_coefficients(coefficients):
        slm.zernike_coefficients = coefficients
        return merit_function()
    
    coefficients = np.array(zernike_coefficients)
    for i in range(len(coefficients)):
        values = np.array([-1,0,1]) * dz + coefficients[i]
        merits = np.zeros(3)
        for j, v in enumerate(values):
            coefficients[i] = v
            merits[j] = test_coefficients(coefficients)
        for dummy in range(10):
            if merits.argmax() == 0:
                values = np.concatenate([[values[0] - dz], values])
                coefficients[i] = values[0]
                m = test_coefficients(coefficients)
                merits = np.concatenate([[m], merits])
            elif merits.argmax() == len(merits) - 1:
                values = np.concatenate([values, [values[0] + dz]])
                coefficients[i] = values[-1]
                m = test_coefficients(coefficients)
                merits = np.concatenate([merits, [m]])
            else:
                break
        coefficients[i] = values[merits.argmax()]
        test_coefficients(coefficients)
    return coefficients

if __name__ == '__main__':
    slm = VeryCleverBeamsplitter()
    shutter = ILShutter("COM3")
    laser = OndaxLaser("COM1")
    cam = OpenCVCamera()
    slm.move_hologram(-1024,0,1024,768)
    # set uniform values so it has a blazing function and no aberration correction
    blazing_function = np.array([  0,   0,   0,   0,   0,   0,   0,   0,   0,  12,  69,  92, 124,
       139, 155, 171, 177, 194, 203, 212, 225, 234, 247, 255, 255, 255,
       255, 255, 255, 255, 255, 255]).astype(np.float)/255.0 #np.linspace(0,1,32)
    
    slm.blazing_function = blazing_function
    slm.update_gaussian_to_tophat(1900,3000, distance=1975e3)
    slm.make_spots([[20,10,0,1],[-20,10,0,1]])
    shutter.open_shutter()
    
"""
    # Optimise SLM for aberrations
    zernike_coefficients = np.zeros(12)
    slm.zernike_coefficients = zernike_coefficients
    slm.update_gaussian_to_tophat(2000,1, distance=2050e3)
    slm.make_spots([[-20,10,0,1]])
    cam.exposure=-2
    def merit_function():
        time.sleep(0.1)
        cam.color_image()
        return np.max(scipy.ndimage.uniform_filter(cam.color_image()[:,:,2], 17))
    zernike_coefficients = optimise_aberration_correction(slm, 
                                    cam, zernike_coefficients, merit_function, 
                                    dz=0.2)
    zernike_coefficients = optimise_aberration_correction(slm, 
                                    cam, zernike_coefficients, merit_function, 
                                    dz=0.05)
"""

"""
    # Find the proper centre of the SLM
    slm.update_gaussian_to_tophat(2000,1, distance=2050e3)
    cam.exposure=-2
    spot = [-20,10,0,1]
    N = 10
    slm.make_spots([spot + [0,0,0.5 * 1.5/N,0]])
    intensity = beam_profile_on_SLM(slm, cam, spot, N, overlap=0.5)
    centroid = np.array(scipy.ndimage.measurements.center_of_mass(intensity))
    actual_centre = (centroid+0.5)/float(N)
    #plt.imshow(intensity)
"""

"""
    # Really thorough optimisation of defocus
    slm.update_gaussian_to_tophat(2000,1500, distance=2050e3)
    cam.exposure = -2
    N = 30
    zernike_coefficients = np.zeros(12)
    img = cam.color_image()[:,:,2]
    focus_stack = np.zeros((N,)+img.shape, dtype=img.dtype)
    for i, d in enumerate(np.linspace(-3,3,N)):
        z = zernike_coefficients.copy()
        z[1] = d
        slm.zernike_coefficients = z
        time.sleep(0.1)
        hide = cam.color_image()
        time.sleep(0.1)
        focus_stack[i,:,:] = cam.color_image()[:,:,2]
    plt.figure()
    plt.imshow(focus_stack[:,240,:],aspect="auto")
"""