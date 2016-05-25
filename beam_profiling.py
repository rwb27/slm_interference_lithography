# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:26:16 2016

@author: hera
"""

from nplab.instrument import Instrument
import time

def beam_profile_on_SLM(spot, N, overlap=0.0):
    """Scan a spot over the SLM, recording intensities as we go"""
    intensities = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            make_spots([spot + [float(i-N/2.0)/N, float(j-N/2.0)/N, 
                                (0.5 + overlap/2.0)/N, 0]])
            time.sleep(0.1);
            cam.color_image()
            intensities[i,j] = cam.color_image()[:,:,2].astype(np.float).sum()
    return intensities

def merit_function():
    time.sleep(0.1)
    cam.gray_image()
    return np.max(cam.gray_image())
    
def optimise_aberration_correction(zernike_coefficients, merit_function, tweak_amounts=[1,0.5,0.1]):
    """Tweak the Zernike coefficients incrementally to optimise them."""

    def test_coefficients(coefficients):
        set_uniform(3, coefficients)
        return merit_function()
    
    coefficients = np.array(zernike_coefficients)
    for dz in tweak_amounts:
        for i in range(len(coefficients)):
            values = np.array([-1,0,1]) * dz + coefficients[i]
            merits = np.zeros(3)
            for j, v in enumerate(values):
                coefficients[i] = v
                merits[j] = test_coefficients(coefficients)
            while True:
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
    return coefficients

                    