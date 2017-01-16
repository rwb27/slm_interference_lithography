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
from nplab.utils.gui import show_guis
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.ndimage
from rgb_to_hdr import calibrate_hdr_from_rgb, hdr_from_rgb
import nplab.utils.gui

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
    
def centroid(img, threshold = 0.5):
    """Return the centroid and peak intensity of the current camera image.
    
    NB the image is assumed to be monochrome and floating-point.
    """
    thresholded = img.astype(np.float) - img.min() - (img.max() - img.min())*threshold
    thresholded[thresholded < 0] = 0
    return scipy.ndimage.measurements.center_of_mass(thresholded)
    
def snapshot_fn(cam, o, s):
    """Create a callable that will return an HDR image"""
    def snap():
        cam.color_image()
        time.sleep(0.1)
        cam.color_image()
        time.sleep(0.1)
        return hdr_from_rgb(o, s, cam.color_image())
    return snap
    
def sequential_shack_hartmann(slm, snapshot_fn, spot, N, overlap=0.0, other_spots=[], pause=False):
    """Scan a spot over the SLM, recording intensities as we go"""
    results = np.zeros((N, N, 3)) # For each position, find X,Y,I
    if pause:
        app = nplab.utils.gui.get_qt_app()
    for i in range(N):
        for j in range(N):
            slm.make_spots([spot + [float(i+0.5-N/2.0)/N, float(j+0.5-N/2.0)/N, 
                                (0.5 + overlap/2.0)/N, 0]] + other_spots)
            hdr = snapshot_fn()
            results[i,j,2] = hdr.sum()
            results[i,j,:2] = centroid(hdr)
            if pause:
                raw_input("spot %d, %d (hit enter for next)" % (i, j))
                app.processEvents()
            print '.',
    return results
    
def plot_shack_hartmann(results):
    """Plot the results of the above wavefront sensor"""
    N = results.shape[0]
    centre = np.mean(results[:,:,:2], axis=(0,1))
    r = np.max(np.abs(results[:,:,:2]-centre))
    f, ax = plt.subplots(1,2)
    for i in range(N):
        for j in range(N):
            u = (i+0.5)/N - 0.5
            v = (j+0.5)/N - 0.5
            shift = (results[i,j,:2] - centre)/r/N
            ax[0].plot([u,u+shift[0]],[v,v+shift[1]],'o-')
    ax[1].imshow(results[:,:,2], cmap='cubehelix')
    
def plot_sh_shifts(ax, results, discard_edges=0):
    """Plot the results of the above wavefront sensor"""
    e = discard_edges
    N = results.shape[0]
    centre = np.mean(results[:,:,:2], axis=(0,1))
    r = np.max(np.abs(results[:,:,:2]-centre))
    for i in range(N):
        for j in range(N):
            if i >= e and j >= e and i<N-e and j<N-e:
                u = (i+0.5)/N - 0.5
                v = (j+0.5)/N - 0.5
                shift = (results[i,j,:2] - centre)/r/N
                ax.plot([u,u+shift[0]],[v,v+shift[1]],'o-')
    
def measure_modes(slm, snapshot_fn, spot, N, dz=1, **kwargs):
    """Repeat sequential shack hartmann for each Zernike mode"""
    out = []
    for i in range(13):
        z = np.zeros(12)
        if i < len(z):
            z[i] = dz
        slm.zernike_coefficients = z
        time.sleep(0.2)
        print ("Mode %d" % i),
        out.append(sequential_shack_hartmann(slm, snapshot_fn, spot, N, **kwargs))
        print
    return out
    
def optimise_aberration_correction(slm, cam, zernike_coefficients, merit_function, dz=1, modes=None):
    """Tweak the Zernike coefficients incrementally to optimise them."""

    def test_coefficients(coefficients):
        slm.zernike_coefficients = coefficients
        return merit_function()
    
    coefficients = np.array(zernike_coefficients)
    for i in (range(len(coefficients)) if modes is None else modes):
        values = np.array([-1,0,1]) * dz + coefficients[i]
        merits = np.zeros(values.shape[0])
        for j, v in enumerate(values):
            coefficients[i] = v
            merits[j] = test_coefficients(coefficients)
        if merits.argmax() == 0:
            pass # might want to go round for another try?
        elif merits.argmax() == len(merits) - 1:
            pass
        coefficients[i] = values[merits.argmax()]
        test_coefficients(coefficients)
    return coefficients

if __name__ == '__main__':
    slm = VeryCleverBeamsplitter()
    shutter = ILShutter("COM1")
    #laser = OndaxLaser("COM4")
    cam = OpenCVCamera()
    slm.move_hologram(-1024,0,1024,768)
    # set uniform values so it has a blazing function and no aberration correction
    blazing_function = np.array([  0,   0,   0,   0,   0,   0,   0,   0,   0,  12,  69,  92, 124,
       139, 155, 171, 177, 194, 203, 212, 225, 234, 247, 255, 255, 255,
       255, 255, 255, 255, 255, 255]).astype(np.float)/255.0 #np.linspace(0,1,32)
    def dim_slm(dimming):
        slm.blazing_function = (blazing_function - 0.5)*dimming + 0.5
    slm.blazing_function = blazing_function
    known_good_zernike = np.array([ 0.5,  4.4, -0.5, -0.3,  0.3,  0.4, -0.3,  0.1,  0.1, -0.2,  0.2, -0.1])
    slm.zernike_coefficients = known_good_zernike
    slm.update_gaussian_to_tophat(1900,3000, distance=3500e3)
    slm.make_spots([[20,10,0,1],[-20,10,0,1]])
    shutter.open_shutter()
    guis = show_guis([shutter, cam], block=False)

"""
# Sequential Shack-Hartmann sensor
slm.make_spots([[20,10,2050,1,0,0,0.15,0],[0,0,0,3,0,0,1,0]])
zernike_coefficients = np.zeros(12)
slm.zernike_coefficients = zernike_coefficients

# Calibrate HDR processing
cam.exposure=-2

## TURN LIGHTS OFF!
o,s = calibrate_hdr_from_rgb(cam.color_image())
hdr = hdr_from_rgb(o,s,cam.color_image())
plt.plot(hdr[240,:])
snap = snapshot_fn(cam, o, s)

res = sequential_shack_hartmann(slm, snap, [20,10,2050,1], 7, overlap=0.5, other_spots=[[0,0,0,2,0,0,1,0]], pause=True)
res = sequential_shack_hartmann(slm, snap, [25,-10,0,1], 5, overlap=0)
res = sequential_shack_hartmann(slm, snap, [25,-10,0,1], 10, overlap=0)
plot_shack_hartmann(res)

# A brutal attempt at modal decomposition
modes = measure_modes(slm, snap, [20,10,2050,1], 5, overlap=0)
flat = modes[12]
f, axes = plt.subplots(3,4)
axes_flat = [axes[i,j] for i in range(3) for j in range(4)]
for m, ax in zip(modes[:12], axes_flat):
    plot_sh_shifts(ax, m - flat, discard_edges=1)
"""
    
"""
    # Optimise SLM for aberrations with modal wavefront sensor
    zernike_coefficients = np.zeros(12)
    slm.zernike_coefficients = zernike_coefficients
    slm.update_gaussian_to_tophat(2000,1, distance=2050e3)
    slm.make_spots([[-20,10,0,1]])
    slm.make_spots([[20,10.2,0,0.3],[0,0,0,1]])
    # or disable gaussian to tophat and
    slm.make_spots([[20,10,2050,1]])
    dim_slm(0.2)
    cam.exposure=0
    def brightest_hdr():
        time.sleep(0.1)
        cam.color_image()
        time.sleep(0.1)
        hdr = hdr_from_rgb(o,s,cam.color_image())
        for i in range(4):
            hdr += hdr_from_rgb(o,s,cam.color_image())
        hdr /= 5
        return np.max(scipy.ndimage.uniform_filter(hdr, 17))
    def brightest_g():
        time.sleep(0.1)
        cam.color_image()
        img = cam.color_image()[...,0]
        avg = np.zeros(shape = img.shape, dtype=np.float)
        for i in range(4):
            avg += cam.color_image()[...,1]
        avg /= 4
        return np.max(scipy.ndimage.uniform_filter(avg, 17))
        
    ## TURN LIGHTS OFF!
    o,s = calibrate_hdr_from_rgb(cam.color_image())
    hdr = hdr_from_rgb(o,s,cam.color_image())
    plt.plot(hdr[240,:])
    def beam_sd():
        cam.color_image()
        time.sleep(0.1)
        hdr = hdr_from_rgb(o,s,cam.color_image())
        x = np.mean(hdr * np.arange(hdr.shape[0])[:,np.newaxis])/np.mean(hdr)
        y = np.mean(hdr * np.arange(hdr.shape[1])[np.newaxis,:])/np.mean(hdr)
        dx2 = np.mean(hdr * ((np.arange(hdr.shape[0])-x)**2)[:,np.newaxis])/np.mean(hdr)
        dy2 = np.mean(hdr * ((np.arange(hdr.shape[1])-y)**2)[np.newaxis,:])/np.mean(hdr)
        sd = np.sqrt(dx2+dy2)
        return 1/sd
    def average_fn(f, n):
        return lambda: np.mean([f() for i in range(n)])
    merit_function = lambda: np.mean([beam_sd() for i in range(3)])
    
    zernike_coefficients = optimise_aberration_correction(slm, cam, zernike_coefficients, brightest_g, dz=0.5, modes=[1])
    zernike_coefficients = optimise_aberration_correction(slm, cam, zernike_coefficients, brightest_g, dz=0.3, modes=[0,1,2])
    for dz in [0.2,0.15,0.1,0.07, 0.05]:
        zernike_coefficients = optimise_aberration_correction(slm, cam, zernike_coefficients, brightest_g, dz=0.1)
    zernike_coefficients = optimise_aberration_correction(slm, cam, zernike_coefficients, average_fn(beam_sd,3), dz=0.1)
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
    #
    plt.ion()
    plt.figure()
    plt.imshow(intensity)
    plt.figure()
    for i in range(10):
        plt.plot(intensity[:,i])
    plt.figure()
    for i in range(10):
        plt.plot(intensity[i,:])
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
"""
# Scan through a parameter, plotting sections of the beam
xsection = np.zeros((30,640,3),dtype=np.uint8)
ysection = np.zeros((xsection.shape[0],480,3),dtype=np.uint8)
rs = np.linspace(-5,5,xsection.shape[0])
for i, r in enumerate(rs):
    slm.zernike_coefficients = [-1.3,0.8,r,0,0,0,0,0,0,0,0,0,]
    time.sleep(0.1)
    img = cam.color_image()
    img = cam.color_image()
    xsection[i,:,:] = np.mean(img[230:250,:,:], axis=0)
    ysection[i,:,:] = np.mean(img[:,310:330,:], axis=1)
f, axes = plt.subplots(1,2)
axes[0].imshow(xsection,aspect='auto',extent = (0,640,rs.max(), rs.min()))
axes[1].imshow(ysection,aspect='auto',extent = (0,480,rs.max(), rs.min()))
"""