# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:26:16 2016

@author: hera
"""

from nplab.instrument import Instrument
from nplab.instrument.light_sources.ondax_laser import OndaxLaser
from nplab.instrument.shutter.southampton_custom import ILShutter
from nplab.instrument.camera.opencv import OpenCVCamera
from nplab.instrument.camera.thorlabs_uc480 import ThorLabsCamera
from slm_interference_lithography import VeryCleverBeamsplitter
from nplab.utils.gui import show_guis
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.ndimage
from rgb_to_hdr import calibrate_hdr_from_rgb, hdr_from_rgb, fit_channels
import nplab.utils.gui
import nplab
from nplab.utils.array_with_attrs import ArrayWithAttrs

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
    
def sequential_shack_hartmann(slm, snapshot_fn, spot, N, overlap=0.0, other_spots=[], pause=False,save=True):
    """Scan a spot over the SLM, recording intensities as we go"""
    results = ArrayWithAttrs(np.zeros((N, N, 3))) # For each position, find X,Y,I
    results.attrs['spot'] = spot
    results.attrs['n_apertures'] = N
    results.attrs['overlap'] = overlap
    results.attrs['other_spots'] = other_spots
    results.attrs['pause_between_spots'] = pause
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
    if save:
        dset = nplab.current_datafile().create_dataset("sequential_shack_hartmann_%d", data=results)
        return dset
    else:
        return results
    
def plot_shack_hartmann(results, threshold=0.1):
    """Plot the results of the above wavefront sensor"""
    N = results.shape[0]
    weights = results[:,:,2] > results[:,:,2].max()*threshold
    centre = np.mean(results[:,:,:2]*weights[:,:,np.newaxis], axis=(0,1))/np.mean(weights)
    r = np.max(np.abs((results[:,:,:2]-centre)*weights[:,:,np.newaxis]))
    f, ax = plt.subplots(1,2)
    for i in range(N):
        for j in range(N):
            if weights[i,j] > 0:
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
    
    
def calibrate_hdr():
    """Fit the red channel vs the blue channel to calibrate the camera
    
    We return a snapshot function that yields an HDR image, made by using the 
    red pixels in the Bayer patern to reconstruct the image when the blue
    channel is saturated.
    """
    s = [0,0,0]
    o = [0,0,1]
    img = cam.color_image()
    s[0], o[0] = fit_channels(img, 0, 2, amin=10, amax=50, plot=True)
    df.create_dataset("hdr_calibration_image_%d",data=img, attrs={'slopes':s,'offsets':o})
    hdr = hdr_from_rgb(o,s,cam.color_image())
    for y in [200,210,220,230,240,250,260,270,280]:
        plt.plot(hdr[y,:])
    df.create_dataset("hdr_calibration_image_%d",data=img, attrs={'slopes':s,'offsets':o})
    return snapshot_fn(cam, s, o)

def focus_stack(N, dz, snap=None):
    """Shift the focus (using Zernike modes) and acquire images."""
    if snap is None:
        global cam
        snap = lambda: cam.color_image()[:,:,2]
    img = snap()
    focus_stack = ArrayWithAttrs(np.zeros((N,)+img.shape, dtype=img.dtype))
    for i, d in enumerate(np.linspace(-dz,dz,N)):
        z = zernike_coefficients.copy()
        z[1] += d
        slm.zernike_coefficients = z
        focus_stack[i,:,:] = snap()
    slm.zernike_coefficients=zernike_coefficients
    plt.figure()
    plt.imshow(focus_stack[:,240,:],aspect="auto")
    focus_stack.attrs["dz"]=dz
    focus_stack.attrs["zernike_coefficients"]=zernike_coefficients
    dset = nplab.current_datafile().create_dataset("zstack_%d",data=focus_stack)
    return dset

if __name__ == '__main__':
    slm = VeryCleverBeamsplitter()
    #shutter = ILShutter("COM1")
    #laser = OndaxLaser("COM4")
    cam = ThorLabsCamera()
    slm.move_hologram(1920,0,1024,768)
    # set uniform values so it has a blazing function and no aberration correction
    blazing_function = np.array([  0,   0,   0,   0,   0,   0,   0,   0,   0,  12,  69,  92, 124,
       139, 155, 171, 177, 194, 203, 212, 225, 234, 247, 255, 255, 255,
       255, 255, 255, 255, 255, 255]).astype(np.float)/255.0 #np.linspace(0,1,32)
    def dim_slm(dimming):
        slm.blazing_function = (blazing_function - 0.5)*dimming + 0.5
    slm.blazing_function = blazing_function
    known_good_zernike = np.array([ 0.1 ,  0.75,  0.5 ,  0.  , -0.55,  0.45,  
                                   0.5 ,  0.5 ,  0.5 , -0.75,  0.05, -0.45])
    slm.zernike_coefficients = known_good_zernike
    slm.zernike_coefficients = np.zeros(12)
    distance = 2325e3
    slm.update_gaussian_to_tophat(1900,3000, distance=distance)
    slm.update_gaussian_to_tophat(1900,1, distance=distance)
    slm.disable_gaussian_to_tophat()
    slm.make_spots([[20,10,0,1],[-20,10,0,1]])
    test_spot = [-20,10,0,1]
    slm.make_spots([test_spot])
    #shutter.open_shutter()
    #guis = show_guis([shutter, cam], block=False)
    def snap():
        cam.gray_image()
        time.sleep(0.1)
        cam.gray_image()
        return cam.gray_image()
        
    df = nplab.current_datafile()

"""
# These variables need to be defined to match the geometry of your set-up
test_spot = [20,-10,0,1];
distance = 2900e3

# Calibrate HDR processing (not needed unless you're 
# struggling, snap is already defined.)
slm.make_spots([test_spot + [0,0,0.075,0]])
slm.update_gaussian_to_tophat(1900,1, distance=distance)
## TURN LIGHTS OFF!
#cam.exposure=-2
#snap = calibrate_hdr()

# Sequential Shack-Hartmann sensor
slm.make_spots([test_spot + [0,0,0.075,0]])
zernike_coefficients = np.zeros(12)
slm.zernike_coefficients = zernike_coefficients
#dim_slm(1)
#cam.exposure=-5
res = sequential_shack_hartmann(slm, snap, [20,-10,0,1], 10, overlap=0.5)
plot_shack_hartmann(res)

# A brutal attempt at modal decomposition (not used)
modes = measure_modes(slm, snap, [20,10,2050,1], 5, overlap=0)
flat = modes[12]
f, axes = plt.subplots(3,4)
axes_flat = [axes[i,j] for i in range(3) for j in range(4)]
for m, ax in zip(modes[:12], axes_flat):
    plot_sh_shifts(ax, m - flat, discard_edges=1)
"""
"""
# Analysis of the intensity distribution on the SLM
# NB slmsize is set at 6.9mm for the CC SLM
slm_size = 6.9
initial_r = 1.9
N = res.shape[0]
x = (np.arange(N)-(N-1)/2.0)/N * slm_size #centres of apertures
gaussian = np.exp(-x**2/initial_r**2)
gaussian /= np.mean(gaussian)
f, axes = plt.subplots(1,2)
# plot the data and a Gaussian fit
for data, ax in zip([res[:,:,2], res[:,:,2].T], axes):
    for i in range(N):
        ax.plot(x, data[:,i], '+', color=plt.cm.gist_rainbow(float(i)/N))
        m = np.polyfit(gaussian, data[:,i], 1)
        ax.plot(x, gaussian*m[0]+m[1], '-', color=plt.cm.gist_rainbow(float(i)/N))
# plot the data against the Gaussian
f, axes = plt.subplots(1,2)
for data, ax in zip([res[:,:,2], res[:,:,2].T], axes):
    for i in range(N):
        ax.plot(gaussian, data[:,i], '-+', color=plt.cm.gist_rainbow(float(i)/N))

# plot the data vs r
r = np.sqrt(x[:,np.newaxis]**2 + x[np.newaxis,:]**2)
f, ax = plt.subplots(1,1)
ax.plot(r.flatten(), res[:,:,2].flatten(), '.')

# find the centroid
plt.figure()
s=slm_size/2.0
plt.imshow(res[:,:,2],extent=(-s,s,-s,s))
for thresh in [0.0,0.1,0.2,0.5,0.9]:
    I = res[:,:,2].copy()
    I /= np.max(I)
    I -= thresh
    I[I<0] = 0
    cx = np.sum(x[:,np.newaxis]*I)/np.sum(I)
    cy = np.sum(x[np.newaxis,:]*I)/np.sum(I)
    hide = plt.plot(cy,cx,'+')
    print "found centroid at {}, {} with threshold {}".format(cx, cy, thresh)

# pick the right values for cx and cy...

# plot the data vs r
r = np.sqrt((x[:,np.newaxis]-cx)**2 + (x[np.newaxis,:]-cy)**2)
f, ax = plt.subplots(1,1)
ax.plot(r.flatten(), res[:,:,2].flatten(), '.')

radii = r.flatten()
I = res[:,:,2].flatten()/I.max()*10
from scipy.interpolate import UnivariateSpline
spl = UnivariateSpline(np.concatenate([radii,-radii]), np.tile(I,2))
rr = np.linspace(0,slm_size/np.sqrt(2),100)
plt.plot(rr,spl(rr))
plt.plot(radii,I,'.')

slm.centre=(cx,cy)
slm.radial_blaze_function = np.ones(384)
inner_edge_i = 1500//18 #bad
sd = 400/18.0
inner_edge_i = 1000//18 #good
sd = 1000/18.0
nrest = 384 - inner_edge_i
radial_blaze_function = np.concatenate([np.ones(inner_edge_i),
                                        np.exp(-(np.arange(nrest))**2/2.0/sd**2)])
slm.radial_blaze_function = radial_blaze_function
"""
"""
# Optimise SLM for aberrations with modal wavefront sensor
zernike_coefficients = np.zeros(12)
slm.zernike_coefficients = zernike_coefficients
#slm.make_spots([test_spot + [0,0,0.5,0]])
slm.make_spots([test_spot])
slm.update_gaussian_to_tophat(1900,1, distance=distance)
#dim_slm(1)
#dim_slm(0.75)
#dim_slm(0.5)
#dim_slm(0.2)
#dim_slm(0.1)
#cam.exposure=-2
def brightest_hdr():
    hdr = snap()
    for i in range(4):
        hdr += snap()
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
def beam_sd():
    cam.color_image()
    time.sleep(0.1)
    hdr = snap()
    x = np.mean(hdr * np.arange(hdr.shape[0])[:,np.newaxis])/np.mean(hdr)
    y = np.mean(hdr * np.arange(hdr.shape[1])[np.newaxis,:])/np.mean(hdr)
    dx2 = np.mean(hdr * ((np.arange(hdr.shape[0])-x)**2)[:,np.newaxis])/np.mean(hdr)
    dy2 = np.mean(hdr * ((np.arange(hdr.shape[1])-y)**2)[np.newaxis,:])/np.mean(hdr)
    sd = np.sqrt(dx2+dy2)
    return 1/sd
def average_fn(f, n):
    return lambda: np.mean([f() for i in range(n)])
merit_function = lambda: np.mean([beam_sd() for i in range(3)])

# Start by autofocusing (optimising mode 2)
zernike_coefficients = optimise_aberration_correction(slm, cam, zernike_coefficients, beam_sd, dz=0.5, modes=[1])
zernike_coefficients = optimise_aberration_correction(slm, cam, zernike_coefficients, beam_sd, dz=0.1, modes=[1])
zernike_coefficients = optimise_aberration_correction(slm, cam, zernike_coefficients, beam_sd, dz=0.3, modes=[0,1,2])
for dz in [0.2,0.15,0.1,0.07]:
    print "step size: {}".format(dz)
    zernike_coefficients = optimise_aberration_correction(slm, cam, zernike_coefficients, beam_sd, dz=dz)
zernike_coefficients = optimise_aberration_correction(slm, cam, zernike_coefficients, average_fn(beam_sd,3), dz=0.1)
zernike_coefficients = optimise_aberration_correction(slm, 
                                cam, zernike_coefficients, merit_function, 
                                dz=0.5)
zernike_coefficients = optimise_aberration_correction(slm, 
                                cam, zernike_coefficients, merit_function, 
                                dz=0.1)
nplab.current_datafile().create_dataset("spot_image_%d",data=cam.color_image(),attrs={"zernike_coefficients":zernike_coefficients})

"""

"""
# Really thorough optimisation of defocus
# Start with a visible, nicely-focused spot
slm.make_spots([test_spot + [0,0,0.5,0]])
slm.update_gaussian_to_tophat(1900,1, distance=distance)
focus_stack(50,5, snap=snap)

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