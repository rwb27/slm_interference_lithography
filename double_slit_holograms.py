# -*- coding: utf-8 -*-
"""
Created on Mon Dec 01 12:04:12 2014

@author: Richard
"""

import numpy as np
from opengl_holograms import OpenGLShaderWindow, UniformProperty
from nplab.instrument.camera.thorlabs_uc480 import ThorLabsCamera
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.ndimage
import nplab.utils.gui
import nplab
from nplab.utils.array_with_attrs import ArrayWithAttrs
from threading import Thread
from scipy.ndimage.filters import gaussian_filter
import h5py 
from scipy.interpolate import UnivariateSpline
from threading import Thread

# This is a big string constant that actually renders the hologram.  See below
# for the useful Python code that you might want to use...
SHADER_SOURCE = """
uniform vec4 colours[10];
uniform vec4 rects[10];
uniform int n;
uniform vec2 unit_cell;
uniform vec4 chessboard_colours[2];

void main(){ // basic gratings and lenses for a single spot
  vec2 uv = gl_TexCoord[0].xy;
  
  // the background is a chessboard of 2 colours
  float chessphase = 0.0;
  for(int i=0; i<2; i++){
    chessphase += uv[i]/unit_cell[i];
    chessphase = floor(mod(chessphase, 2));
  }
  gl_FragColor = chessboard_colours[int(chessphase)];
  
  for(int i=0; i<n; i+=1){
    vec4 b = rects[i];
    if(uv[0] >= b[0] && uv[1] >= b[1] && uv[0] < b[2] && uv[1] < b[3]){
        gl_FragColor = colours[i];
    }
  }
//    gl_FragColor = vec4(0,0,1,1);
}
"""

class SlitHologram(OpenGLShaderWindow):
    """A hologram generation class for interference lithography.
    
    This generates holograms with the following features:
    
    * XYZ+intensity control of multiple beams
    * control of the aperture of each beam on the SLM
    * rotationally-symmetric beam shaping
    * aberration correction
    """
    def __init__(self, **kwargs):
        """Create a hologram generator for interference lithography."""
        super(SlitHologram, self).__init__(**kwargs)
        self.shader_source = SHADER_SOURCE
        self.n=0
        self.unit_cell=[0.01,0.01]
        self.chessboard_colours = [0,0,0,1,   1,1,1,1]
    
        
    colours = UniformProperty(0, max_length=10*4)
    rects = UniformProperty(1, max_length=10*4)
    n = UniformProperty(2, max_length=1)
    unit_cell = UniformProperty(3, max_length=2)
    chessboard_colours = UniformProperty(4, max_length=2*4)
    
    def set_phases(self, *args):
        """Set the phases of the slits, by setting all 3 colour channels equal."""
        colours = [[p,p,p,1.0] for p in args]
        self.colours = np.array(colours).flatten()
        
    def set_chessboard(self, phase_a, phase_b, width, height):
        """Set up the chessboard to be 1 pixel squares"""
        self.unit_cell = [1.0/width, 1.0/height]
        self.chessboard_colours = [phase_a, phase_a, phase_a, 1.0,
                                   phase_b, phase_b, phase_b, 1.0]
    
    def cycle_phases(self, interval, phases, repeat=1):
        """Cycle through phase values."""
        for i in range(repeat):
            for p in phases:
                self.set_phases(*p)
                time.sleep(interval)

def measure_phase_throw(slm, cam, grays=np.linspace(0,1,32), axis=1, repeats=5):
    """Cycle through gray levels and measure phase shift"""
    slm.set_phases(0.5,0)
    time.sleep(0.5)
    cam.gray_image()
    template = create_template(np.sum(cam.gray_image(), axis=axis))
    phases = np.zeros((len(grays), repeats))
    periods = np.zeros_like(phases)
    for i, g in enumerate(grays):
        slm.set_phases(0.5,g)
        time.sleep(0.5)
        cam.gray_image()
        for j in range(repeats):
            phases[i,j], periods[i,j] = find_phase_and_period(np.sum(cam.gray_image(),axis=1), template)
    return grays, phases, periods

def phase_throw_image(slm, cam, grays=np.linspace(0,1,32), axis=1, dt=0.1):
    marginal = np.sum(cam.gray_image(), axis=axis)
    slices = np.zeros((len(grays), len(marginal)))
    for i, g in enumerate(grays):
        slm.set_phases(0.5,g)
        time.sleep(dt)
        slices[i,:] = np.sum(cam.gray_image(), axis=axis)
    return slices
         
        
def find_rising_zero_crossings(y, fit_r=0, fit_deg=1):
    """Find the (interpolated) zero crossings of an array."""
    crossings = np.argwhere(np.logical_and(y[1:]>0, y[:-1]<0))
    subpixel_crossings = []
    for index in crossings:
        i = index[0] # we're only using 1d arrays
        if i>fit_r and i<len(y)-1-fit_r:
            cx = np.arange(i-1, i+fit_r+2, dtype=int) # crossing is between i, i+1
            fit = np.polyfit(cx, y[cx], deg=fit_deg)
            roots = np.roots(fit)
            subpixel_crossings.append(roots[np.argmin((roots-i)**2)])
            
        #fractional_shift = abs(y[i+1])/(abs(y[i])+abs(y[i+1]))
        #subpixel_crossings.append(i+fractional_shift)
    return np.array(subpixel_crossings)
    
def create_template(marginal, smoothing=20):
    """Make a template to use for finding phase"""
    return marginal - gaussian_filter(marginal,smoothing)

def find_phase_and_period(marginal, template, crop=200, crossing_args={}):
    corr = np.correlate(marginal, template, mode="full")
    # Corr will have a maximum at element (N-1) and is (2N-1) long.
    # We want to find peaks - so find the zero crossings of the first
    # derivative (for the first few points, we don't care about noise)
    first_derivative = np.diff(corr[len(marginal)-1:])[0:crop]
    minima = find_rising_zero_crossings(first_derivative, **crossing_args) + 0.5
    maxima = find_rising_zero_crossings(-first_derivative, **crossing_args) + 0.5
    # Minima and maxima always alternate - but we should maybe check...
    assert abs(len(minima) - len(maxima)) <= 1, "Maxima/minima don't match!"
    x = np.arange(len(minima) + len(maxima))/2.0
    extrema = np.zeros_like(x)
    if minima[0] < maxima[0]:
        x += 0.5
        extrema[0::2] = minima
        extrema[1::2] = maxima
    else:
        extrema[0::2] = maxima
        extrema[1::2] = minima
    m = np.polyfit(x, extrema, 1)
    #plt.plot(x, extrema, "r+")
    #plt.plot(x, m[0]*x + m[1])
    period = m[0]
    phase = m[1] / period * 2 * np.pi
    return phase, period
    
def unwrap_phase(y):
    """Remove phase wraps from a 1D array of phase values"""
    dy = np.diff(y)
    while np.any(np.abs(dy) > np.pi):
        dy[dy > np.pi] -= 2*np.pi
        dy[dy < -np.pi] += 2*np.pi
    # np.cumsum doesn't prepend a zero (which would undo np.diff more neatly)
    # hence prepending an extra element.
    return np.concatenate((np.array([0]), np.cumsum(dy))) + y[0]

def lookup_table_from_phases(grays, phases, plot=False):
    """Return a 32-element look up table based on a phase measurement."""
    grays = np.array(grays)
    phases = np.array(phases)
    unwrapped_phases = np.zeros_like(phases)
    for i in range(phases.shape[1]):
        unwrapped_phases[:,i] = unwrap_phase(phases[:,i])
    phase = np.mean(unwrapped_phases, axis=1)
    phase -= np.mean(phase)
    
    gradient = np.polyfit(grays, phase, 1)[0]
    phase_sd = np.mean(np.diff(phase-grays*gradient)**2)**0.5
    gray_sd = phase_sd/gradient
    
    lut_phase = np.linspace(-np.pi,np.pi,32)
    sort_indices = np.argsort(phase)
    spline = UnivariateSpline(phase[sort_indices], grays[sort_indices], w=np.ones_like(grays)/gray_sd, k=3, ext="const")
    
    if(plot):
        plt.plot(phase, grays, '+')
        plt.plot(phase, spline(phase), '-')
        plt.plot(lut_phase, spline(lut_phase), 'o')
    
    return lut_phase, spline(lut_phase)
    
if __name__ == "__main__":
    slm = SlitHologram()
    #slm.move_hologram(0,0,512,512)
    slm.move_hologram(1920,0,1920,1152)
    slm.set_aspect(4.0/3.0)
    slm.n = 0
    slm.unit_cell = [1.0/256 for i in range(2)]
    slm.set_chessboard(0.25,0.75,800,600)
    slm.rects = [0,0,1,0.5,   0,0.5,1,1]
    #slm.rects = [0,0.2,1,0.3,   0,0.7,1,0.8]
    slm.colours = [0.5,0,0,1,   0,0.5,0.5,1]
    slm.n = 2
    
    cam = ThorLabsCamera()
    #Thread(target=slm.cycle_phases, args=(0.1, [(0.5,p) for p in np.linspace(0,1,20)], 5)).start()
    df = nplab.current_datafile()
    cam.exposure = 5
    cam.live_view = True
  
"""
    # This block will take a detailed gray level to phase measurement

    data_group = df.create_group("phase_sweep_%d")
    data_group.attrs['SLM'] = "Medowlark"
    data_group.attrs['laser'] = "HeNe"
    gray_levels = np.linspace(0,1,256)
    data_group['gray_levels'] = gray_levels
    
    img = cam.gray_image()
    marginal = np.sum(img, axis=1)
    template = create_template(marginal)
    data_group['template'] = template
    data_group['template'].attrs['description'] = "We correlate this against the marginal distribution of the images"
    data_group['template_image'] = img
    data_group['template_image'].attrs['description'] = "This is the image that the template is generated from"
    
    fringes = phase_throw_image(slm, cam, grays=gray_levels)
    data_group['marginals_vs_gray'] = fringes
    data_group['marginals_vs_gray'].attrs['description'] = "Each row is the marginal distribution of the image for each gray level"
    
    phases = np.zeros((fringes.shape[0]))
    pa = np.zeros_like(phases)
    for i in range(fringes.shape[0]):
        marginal = fringes[i,:]
        phases[i], period = find_phase_and_period(marginal, template, crossing_args={'fit_r':0, 'fit_deg':1})
        pa[i], period = find_phase_and_period(marginal, template, crossing_args={'fit_r':2, 'fit_deg':2})
    plt.imshow(fringes[:,400:600], aspect="auto")
    plt.plot(phases*period/2/np.pi+100, np.arange(len(phases)), 'r-', linewidth=3)
    
    grays, phases, periods = measure_phase_throw(slm, cam, grays=gray_levels, axis=1, repeats=5)
    data_group['phases'] = phases
    data_group['phases'].attrs['description'] = "5 measurements of phase for each gray level, in radians"
    data_group['periods'] = periods
    plt.figure()
    for i in range(phases.shape[1]):
        plt.plot(grays, phases[:,i], '.')
    plt.plot(grays, np.mean(phases, axis=1), '-')
    df.close()
"""
"""
    # To turn that measurement into a look-up table:
    g = data_group
    LUT = lookup_table_from_phases(g['gray_levels'], g['phases'], plot=True)
    np.savez("medowlark_633_lut.npz", phases=LUT[0], grays=LUT[1])
"""
