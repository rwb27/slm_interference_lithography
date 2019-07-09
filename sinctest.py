# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:38:15 2019

@author: kh302
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq
from nplab.utils.array_with_attrs import ArrayWithAttrs

class Beam(np.ndarray):
    """Beams are represented as ndarrays with some extra properties."""
    
    attributes_to_copy = [ #attributes listed here will be copied/preserved
        "wavelength", #wavelength of the beam
        "dx", #spacing between pixels, in real-space beams
        "dk", #spacing between pixels, in reciprocal-space beams
        ]
    
    def __new__(cls, input_array, attrs=None, **kwargs):
        """Make a new beam based on a numpy array."""
        # the input array should be a numpy array, then we cast it to this type
        obj = np.asarray(input_array).view(cls)
        # next, add the attributes, copying from keyword args if possible.
        for attr in cls.attributes_to_copy:
            setattr(obj, attr, kwargs.get(attr, None))
        if attrs is not None: # copy metadata from another object
            obj.copy_attrs(attrs)
        return obj
    
    def __array_finalize__(self, obj):
        # this is called by numpy when the object is created (__new__ may or
        # may not get called)
        if obj is None: return # if obj is None, __new__ was called - do nothing
        # if we didn't create the object with __new__,  we must add the attrs.
        # We copy these from the source object if possible or create 
        # new ones and set to None if not.
        for attr in self.attributes_to_copy:
            setattr(self, attr, getattr(obj, attr, None))
    
    def copy_attrs(self, obj, exclude=[]):
        """Copy the non-array data from another beam.
        
        obj: the object to copy from
        exclude: a list of attribute names not to copy
        
        NB this will only copy attributes that are named in
        the class as ones to copy.
        """
        for attr in self.attributes_to_copy:
            if attr not in exclude:
                setattr(self, attr, getattr(obj, attr, None))
                
    @property
    def k(self):
        """Magnitude of the wavevector"""
        return 2*np.pi/self.wavelength
    @property
    def kx(self):
        """Return the x component of the wavevector for each column"""
        dim = 0 if self.ndim == 1 else -2
        fractions = fftshift(fftfreq(self.shape[dim]))
        return (fractions*self.dk*self.shape[dim])[:,np.newaxis]
    @property
    def ky(self):
        """Return the y component of the wavevector for each row"""
        assert self.ndim > 1, "This property needs 2D beams"
        fractions = fftshift(fftfreq(self.shape[-1]))
        return (fractions*self.dk*self.shape[-1])[np.newaxis,:]
    @property
    def kz(self):
        """Z component of the wavevector for each pixel."""
        return np.sqrt(self.k**2 - self.kx**2 - self.ky**2) 
    
    @property
    def x(self):
        """Return the x position for each column"""
        N = self.shape[0 if self.ndim == 1 else -2]
        return (np.linspace(-(N-1)/2.0,(N-1)/2.0,N)*self.dx)[:,np.newaxis]
    @property
    def y(self):
        """Return the y position for each row"""
        assert self.ndim > 1, "This property needs 2D beams"
        N = self.shape[-1]
        return (np.linspace(-(N-1)/2.0,(N-1)/2.0,N)*self.dx)[np.newaxis,:]
    
def i2x(i,beam,index=0):
    """Convert an index in a beam array to a position"""
    dx = beam.dx
    return (i - np.floor(beam.shape[index]/2)) * dx
def i2k(i, beam, index=0):
    """Convert an index in a beam array to a k vector"""
    dk = beam.dk
    return (i - np.floor(beam.shape[index]/2)) * dk
    
def tophat_beam(N, dx, wavelength, r):
    """Create a top-hat beam"""
    beam = Beam(np.zeros((N,N), dtype=np.complex), 
                dx=dx, 
                wavelength=wavelength)
    beam[beam.x**2+beam.y**2 < r**2] = 1
    return beam

def gaussian_beam(N, dx, wavelength, r):
    """Create a Gaussian beam with 1/e radius r"""
    beam = Beam(np.zeros((N,N), dtype=np.complex), 
                dx=dx, 
                wavelength=wavelength)
    # the [:,:] means we just replace the data, we
    # don't make a whole new object
    beam[:,:] = np.exp(-(beam.x**2 + beam.y**2)/(2 * r**2))
    return beam

def sinc_beam(N, dx, wavelength, r):
    """Create a Sinc beam with 1/e radius r"""
    beam = Beam(np.zeros((N,N), dtype=np.complex), dx=dx, wavelength=wavelength)
    # the [:,:] means we just replace the data, we
    # don't make a whole new object
    beam[:,:] = np.sinc(np.sqrt((beam.x**2 + beam.y**2)/(2 * r**2)))
    return beam

N=512
beam_size = 20000 #we're working in um
wavelength = 0.405#um

def FFT(beam):
    """Take the FFT of a beam"""
    beam_ft = Beam(fftshift(fftn(beam)), attrs=beam)
    beam_ft.dk = 2*np.pi/(beam.dx*beam.shape[0])
    beam_ft.dx = None # Fourier-space beams shouldn't have dx
    return beam_ft
def IFFT(beam_ft):
    """Take the IFFT of a beam"""
    beam = Beam(ifftn(ifftshift(beam_ft)), attrs=beam_ft)
    beam.dx = 2*np.pi/(beam_ft.dk*beam_ft.shape[0])
    beam.dk = None # Real-space beams shouldn't have dk
    return beam

f, ax = plt.subplots(1,3)
beam = tophat_beam(N, beam_size/N, wavelength, 1000)
beam = gaussian_beam(N, beam_size/N, wavelength, 500)
ax[0].imshow(np.abs(beam),cmap="cubehelix")
ax[1].imshow(np.abs(FFT(beam)),cmap="cubehelix")
ax[2].imshow(np.abs(IFFT(FFT(beam))),cmap="cubehelix")

plt.show()

def propagate_incremental(beam, dz, N=1):
    """Propagate a beam by phase-shifting its FT by kz*dz
    
    The propagation is performed N times, and the beam is
    returned as a 3D array, where the first index is the
    propagation number.  The returned array will have size
    N+1, as the original beam is included.
    
    In the future this should include absorbing edges."""
    # Make a new beam to store the result
    propagation = Beam(
        np.empty((N+1,)+beam.shape, dtype=np.complex),
        attrs=beam)
    propagation[0,:,:] = beam
    propagator = np.exp(1j * FFT(beam).kz * dz)
    for i in range(N):
        beam_ft = FFT(propagation[i,:,:])
        beam_ft *= propagator
        propagation[i+1,:,:] = IFFT(beam_ft)
    return propagation

def propagate_fast(beam, dz):
    """Quickly propagate a beam by a given dz (in one step).
    
    This corresponds to a single iteration of 
    propagate_incremental."""
    beam_ft = FFT(beam)
    return IFFT(beam_ft * np.exp(1j * beam_ft.kz * dz))

import matplotlib.colorbar as cbar
dz = 10e3
N=512
pbeam = tophat_beam(N, 20000.0/N, 0.4, 2000)
propagation = propagate_incremental(pbeam, dz, 100)
profile = np.abs(propagation[:,:,propagation.shape[2]/2])
plt.imshow(profile.T,cmap="cubehelix",aspect="auto")

plt.show()

def phase_and_intensity_image(beam,vmin=0,vmax=None):
    """Return an RGB array where brightness=intensity and hue=phase.
    
    vmin and vmax set the max and min intensity values.
    """
    if vmax is None:
        vmax = float(np.max(np.abs(beam)**2))
    normalised_phase = (np.angle(beam) + np.pi)/(2*np.pi/3)
    # we use a 3-segment colour map: cyan -> magenta -> yellow -> cyan
    colours = np.array([[0,1,1],[1,0,1],[1,1,0],[0,1,1]]).astype(np.float)
    segment = np.floor(normalised_phase).astype(np.int) # 1, 2, or 3
    remainder = normalised_phase - segment
    # find the starting (A) and ending (B) colours for each pixel
    A = np.array([ colours[:,i][segment] for i in range(3)])
    B = np.array([ colours[:,i][segment+1] for i in range(3)])
    hued_image = A * (1-remainder) + B * remainder
    return (255.99 * hued_image.transpose(1,2,0)
            * ((np.abs(beam)**2 - vmin)/(vmax-vmin))[:,:,np.newaxis]
            ).astype(np.uint8)
def show_beam(beam, axes=None, length_units=1000, **kwargs):
    """display a beam, by plotting a phase/intensity image.
    
    beam: the beam to be plotted
    axes: a matplotlib axes object to plot in (or None)
    length_units: divisor for the X/Y axes (e.g. 1000 gives mm)
    Extra keyword arguments are passed to axes.imshow()
    """
    if axes is None:
        axes=plt
    plot_args = {"aspect":1}
    try:
        u=length_units
        plot_args['extent']=(np.min(beam.x)/u,np.max(beam.x)/u,
                             np.min(beam.y)/u,np.max(beam.y)/u)
    except:
        pass #ignore errors if we're in units of k
    plot_args.update(kwargs)
    axes.imshow(phase_and_intensity_image(beam),**plot_args)
    plt.show()
    # Define the simulation parameters
N = 512
beamsize = 20000
wavelength = 0.4
initial_waist = 3000
propagation_distance = 5e5

# We start with a Gaussian beam
initial_beam = gaussian_beam(N, beamsize/N, wavelength, initial_waist)

# Next, calculate the phase shift as a function of radius
dr = initial_beam.dx/2.0
phase_shift = np.zeros(int(beamsize/np.sqrt(2)/dr)) #array the right length
for i in range(len(phase_shift)):
    #this is just a lens
    f = propagation_distance * 1.2
    
    r = i*dr
    phase_shift[i] = -r**2 * initial_beam.k / 2 / f

# Turn this 1D radial array into a 2D complex array
r = np.sqrt(initial_beam.x**2+initial_beam.y**2)
hologram = np.empty_like(initial_beam)
for i in range(hologram.shape[0]):
    for j in range(hologram.shape[1]):
        hologram[i,j] = np.exp(1j * phase_shift[int(np.floor(r[i,j]/dr))])
        
# Apply the hologram and propagate
output = propagate_fast(initial_beam * hologram, propagation_distance)

plt.imshow(np.abs(output)**2, cmap="cubehelix")

plt.show()

# Define the simulation parameters
N = 512
beamsize = 20000.0
wavelength = 0.4
initial_waist = 3000.0
target_radius = 4000.0
propagation_distance = 5.0e5

# We start with a Gaussian beam
initial_beam = gaussian_beam(N, beamsize/N, wavelength, initial_waist)

# Next, calculate the phase shift as a function of radius
def cumulative_I_gaussian(r, w):
    """The fraction of a 2D Gaussian contained within a given radius."""
    return 1 - np.exp(-r**2/(2*w**2))
def cumulative_I_tophat(r, r_beam):
    """Fraction of a top-hat contained within a given radius."""
    if r<r_beam:
        return (r/r_beam)**2
    else:
        return 1
def inverse_cumulative_I_tophat(I_cum, r_beam):
    """Radius at which a given fraction of the beam is enclosed."""
    return np.sqrt(I_cum) * r_beam
    
dr = initial_beam.dx/2.0
phase_shift = np.zeros(int(beamsize/np.sqrt(2)/dr)) #array the right length
tilt = np.zeros(len(phase_shift)-1)
for i in range(len(tilt)):
    r = (i+0.5)*dr
    target_r = inverse_cumulative_I_tophat(
                    cumulative_I_gaussian(r, initial_waist),
                    target_radius)
    tilt[i] = (target_r-r)/propagation_distance
for i in range(1,len(phase_shift)):
    #this is just a lens
    phase_shift[i] = phase_shift[i-1] + tilt[i-1] * initial_beam.k * dr

# Turn this 1D radial array into a 2D complex array
r = np.sqrt(initial_beam.x**2+initial_beam.y**2)
hologram = np.empty_like(initial_beam)
for i in range(hologram.shape[0]):
    for j in range(hologram.shape[1]):
        hologram[i,j] = np.exp(1j * phase_shift[int(np.floor(r[i,j]/dr))])
        
# Apply the hologram and propagate
propagation = propagate_incremental(
    initial_beam * hologram, 
    propagation_distance/100, 
    100
    )
#plt.imshow(
#    profile.T,
#    cmap="cubehelix",
#    aspect="auto",
#    extent=(0,propagation_distance,np.min(profile.x),np.max(profile.x),)
#    )
profile = np.abs(propagation[:,:,propagation.shape[2]/2])
output = propagation[-1,:,:]

f, ax = plt.subplots(1,3,figsize=(9,3))
#ax[0].plot(np.arange(len(phase_shift))*dr/1000, phase_shift)
show_beam(initial_beam,ax[0])
show_beam(profile.T,
          ax[1],
          extent=(0,propagation_distance/1000,
                  np.min(profile.x)/1000,np.max(profile.x)/1000),
          aspect="auto",
         )
show_beam(output,ax[2])

# Define the simulation parameters
N = 512
beamsize = 20000.0
wavelength = 0.4
initial_waist = 3000.0
target_radius = 3200.0
hologram_size = 9000.0
propagation_distance = 5.0e5

# We start with a Gaussian beam
initial_beam = gaussian_beam(N, beamsize/N, wavelength, initial_waist)

# Next, calculate the phase shift as a function of radius
def cumulative_I_gaussian(r, w):
    """The fraction of a 2D Gaussian contained within a given radius."""
    return 1 - np.exp(-r**2/(w**2)) #NB no 2, it's intensity not |E|
def cumulative_I_tophat(r, r_beam):
    """Fraction of a top-hat contained within a given radius."""
    if r<r_beam:
        return (r/r_beam)**2
    else:
        return 1
def inverse_cumulative_I_tophat(I_cum, r_beam):
    """Radius at which a given fraction of the beam is enclosed."""
    #threshold=0.999
    if I_cum < 1:#threshold:
        return np.sqrt(I_cum) * r_beam
    else:
        return r_beam
    
dr = initial_beam.dx/2.0
phase_shift = np.zeros(int(beamsize/np.sqrt(2)/dr)) #array the right length
tilt = np.zeros(len(phase_shift)-1)
for i in range(len(tilt)):
    r = (i+0.5)*dr
    target_r = inverse_cumulative_I_tophat(
                    cumulative_I_gaussian(r, initial_waist),
                    target_radius)
    tilt[i] = (target_r-r)/propagation_distance
for i in range(1,len(phase_shift)):
    #this is just a lens
    phase_shift[i] = phase_shift[i-1] + tilt[i-1] * initial_beam.k * dr

# Turn this 1D radial array into a 2D complex array
r = np.sqrt(initial_beam.x**2+initial_beam.y**2)
hologram = np.empty_like(initial_beam)
for i in range(hologram.shape[0]):
    for j in range(hologram.shape[1]):
        hologram[i,j] = np.exp(1j * phase_shift[int(np.floor(r[i,j]/dr))])
# Crop the hologram so it's only nonzero on the SLM
hologram[np.abs(initial_beam.x + 0*hologram) > hologram_size] = 0.  
hologram[np.abs(initial_beam.y + 0*hologram) > hologram_size] = 0.  
# Apply the hologram and propagate
propagation = propagate_incremental(
    initial_beam * hologram, 
    propagation_distance/100, 
    100
    )
profile = propagation[:,:,propagation.shape[2]/2]
output = propagation[-1,:,:]

f, ax = plt.subplots(1,3,figsize=(9,3))
#ax[0].plot(np.arange(len(phase_shift))*dr/1000, phase_shift)
show_beam(initial_beam*hologram,ax[0])
show_beam(profile.T,
          ax[1],
          extent=(0,propagation_distance/1000,
                  np.min(profile.x)/1000,np.max(profile.x)/1000),
          aspect="auto",
         )
show_beam(output,ax[2])