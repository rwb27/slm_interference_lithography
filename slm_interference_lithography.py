# -*- coding: utf-8 -*-
"""
Created on Mon Dec 01 12:04:12 2014

@author: Richard
"""

import io
import socket
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq

fn_array = np.zeros((12,))
        
def I_cal(I_set):
    return table[round(float(I_set)*20)/20]
def UDP_send(message):
    sock.sendto(message,ENGINE)
    data, addr = sock.recvfrom(128)
def set_uniform(uniform_id, value):
    """Set the value of a uniform variable.
    
    uniform_id : int
        The number (starting from 0) of the uniform to set
    value : list of float
        The value to give the uniform variable.
    """
    try:
        value[0]
    except:
        value = [value]
    UDP_send("<data>\n<uniform id={0}>\n".format(uniform_id) +
        " ".join(map(lambda x: "%f" % x, value)) +
        "</uniform>\n</data>\n")

def sq_array(x,y,dk=1,d=None,f=0.0,n=3,reference=False):
    x = float(x)
    y = float(y)
    xs = dk
    ys = dk
    r = 1.0 if reference else 0.5/n
    if d is None: d = (n-1.0)/float(n)
    seq=np.linspace(-0.5,0.5,n)
#    make_spots([[x,y,f,1,0,0,r,0],[x-xs,y-ys,f,1,-d,-d,r,0],[x-xs,y,f,1,-d,0,r,0],[x-xs,y+ys,f,1,-d,d,r,0],[x,y+ys,f,1,0,d,r,0],[x+xs,y+ys,f,1,d,d,r,0],[x+xs,y,f,1,d,0,r,0],[x+xs,y-ys,f,1,d,-d,r,0],[x,y-ys,f,1,0,-d,r,0],])
    make_spots([[x+xs*u,y+ys*v,f,1,d*u,d*v,r,0] for u in seq for v in seq])
def hex_array(x,y,dk=1,d=None,f=0.0,n=2,reference=False):
    x = float(x)
    y = float(y)
    xs = dk
    ys = dk
    r = 1.0 if reference else 0.5/(2*(n+0.5))
    if d is None: d = 0.5/(n+0.5)
    array = [[x,y,f,1,0,0,r,0]]
    if n > 0:
        for z in np.linspace(0,2*np.pi,7):
            v = d*np.sin(z)
            u = d*np.cos(z)
            array.append([x+xs*u,y+ys*v,f,1,u,v,r,0])
        if n > 1:
            for z in np.linspace(0,2*np.pi,7):
                v = 2*d*np.sin(z)
                u = 2*d*np.cos(z)
                array.append([x+xs*u,y+ys*v,f,1,u,v,r,0])
                v1 = np.sqrt(3)*d*np.sin(z+np.pi/6)
                u1 = np.sqrt(3)*d*np.cos(z+np.pi/6)
                array.append([x+xs*u1,y+ys*v1,f,1,u1,v1,r,0])
    make_spots(array)
def flash_sq(*args,**kwargs):
    while True:
        sq_array(*args, **kwargs)
        time.sleep(0.5)
        sq_array(*args, reference=True, **kwargs)
        time.sleep(0.5)
def flash_hex(*args,**kwargs):
    while True:
        hex_array(*args, **kwargs)
        time.sleep(0.5)
        hex_array(*args, reference=True, **kwargs)
        time.sleep(0.5)
def vary_fn(n,x):
    global fn_array
    fn_array[n] = x
    UDP_send("""<data><uniform id=3>""" + " ".join(map(lambda x: "%f" % x, fn_array)) + """</uniform></data>""")
def make_spots(spots):
    """Use the gratings-and-lenses algorithm to make a number of spots.
    
    Spots should be passed as a list of lists.  Each list is one spot, elements
    are:
    [0] x position in mm
    [1] y position in mm
    [2] z (focus) position in mm
    [3] I intensity
    [4] u centre of spot in back aperture
    [5] v centre of spot in back aperture
    [6] r size of spot in back aperture
    [7] not currently used
    
    Spots should be either 4 or 8 elements long.
    """
    alt = [0,0,1,0]
    if len(spots[0]) == 4:
        for x in spots:
            x.extend(alt) #if the spots are missing NA information, add it
#    for x in spots:
#        x[3] = I_cal(x[3])
    spots = np.array(spots)
    assert spots.shape[1]==8, "Spots are 8 elements long - your array must be (n,8)"
    UDP_send("""<data>
    <uniform id="0">
    %s
    </uniform>
    <uniform id="1">
    %d
    </uniform>
    </data>
    """ % (" ".join(np.char.mod("%f", np.reshape(spots,spots.shape[0]*spots.shape[1]))),spots.shape[0]))
def move_hologram(x=0, y=0, w=1024, h=768):
    """Move the hologram on the screen."""
    UDP_send("""<data>
<window_rect>
{0},{1},{2},{3}
</window_rect>
</data>
""".format(x, y, w, h))
def set_centre(x=0.5, y=0.5):
    """Move the hologram onto the primary monitor to check it."""
    UDP_send("""<data>
<uniform id=6>
{0} {1}
</uniform>
</data>
""".format(x, y))

def setup_shader():
    """Set up the shader program that will render our holograms."""
    UDP_send("""<data>
<network_reply>
1
</network_reply>
<window_rect>
-1024,0,1024,768
</window_rect>
<shader_source>
uniform vec4 spots[100];
uniform int n;
uniform float blazing[32];
uniform float zernikeCoefficients[12];
uniform float radialPhase[384];
uniform float radialPhaseDr = 0.009 * 2.0;
const float k=15700; //units of mm-1
const float f=1975.0; //mm
const vec2 slmsize=vec2(6.9,6.9); //size of SLM
uniform vec2 slmcentre=vec2(0.5,0.5); //centre of SLM
const float pi = 3.141;

float wrap2pi(float phase){
  return mod(phase + pi, 2*pi) -pi;
}

float phase_to_gray(float phase){
  return phase/2.0/pi +0.5;
}

vec2 unitvector(float angle){
  return vec2(cos(angle), sin(angle));
}

float apply_LUT(float phase){
  int phint = int(floor((phase/2.0/pi +0.5)*30.9999999)); //blazing table element just before our point
  float alpha = fract((phase/2.0/pi +0.5)*30.9999999); //remainder
  return mix(blazing[phint], blazing[phint+1], alpha); //this uses the blazing table with linear interpolation
}

float zernikeAberration(){
//this function is exactly the same as zernikeCombination, except that it uses a uniform
//called zernikeCoefficients as its argument.  This avoids copying the array = more efficient.
  //takes a 12-element array of coefficients, and returns a weighted sum
  //of Zernike modes.  This should now be THE way of generating aberration
  //corrections from Zernikes...
  float x = 2.0*gl_TexCoord[0].x - 1.0;
  float y = 2.0*gl_TexCoord[0].y - 1.0;
  float r2 = x*x+y*y;
  float a = 0.0;
  a += zernikeCoefficients[0] * (2.0*x*y);                                                //(2,-2)
  a += zernikeCoefficients[1] * (2.0*r2-1.0);                                           //(2,0)
  a += zernikeCoefficients[2] * (x*x-y*y);                                               //(2,2)
  a += zernikeCoefficients[3] * (3.0*x*x*y-y*y*y);                                 //(3,-3)
  a += zernikeCoefficients[4] * ((3.0*r2-2.0)*y);                                    //(3,-1)
  a += zernikeCoefficients[5] * ((3.0*r2-2.0)*x);                                    //(3,1)
  a += zernikeCoefficients[6] * (x*x*x-3.0*x*y*y);                                 //(3,3)
  a += zernikeCoefficients[7] * (4.0*x*y*(x*x-y*y));                              //(4,-4)
  a += zernikeCoefficients[8] * ((4.0*r2-3.0)*2.0*x*y);                          //(4,-2)
  a += zernikeCoefficients[9] * (6.0*r2*r2-6*r2+1);                               //(4,0)
  a += zernikeCoefficients[10] * ((4.0*r2-3.0)*(x*x-y*y));                      //(4,2)
  a += zernikeCoefficients[11] * (x*x*x*x-6.0*x*x*y*y+y*y*y*y);          //(4,4)
  return a;
}

float radialPhaseFunction(vec2 uv){
  //calculate a radially-symmetric phase function from the uniform radialPhase
  float r = sqrt(dot(uv,uv));
  int index = int(floor(r/radialPhaseDr));
  float alpha = fract(r/radialPhaseDr);
  return radialPhase[index] * (1.0-alpha) + radialPhase[index+1] * alpha;
}

void main(){ // basic gratings and lenses for a single spot
  vec2 uv = (gl_TexCoord[0].xy - slmcentre)*slmsize;
  vec3 pos = vec3(k*uv/f, -k*dot(uv,uv)/(2.0*f*f));
  float phase, real=0.0, imag=0.0;
  for(int i; i<2*n; i+=2){
    if(length(uv/slmsize - spots[i+1].xy) < spots[i+1][2]){
      phase = dot(pos, spots[i].xyz);
      float amp = spots[i][3];
      real += amp * sin(phase);
      imag += amp * cos(phase);
    }
  }
  phase = atan(real, imag);
  phase += zernikeAberration();
  phase = wrap2pi(phase);
  phase += radialPhaseFunction(uv);
  phase = wrap2pi(phase);
  float g = apply_LUT(phase);
  gl_FragColor=vec4(g,g,g,1.0);
}
</shader_source>
</data>
""")

def gaussian_to_tophat_phase(N, dr, wavelength, initial_waist, target_radius, propagation_distance, wrap=False):
    """Calculate a radial phase function to re-map from gaussian to top-hat.
    
    N: number of points to calculate
    dr: radial spacing between points
    wavelength: wavelength of the light
    initial_waist: 1/e radius of starting beam
    target_radius: size of target top-hat beam
    propagation_distance: distance from SLM to target plane
    
    Units can be anything - but should be self-consistent.  Microns are good.
    """
    k = np.pi*2/wavelength
    # Next, calculate the phase shift as a function of radius
    def cumulative_I_gaussian(r, w):
        """The fraction of a 2D Gaussian contained within a given radius."""
        return 1 - np.exp(-r**2/(w**2)) #NB there's no 2 as it's *intensity*
    def cumulative_I_tophat(r, r_beam):
        """Fraction of a top-hat contained within a given radius."""
        if r<r_beam:
            return (r/r_beam)**2
        else:
            return 1
    def inverse_cumulative_I_tophat(I_cum, r_beam):
        """Radius at which a given fraction of the beam is enclosed."""
        return np.sqrt(I_cum) * r_beam
    
    phase_shift = np.zeros((N)) #array the right length
    tilt = np.zeros(len(phase_shift)-1)
    for i in range(len(tilt)):
        r = (i+0.5)*dr
        target_r = inverse_cumulative_I_tophat(
                        cumulative_I_gaussian(r, initial_waist),
                        target_radius)
        tilt[i] = (target_r-r)/propagation_distance
    for i in range(1,len(phase_shift)):
        #this is just a lens
        phase_shift[i] = phase_shift[i-1] + tilt[i-1] * k * dr
    if wrap:
        return ((phase_shift + np.pi) % (2 * np.pi)) - np.pi
    else:
        return phase_shift
        
def update_gaussian_to_tophat(initial_r, final_r, distance=750e3):
    """Update the parameters used to map gaussian -> top hat"""
    gaussian_to_tophat = gaussian_to_tophat_phase(384, #length
                                                  9*2, #pixel size/um
                                                  0.4, #wavelength/um
                                                  initial_r, #initial 1/e radius
                                                  final_r, #final radius
                                                  distance, #propagation distance
                                                  wrap=False) #don't phase-wrap
    set_uniform(4, gaussian_to_tophat)

def disable_gaussian_to_tophat():
    """Set the radial phase function to zero"""
    set_uniform(4, np.zeros((384,)))
    
def make_shack_hartmann(N, width, x=0, y=0, z=0, ref=False):
    """Make a square-array shack-hartmann sensor.
    
    N: number of spots across the sensor
    width: size of the spot array in the focal plane
    x,y,z: centre of the spot array in the focal plane
    ref: generate a reference array
    """
    us = (np.arange(N)-(N-1.0)/2)/N
    vs = us
    spots = [[u*width+x,v*width+y,z,1,u,v,1.0 if ref else 0.5/N,0] for u in us for v in vs]
    make_spots(spots)
    #return spots
 
if __name__ == "__main__":
    UDP_IP = "127.0.0.1"
    UDP_PORT = 61557
    ENGINE = (UDP_IP, UDP_PORT)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #make a UDP socket
    sock.settimeout(1.0)

    setup_shader();

    # set uniform values so it has a blazing function and no aberration correction
    blazing_function = np.array([  0,   0,   0,   0,   0,   0,   0,   0,   0,  12,  69,  92, 124,
       139, 155, 171, 177, 194, 203, 212, 225, 234, 247, 255, 255, 255,
       255, 255, 255, 255, 255, 255]).astype(np.float)/255.0 #np.linspace(0,1,32)
    
    set_uniform(0, [1.0, 5.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,])
    set_uniform(1, 1)
    set_uniform(2, blazing_function)
    set_uniform(3, np.zeros((12,)))
    update_gaussian_to_tophat(1900,3000, distance=1975e3)
    make_spots([[20,10,0,1],[-20,10,0,1]])
    
