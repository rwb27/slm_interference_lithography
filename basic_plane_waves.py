# -*- coding: utf-8 -*-
"""
Created on Thu May 03 11:42:09 2018

@author: kh302
"""

import numpy as np
from opengl_holograms import OpenGLShaderWindow, UniformProperty
# This is a big string constant that actually renders the hologram.  See below
# for the useful Python code that you might want to use...
IL_SHADER_SOURCE = """
uniform vec4 spots[100];
uniform int n;
uniform float blazing[32];
uniform float zernikeCoefficients[12];

const float k=15700; //units of mm-1
const float f=3500.0; //mm
const vec2 slmsize=vec2(6.9,6.9); //size of SLM in mm
uniform vec2 slmcentre=vec2(0.5,0.5); //centre of SLM in units of 0 to 1
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
  //takes a 12-element uniform array of coefficients, and returns a weighted sum
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

void main(){ // basic gratings and lenses
  vec2 uv = (gl_TexCoord[0].xy - slmcentre)*slmsize; // position relative to SLM centre in mm
  vec3 pos = vec3(k*uv/f, -k*dot(uv,uv)/(2.0*f*f));  // scale by k/f and add radial coordinate
  float basephase = zernikeAberration();
  float phase, real=0.0, imag=0.0;
  for(int i; i<2*n; i+=2){
    if(length(uv/slmsize - spots[i+1].xy) < spots[i+1][2]){
      phase = dot(pos, spots[i].xyz) + basephase;
      float amp = spots[i][3];
      real += amp * sin(phase);
      imag += amp * cos(phase);
    }
  }
  
  phase = atan(real, imag);
  float g = apply_LUT(phase);
  gl_FragColor=vec4(g,g,g,1.0);
}
"""

class VeryCleverBeamsplitter(OpenGLShaderWindow):
    """A hologram generation class for interference lithography.
    
    This generates holograms with the following features:
    
    * XYZ+intensity control of multiple beams
    * control of the aperture of each beam on the SLM
    * rotationally-symmetric beam shaping
    * aberration correction
    """
    def __init__(self, **kwargs):
        """Create a hologram generator for interference lithography."""
        super(VeryCleverBeamsplitter, self).__init__(**kwargs)
        self.shader_source = IL_SHADER_SOURCE
        self.centre = [0.5, 0.5]
        self.blazing_function = np.linspace(0,1,32)
        self.zernike_coefficients = np.zeros(12)

    def make_spots(self, spots):
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
        dummy_na_parameters = [0,0,1,0]
        if len(spots[0]) == 4:
            for x in spots:
                x.extend(dummy_na_parameters) #if the spots are missing NA information, add it
    #    for x in spots:
    #        x[3] = I_cal(x[3])
        spots = np.array(spots)
        assert spots.shape[1]==8, "Spots are 8 elements long - your array must be (n,8)"
        self.set_uniform(0, np.reshape(spots,spots.shape[0]*spots.shape[1]))
        self.set_uniform(1, spots.shape[0])
        
    centre = UniformProperty(6, max_length=2)
    blazing_function = UniformProperty(2, max_length=32)
    zernike_coefficients = UniformProperty(3, max_length=12)

    def make_shack_hartmann(self, N, width, x=0, y=0, z=0, ref=False):
        """Make a square-array shack-hartmann sensor.
        
        N: number of spots across the sensor
        width: size of the spot array in the focal plane
        x,y,z: centre of the spot array in the focal plane
        ref: generate a reference array
        """
        us = (np.arange(N)-(N-1.0)/2)/N
        vs = us
        spots = [[u*width+x,v*width+y,z,1,u,v,1.0 if ref else 0.5/N,0] for u in us for v in vs]
        self.make_spots(spots)
 
if __name__ == "__main__":
    slm = VeryCleverBeamsplitter()
    slm.move_hologram(1920,0,1024,768)
    # set uniform values so it has a blazing function and no aberration correction
    blazing_function = np.array([  0,   0,   0,   0,   0,   0,   0,   0,   0,  12,  69,  92, 124,
       139, 155, 171, 177, 194, 203, 212, 225, 234, 247, 255, 255, 255,
       255, 255, 255, 255, 255, 255]).astype(np.float)/255.0 #np.linspace(0,1,32)
    
    slm.blazing_function = blazing_function
    slm.make_spots([[10,-10,0,1],[-10,-10,0,1]])
    slm.make_spots([[10,-10,0,1],[-10,-10,0,1],[2,-15,0,1]])
    slm.make_spots([[0,-10,0,1]])
    slm.make_spots([[-10,0,0,1],[-40,0,0,1]])
    
