# -*- coding: utf-8 -*-
"""
Created on Mon Dec 01 12:04:12 2014

@author: Richard
"""

import numpy as np
from opengl_holograms import OpenGLShaderWindow, UniformProperty

RADIAL_ARRAY_LENGTH = 576

# This is a big string constant that actually renders the hologram.  See below
# for the useful Python code that you might want to use...
IL_SHADER_SOURCE = """
uniform vec4 spots[100];
uniform int n;
uniform float blazing[32];
uniform float zernikeCoefficients[12];
uniform float radialPhase["""+str(RADIAL_ARRAY_LENGTH)+"""];
uniform float radialPhaseDr = 0.009 * 2.0;
uniform float radialBlaze["""+str(RADIAL_ARRAY_LENGTH)+"""];
uniform float k=15700; //units of mm-1
uniform float f=3500.0; //mm
uniform vec2 slmsize=vec2(17.6,10.7); //size of SLM prev 6.9
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
  return mix(radialPhase[index], radialPhase[index+1], alpha);
}
float radialBlazeFunction(vec2 uv){
  //calculate a radially-symmetric phase function from the uniform radialPhase
  float r = sqrt(dot(uv,uv));
  int index = int(floor(r/radialPhaseDr));
  float alpha = fract(r/radialPhaseDr);
  return mix(radialBlaze[index], radialBlaze[index+1], alpha);
}

void main(){ // basic gratings and lenses for a single spot
  vec2 uv = (gl_TexCoord[0].xy - slmcentre)*slmsize;
  vec3 pos = vec3(k*uv/f, -k*dot(uv,uv)/(2.0*f*f));
  float basephase = zernikeAberration() + radialPhaseFunction(uv);
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
  g = (g-0.5) * radialBlazeFunction(uv) + 0.5;
  gl_FragColor=vec4(g,g,g,1.0);
  //gl_FragColor=vec4(1,1,0,1);
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
        #self.centre = [0.5, 0.5] #prev value
        self.active_area = [17.6, 10.7]
        #self.active_area = [6.9, 6.9]
        self.blazing_function = np.linspace(0,1,32)
        self.zernike_coefficients = np.zeros(12)
        self.radial_phase_dr = 0.009*2
        self.wavevector = 15700 # 2pi/wavelength with wavelength in mm
        self.focal_length = 3500
        self.disable_gaussian_to_tophat()
        self.radial_blaze_function = np.ones(RADIAL_ARRAY_LENGTH)
        self.input_beam_cumulative_I = np.linspace(0, 1, RADIAL_ARRAY_LENGTH-1)

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
        
    blazing_function = UniformProperty(2, max_length=32)
    zernike_coefficients = UniformProperty(3, max_length=12)
    radial_phase_function = UniformProperty(4, max_length=RADIAL_ARRAY_LENGTH)
    radial_phase_dr = UniformProperty(5, max_length=1)
    radial_blaze_function = UniformProperty(6, max_length=RADIAL_ARRAY_LENGTH)
    wavevector = UniformProperty(7, max_length=1) #k
    focal_length = UniformProperty(8, max_length=1) #f
    active_area = UniformProperty(9, max_length=2)
    centre = UniformProperty(10, max_length=2)
    

    @property
    def input_beam_r(self):
        """A set of radial coordinates against which we define the input beam.
        """
        return (np.arange(RADIAL_ARRAY_LENGTH-1) + 0.5) * self.radial_phase_dr

    def set_gaussian_input_beam(self, waist):
        """Use the cumulative intensity profile of a Gaussian beam
    
        Set self.input_beam_cumulative_I to be the fraction of a 2D Gaussian 
        contained within a given radius.  The radial coordinates are set
        by self.radial_phase_dr
            
        The equation of the underlying gaussian is np.exp(-r**2/(w**2))
        remember d/dx exp(-r^2/w^2) = exp(-r^2/w^2) * (-2r/w^2)
        so int[0,R] e**(-r**2/w**2) r dr
        = (2r/w^2) * [1 - exp(-r^2/w^2)]
        The normalisation takes care of the 2r/w^2 term.
        
        NB the radial coordinate here has **one fewer** element than the
        eventual radial phase function/radial blazing function, because it gets
        numerically integrated.  The first point here is r=0.5*radial_phase_dr
        """
        r = self.input_beam_r
        self.input_beam_cumulative_I = 1 - np.exp(-r**2/(waist**2)) #NB there's no 2 as it's *intensity*
        #    r, I, smoothI, dummyval/r = measure_intensity(100,1, 360, darkfieldimage)
            
        #    return I
        
    def reshape_to_tophat(self, target_radius, distance=None):
        """Calculate a radial phase function to re-map from gaussian to top-hat.
        
        target_radius: size of target top-hat beam
        distance: distance from SLM to target plane (defaults to self.focal_length)
        
        Units should be self-consistent, probably mm.
        """
        if distance is None:
            distance =  self.focal_length
        N = RADIAL_ARRAY_LENGTH
        dr = (self.radial_phase_dr)
        k = float(self.wavevector)
        # Next, calculate the phase shift as a function of radius
        def cumulative_I_tophat(r, r_beam):
            """Fraction of a top-hat contained within a given radius."""
            if r<r_beam:
                return (r/r_beam)**2
            else:
                return 1
        def inverse_cumulative_I_tophat(I_cum, r_beam):
            """Radius at which a given fraction of the beam is enclosed.
            
            NB you might expect to numerically invert cumulative_I_tophat, and
            that's why that function is defined.  But actually it's easier to
            do analytically, which is what we do here.
            
            R_beam is the radius of the top-hat beam.
            """
            return np.sqrt(I_cum) * r_beam
        
        phase_shift = np.zeros((N)) #array the right length
        r = self.input_beam_r
        target_r = inverse_cumulative_I_tophat(
                            self.input_beam_cumulative_I,
                            target_radius)
        tilt = (target_r-r)/distance
        for i in range(1,len(phase_shift)):
            #this is just a lens
            phase_shift[i] = phase_shift[i-1] + tilt[i-1] * k * dr
        self.radial_phase_function = phase_shift
        
    def reshape_gaussian_to_tophat(self, initial_r, final_r, distance=None):
        """Update the parameters used to map gaussian -> top hat"""
        self.set_gaussian_input_beam(initial_r)
        self.reshape_to_tophat(final_r, distance=distance)

    def disable_gaussian_to_tophat(self):
        """Set the radial phase function to zero"""
        #self.set_uniform(4, np.zeros((384,)))
        self.radial_phase_function = np.zeros((RADIAL_ARRAY_LENGTH,))
    
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
    slm.move_hologram(1920,0,1920,1152)
    # set uniform values so it has a blazing function and no aberration correction
    blazing_function = np.array([  0,   0,   0,   0,   0,   0,   0,   0,   0,  12,  69,  92, 124,
       139, 155, 171, 177, 194, 203, 212, 225, 234, 247, 255, 255, 255,
       255, 255, 255, 255, 255, 255]).astype(np.float)/255.0 #np.linspace(0,1,32)
    
    slm.blazing_function = blazing_function
    slm.update_gaussian_to_tophat(1900*3/2,3000, distance=1900e3)
    slm.zernike_coefficients = np.zeros(12)
    slm.wavevector = 2*np.pi*1e6/633.0
    slm.focal_length = 3500e3 #3500e3
    slm.centre = [0.5, 0.5]
    slm.make_spots([[20,-10,0,1],[-20,-10,0,1]])
   