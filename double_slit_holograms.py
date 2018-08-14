# -*- coding: utf-8 -*-
"""
Created on Mon Dec 01 12:04:12 2014

@author: Richard
"""

import numpy as np
from opengl_holograms import OpenGLShaderWindow, UniformProperty


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


 
if __name__ == "__main__":
    slm = SlitHologram()
    slm.move_hologram(0,0,512,512)
    slm.n = 0
    slm.unit_cell = [1.0/256 for i in range(2)]
    slm.rects = [0,0,1,0.5,   0,0.5,1,1]
    slm.rects = [0,0.2,1,0.3,   0,0.7,1,0.8]
    slm.colours = [0.5,0,0,1,   0,0.5,0.5,1]
    slm.n = 2
    
