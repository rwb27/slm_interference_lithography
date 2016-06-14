# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:39:16 2016

@author: Richard Bowman

This is a Python wrapper for the Red Tweezers hologram engine to make it a bit
more "Pythonic".

See the documentation for `OpenGLShaderWindow` for more information.

"""

import socket

#from nplab.instrument import Instrument
class UniformProperty(object):
    def __init__(self, uniform_id, max_length=-1):
        """Create a property-like object to set the value of a uniform variable
        
        Arguments:
        uniform_id : int
            The number, starting from zero, of the uniform to set.  This is the
            order they are defined in the shader source.  In the future, it may
            be possible to look them up by name.
        max_length : int (optional)
            If supplied, truncate the number of values to be no more than a
            set number.  If -1 or 0, ignore it.
        """
        super(UniformProperty, self).__init__()
        self.__doc__ = """Set the uniform variable's value"""
        self.uniform_id = uniform_id
        self.max_length = max_length
        
    def __set__(self, obj, value):
        try:
            if len(value) > self.max_length:
                value = value[0:self.mmax_length]
        except TypeError:
            #if the object wasn't iterable we end up here...
            value = [value]
        obj.set_uniform(self.uniform_id, value)

class OpenGLShaderWindow(object):
    """Python control interface to the Red Tweezers hologram engine.
    
    The hologram engine is a small program written in C that renders patterns
    to the screen based on OpenGL Shader Language.  It accepts input (including
    the shader program) through a UDP socket, meaning you can completely
    redefine what's going on in the GPU from the comfort of Python.
    
    For documentation about Red Tweezers, the suite of programs from which
    the hologram engine here is taken, please see the paper here:
    http://dx.doi.org/10.1016/j.cpc.2013.08.008
    
    # Subclassing Notes
    It's encouraged to subclass this class for whatever sort of hologram you
    would like to create, and use `UniformProperty` attributes to make a more
    friendly interface to the shader's parameters.  You probably want to set
    the shader source in the `__init__` method.
    """
    def __init__(self, host="127.0.0.1", port=61557):
        """Construct an object to control a GLSL window (for an SLM)
        
        Arguments:
        host : string
            The IP address or hostname of the machine where the SLM is 
            connected (usually the default of 127.0.0.1, which corresponds
            to this computer, is correct)
        port: int
            The UDP port to send to.  This can usually be left at the default
            of 61557, except when you are using multiple SLMs on one PC.
        """
        super(OpenGLShaderWindow, self).__init__()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)
        self.host = host
        self.port = port
        
        # We want to make sure we always get a reply
        self.query("<data>\n<network_reply>1</network_reply>\n</data>\n")
        
    def query(self, message):
        """Send a string to the engine, and await a response."""
        self.sock.sendto(message,(self.host, self.port))
        return self.sock.recvfrom(128)
        
    def set_shader_source(self, source):
        """Set the fragment shader that renderes the pattern."""
        # TODO: automatically set up attributes?
        self.query("<data>\n<shader_source>\n{0}\n</shader_source>\n</data>\n".format(source))
    shader_source = property(fset=set_shader_source)
    
    def move_hologram(self, x=0, y=0, w=1024, h=768):
        """Move the hologram on the screen."""
        self.query("<data>\n"
                   "<window_rect>{0},{1},{2},{3}</window_rect>\n"
                   "</data>\n".format(x, y, w, h))
                   
    def set_uniform(self, uniform_id, value):
        """Set the value of a uniform variable.
        
        uniform_id : int
            The number (starting from 0) of the uniform to set
        value : list of float
            The value to give the uniform variable.
        """
        try: #TODO: nicer way of testing it's iterable
            value[0]
        except:
            value = [value]
        self.query("<data>\n<uniform id={0}>\n".format(uniform_id) +
            " ".join(map(lambda x: "%f" % x, value)) +
            "</uniform>\n</data>\n")
            
    