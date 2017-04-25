#!/usr/bin/blender -P


"""
NOTE: This file could only be run with
blender-python (which uses python3).

blender -P BlenderVis.py -- <simdir1> <simdir2> ...

Or run directly using

./BlenderVis.py -- <simdir1> <simdir2> ...
"""

import bpy
import os, sys
import optparse

script_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_path)
from BlenderScene import *
from BlenderObject import *

def ParseArgs():
    parser = optparse.OptionParser()
    parser.add_option("", "--fps", dest="fps",
            default=1000.0, help="set fps for rendering")
    script = os.path.basename(__file__)
    index = [i for i, arg in enumerate(sys.argv) if script in arg][0]
    return parser.parse_args(sys.argv[index+1:])

def LoadRodScene(options, number, path):
    scene = RodScene(options.fps, scene=number)
    for name in filter(lambda x: ".mat" in x, os.listdir(path)):
        frame = int(name.strip(".mat"))
        filepath = os.path.join(path, name)
        scene.Load(frame, filepath)
    scene.SetFrame(0)
    return scene

def SaveBlend(filename):
    bpy.ops.wm.save_as_mainfile(filepath=filename)

if __name__ == "__main__":
    options, args = ParseArgs()
    scenes = [ LoadRodScene(options, i, path) for i, path in enumerate(args) ]
    scenes[0].Activate()
