#!/usr/bin/blender -P


"""
NOTE: This file could only be run with
blender-python (which uses python3).

blender -P RodVis.py -- <simdir1> <simdir2> ...

Or run directly using

./RodVis.py -- <simdir1> <simdir2> ...
"""

import bpy
import os, sys
import optparse
from bpy.app.handlers import persistent

script_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_path)
from BlenderScene import *
from BlenderObject import *

scene_dict = dict()

def ParseArgs():
    parser = optparse.OptionParser()
    parser.add_option("", "--fps", dest="fps",
            default=1000.0, help="set fps for rendering")
    script = os.path.basename(__file__)
    index1 = [i for i, arg in enumerate(sys.argv) if script in arg]
    index2 = [i for i, arg in enumerate(sys.argv) if arg == "--"]
    index = max(index1 + index2)
    return parser.parse_args(sys.argv[index+1:])

@persistent
def LoadFrameCallback(scn):
    if scn not in scene_dict:
        return
    scene = scene_dict[scn]
    path = scene.cache_path
    frame = scn.frame_current

    filepath = os.path.join(path, "%d.mat" % (frame-1))
    if os.path.exists(filepath):
        scene.Load(frame, filepath)

def LoadRodScene(options, number, path):
    scene = RodScene(options.fps, scene=number)
    scene.SetCachePath(path)
    for name in filter(lambda x: ".mat" in x, os.listdir(path)):
        frame = int(name.strip(".mat"))
        filepath = os.path.join(path, name)
        scene.Load(frame, filepath)
    scene.SetFrame(0)
    # scene_dict[scene.scene] = scene
    # bpy.app.handlers.frame_change_pre.append(LoadFrameCallback)
    return scene

def LoadBlend(filename):
    bpy.ops.wm.open_mainfile(filepath=filename)

def SaveBlend(filename):
    bpy.ops.wm.save_as_mainfile(filepath=filename)

if __name__ == "__main__":
    options, args = ParseArgs()
    scenes = [ LoadRodScene(options, i, path) for i, path in enumerate(args) ]
    scenes[0].Activate()
