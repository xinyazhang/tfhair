#!/usr/bin/blender -P


"""
NOTE: This file could only be run with
blender-python (which uses python3).

blender -P HairVis.py -- <simdir>

Or run directly using

./HairVis.py -- <simdir>
"""

import bpy
import os, sys
import optparse
from bpy.app.handlers import persistent

script_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_path)
from BlenderScene import *
from BlenderObject import *

scene = None

def ParseArgs():
    parser = optparse.OptionParser()
    parser.add_option("", "--fps", dest="fps", default=1000.0, type=float, help="set fps for rendering")
    parser.add_option("", "--dump", dest="dump", help="set dump file")
    parser.add_option("", "--template", dest="template", default="SimpleHair.blend", help="set template file")
    script = os.path.basename(__file__)
    index1 = [i for i, arg in enumerate(sys.argv) if script in arg]
    index2 = [i for i, arg in enumerate(sys.argv) if arg == "--"]
    index = max(index1 + index2)
    return parser.parse_args(sys.argv[index+1:])

def LoadBlend(filename):
    bpy.ops.wm.open_mainfile(filepath=filename)

def SaveBlend(filename):
    bpy.ops.wm.save_as_mainfile(filepath=filename)

@persistent
def LoadFrameCallback(scn):
    path = scene.cache_path
    frame = scn.frame_current

    filepath = os.path.join(path, "%d.mat" % (frame-1))
    if os.path.exists(filepath):
        scene.Load(frame, filepath)

def Setup(options, args):
    global scene
    scene = HairScene(options.fps)
    if options.dump is not None:
        scene.Dump(options.dump)
    else:
        scene.SetCachePath(args[0])
        if len(args) > 1:
            scene.SetMetadata(args[1])
        bpy.app.handlers.frame_change_pre.append(LoadFrameCallback)

        path = scene.cache_path
        names = filter(lambda x: ".mat" in x, os.listdir(path))
        frames = map(lambda x: int(x.strip(".mat")), names)
        end_frame = max(frames) + 1
        scene.SetEndFrame(end_frame)

@persistent
def LoadBlendCallback(scn):
    """Fix bpy.context if some command (like .blend import) changed/emptied it"""
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        override = {'window': window, 'screen': screen, 'area': area, 'region': region}
                        bpy.ops.screen.screen_full_area(override)
                        break

    # parse args and options
    options, args = ParseArgs()
    Setup(options, args)

if __name__ == "__main__":
    # set up callback function for loading blend files
    bpy.app.handlers.load_post.append(LoadBlendCallback)

    # init hair scene with head and scalp
    options, _ = ParseArgs()
    LoadBlend(os.path.join(script_path, "templates", options.template))
