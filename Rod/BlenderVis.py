#!blender -P


"""
NOTE: This file could only be run with
blender-python (which uses python3).

blender -b -P BlenderVis.py <arguments>

Or run directly using

./BlenderVis.py <arguments>
"""

import bpy
import json
import mathutils

scene = bpy.context.scene

def delete_object_by_name(name):
    if name in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[name].select = True
        bpy.ops.object.delete()

class RodState(object):

    def __init__(self, name, xs, thetas):
        super(RodState, self).__init__()
        self.name = name
        self.xs = xs
        self.thetas = thetas
        self._init_objects()

    def _init_objects(self):
        curve_data = bpy.data.curves.new("%s.centerline" % self.name, type='CURVE')
        polyline = curve_data.splines.new("NURBS")
        polyline.points.add(len(self.xs))
        for i, (x, y, z) in enumerate(self.xs):
            polyline.points[i].co = (x, y, z, 1.0)
        self.centerline = bpy.data.objects.new("%s.centerline" % self.name, curve_data)
        scene.objects.link(self.centerline)

    def select(self):
        scene.objects.active = self.centerline
        self.centerline.select = True

def init_scene():
    delete_object_by_name("Cube")

def init_rods():
    xs = [ [-1.0, 1.0, 0.0], [ 0.0, 0.0, 0.0], [ 1.0, 1.0, 0.0], [ 2.0, 0.0, 0.0] ]
    thetas = [ 0.0, 0.0, 0.0 ]
    rod = RodState("rod", xs, thetas)

if __name__ == "__main__":
    init_scene()
    init_rods()
