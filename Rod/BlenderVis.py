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
        for i, (x, y, z) in enumerate(xs):
            polyline.points[i].co = (x, y, z, 1.0)
        self.centerline = bpy.data.objects.new("%s.centerline" % self.name, curve_data)
        # add to scene
        scene.objects.link(self.centerline)
        scene.objects.active = self.centerline
        self.centerline.select = True


for ob in bpy.data.objects:
    print (ob.name)
    try:
        ob.material_slot_remove()
        print ("removed material from " + ob.name)
    except:
        print (ob.name + " does not have materials.")

if __name__ == "__main__":
    print(scene)
    print(bpy.data.objects)
    xs = [ [-1.0, 1.0, 0.0], [ 0.0, 0.0, 0.0], [ 1.0, 1.0, 0.0] ]
    thetas = [ 0.0, 0.0, 0.0 ]
    rod = RodState("rod", xs, thetas)
