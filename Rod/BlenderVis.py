#!blender -P


"""
NOTE: This file could only be run with
blender-python (which uses python3).

blender -b -P BlenderVis.py <arguments>

Or run directly using

./BlenderVis.py <arguments>
"""

import bpy
import os, json
import mathutils
import numpy as np
from math import sin, cos

scene = bpy.context.scene

# simulation settings
simtime = 0
timestep = 0.1
fps = scene.render.fps
keyframe = 0

# render settings
checkerboard_tex = bpy.data.textures.new("Checkerboard", type="IMAGE")
checkerboard_tex.image = bpy.data.images.load(os.path.abspath("./textures/checkerboard.jpg"))
rod_material = bpy.data.materials.new(name="RodMaterial")
tex_slot = rod_material.texture_slots.add()
tex_slot.texture = checkerboard_tex

def tick():
    global simtime, keyframe
    simtime += timestep
    keyframe = int(simtime * fps)
    scene.frame_set(keyframe)

def delete_object_by_name(name):
    if name in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[name].select = True
        bpy.ops.object.delete()

def create_bevel_circle():
    bpy.ops.curve.primitive_bezier_circle_add()

class RodState(object):

    def __init__(self, name, knots, radius, style="rod"):
        super(RodState, self).__init__()
        self.name = name
        self.radius = radius
        self.rod_tilt = 0
        self.style = style
        self._init_objects(knots)

    def _init_objects(self, knots):
        # generate NURBS curve
        curve_data = bpy.data.curves.new("%s.centerline" % self.name, type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.bevel_object = bpy.data.objects["BezierCircle"]
        # curve type
        self.polyline = curve_data.splines.new("NURBS")
        self.polyline.points.add(knots - len(self.polyline.points))
        self.polyline.use_endpoint_u = True     # connect start and stop points
        # add curve to scene
        self.centerline = bpy.data.objects.new("%s.centerline" % self.name, curve_data)
        self.centerline.data.materials.append(rod_material)
        scene.objects.link(self.centerline)

    def select(self):
        scene.objects.active = self.centerline
        self.centerline.select = True

    def update(self, xs, thetas):
        # add inter control points
        for i, (x, y, z) in enumerate(xs):
            pt = self.polyline.points[i]
            pt.co = (x, y, z, 1.0)
            pt.radius = self.radius
            pt.tilt = thetas[i] + i * self.rod_tilt

        # add keyframes
        for point in self.polyline.points:
            point.keyframe_insert('co', frame=keyframe)
            point.keyframe_insert('tilt', frame=keyframe)

def init_scene():
    delete_object_by_name("Cube")

def init_rods():
    pass

def save_blend(filename):
    bpy.ops.wm.save_as_mainfile(filepath=filename)

def test_sim():
    radius = 0.1
    intervals = np.arange(0, 6.28, 0.01)
    knots = len(intervals)
    rod = RodState("rod", knots, radius, "rod")

    for i in range(100):
        xs = [ [ sin(x + simtime), x, 0.0 ] for x in intervals ]
        thetas = [ 0 ] * len(intervals)
        rod.update(xs, thetas)
        if i != 100: tick()
    scene.frame_set(0)

if __name__ == "__main__":
    create_bevel_circle()
    init_scene()
    init_rods()
    test_sim()
