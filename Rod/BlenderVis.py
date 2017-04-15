#!blender -P


"""
NOTE: This file could only be run with
blender-python (which uses python3).

blender -b -P BlenderVis.py <arguments>

Or run directly using

./BlenderVis.py <arguments>
"""

import bpy
import os, sys
import mathutils
import numpy as np
from optparse import OptionParser
from math import sin, cos, acos, atan2, pi

scene = bpy.context.scene
scene.render.engine = 'CYCLES'

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

# axis settings
axis_radius = 0.002
axis_length = 0.5
axis_group = None

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

def set_axis_color(ob, name):
    ob.active_material = bpy.data.materials.new(name="AxisMaterial.%s" % name)
    if name == "x":
        ob.active_material.diffuse_color = (1, 0, 0)
    elif name == "y":
        ob.active_material.diffuse_color = (0, 1, 0)
    elif name == "z":
        ob.active_material.diffuse_color = (0, 0, 1)

def create_axis_dim(name):
    # add axis cylinder
    bpy.ops.mesh.primitive_cylinder_add(
        radius=axis_radius,
        depth=axis_length,
        location = (0, 0, 0))
    ob = bpy.context.object
    ob.name = "axis.%s" % name
    ob.show_name = True
    me = ob.data
    me.name = "axis.%s.mesh" % name
    if name == "x":
        bpy.ops.transform.rotate(value=pi/2.0, axis=(0, 1, 0))
        bpy.ops.transform.translate(value=(axis_length/2.0, 0, 0))
    elif name == "y":
        bpy.ops.transform.rotate(value=pi/2.0, axis=(1, 0, 0))
        bpy.ops.transform.translate(value=(0, axis_length/2.0, 0))
    elif name == "z":
        bpy.ops.transform.translate(value=(0, 0, axis_length/2.0))
    set_axis_color(ob, name)
    # add axis cone
    bpy.ops.mesh.primitive_cone_add(
        radius1=axis_radius * 10,
        depth=axis_length * 0.1,
        location = (0, 0, 0))
    ob = bpy.context.object
    ob.name = "axis.%s.cone" % name
    ob.show_name = True
    me = ob.data
    me.name = "axis.%s.cone.mesh" % name
    if name == "x":
        bpy.ops.transform.rotate(value=pi/2.0, axis=(0, 1, 0))
        bpy.ops.transform.translate(value=(axis_length*1.05, 0, 0))
    elif name == "y":
        bpy.ops.transform.rotate(value=-pi/2.0, axis=(1, 0, 0))
        bpy.ops.transform.translate(value=(0, axis_length*1.05, 0))
    elif name == "z":
        bpy.ops.transform.translate(value=(0, 0, axis_length*1.05))
    set_axis_color(ob, name)

def init_axis_group():
    global axis_group
    create_axis_dim("x")
    create_axis_dim("y")
    create_axis_dim("z")
    axis_group = bpy.data.groups.new(name="AxisGroup")
    for axis in [ "x", "y", "z" ]:
        axis_group.objects.link(bpy.data.objects["axis.%s" % axis])
        axis_group.objects.link(bpy.data.objects["axis.%s.cone" % axis])
        scene.objects.unlink(bpy.data.objects["axis.%s" % axis])
        scene.objects.unlink(bpy.data.objects["axis.%s.cone" % axis])

def create_axis(name):
    instance = bpy.data.objects.new("axis.%s" % name, None)
    instance.dupli_type = "GROUP"
    instance.dupli_group = axis_group
    instance.hide = True
    scene.objects.link(instance)
    return instance

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
        # add bishop frames
        self.bishops = [ create_axis(i) for i in range(knots) ]
        for i in range(0, knots, int(knots/4)):
            self.bishops[i].hide = False

    def _compute_initial_bishop(self, pt1, pt2, theta):
        axis1 = (pt2 - pt1).normalized()
        axis2 = mathutils.Vector([-axis1.y, axis1.x, 0]).normalized()
        axis3 = axis1.cross(axis2)
        bishop = [ axis1, axis2, axis3 ]
        return bishop

    def _compute_next_bishop(self, bishop, pt0, pt1, pt2, theta):
        segment1_dir = (pt1 - pt0).normalized()
        segment2_dir = (pt2 - pt1).normalized()
        axis = segment1_dir.cross(segment2_dir)
        angle = acos(segment1_dir.dot(segment2_dir))
        q_bend = mathutils.Quaternion(axis, angle)
        next_bishop = [ q_bend * d for d in bishop ]
        q_twist = mathutils.Quaternion(segment1_dir, theta)
        next_bishop[1] = q_twist * next_bishop[1]
        next_bishop[2] = q_twist * next_bishop[2]
        return next_bishop

    def _compute_euler_angle_from_bishop(self, bishop):
        e1, e2, e3 = bishop
        angle1 = atan2(e1.y, e1.x)
        angle2 = atan2(e1.z, e1.x)
        mat = mathutils.Matrix()
        mat[0][0:3] = e1
        mat[1][0:3] = e2
        mat[2][0:3] = e3
        mat.invert()
        return mat.to_euler()

    def select(self):
        scene.objects.active = self.centerline
        self.centerline.select = True

    def update(self, xs, thetas):
        # compute bishop frame
        vector_xs = list(map(lambda x : mathutils.Vector(x), xs))
        bishops = [ [] ] * (len(vector_xs))
        bishops[0] = self._compute_initial_bishop(vector_xs[0], vector_xs[1], thetas[0])
        for i in range(1, len(vector_xs)-1):
            bishops[i] = self._compute_next_bishop(bishops[i-1],
                vector_xs[i-1], vector_xs[i], vector_xs[i+1], thetas[i])
        bishops[len(vector_xs) - 1] = bishops[len(vector_xs) - 2]

        # update inter control points
        for i, (x, y, z) in enumerate(xs):
            pt = self.polyline.points[i]
            pt.co = (x, y, z, 1.0)
            pt.radius = self.radius
            pt.tilt = thetas[i] + i * self.rod_tilt

        # add keyframes for control points
        for point in self.polyline.points:
            point.keyframe_insert('co', frame=keyframe)
            point.keyframe_insert('tilt', frame=keyframe)

        # update bishop frame
        for i, (x, y, z) in enumerate(xs):
            self.bishops[i].location = mathutils.Vector([x, y, z])
            self.bishops[i].rotation_euler = self._compute_euler_angle_from_bishop(bishops[i])

        # add keyframe for bishop frame
        for i, (x, y, z) in enumerate(xs):
            self.bishops[i].keyframe_insert('location', frame=keyframe)
            self.bishops[i].keyframe_insert('rotation_euler', frame=keyframe)

def init_scene():
    delete_object_by_name("Cube")
    create_bevel_circle()
    init_axis_group()
    bpy.data.objects["BezierCircle"].hide = True

def run_sim(data):
    n_timesteps, n_rods, n_centerpoints, n_dim = data.shape
    radius = 0.1

    # init rods
    rods = [ RodState("rod-%d" % i, n_centerpoints, radius, "rod") for i in range(n_rods) ]
    for t in range(n_timesteps):
        time_slice = data[t,:,:,:]
        for i, rod in enumerate(rods):
            xs = time_slice[i,:,0:3]
            ts = time_slice[i,:,3]
            rod.update(xs, ts)
        tick()
    scene.frame_set(0)

def save_blend(filename):
    bpy.ops.wm.save_as_mainfile(filepath=filename)

def fake_data():
    intervals = np.arange(0, 6.28, 0.1)
    # intervals = np.arange(0, 6.28, 1)
    knots = len(intervals)
    n_timesteps = 100
    n_rods = 1
    data = np.zeros(shape=[n_timesteps, n_rods, knots, 4], dtype=float)
    for t in range(n_timesteps):
        data[t,0,:,:] = [ [ sin(x + simtime), x, x / 2.0, 0.0 ] for x in intervals ]
        tick()
    global simtime
    simtime = 0
    return data

def option_parser():
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename",
            help="write report to FILE", metavar="FILE")
    parser.add_option("", "--bishop-frame", dest="bishop",
            help="render bishop frame", default=True)

    index = 0
    for i, arg in enumerate(sys.argv):
        if "BlenderVis.py" in arg:
            index = i
            break
    (options, args) = parser.parse_args(sys.argv[index:])
    return options

if __name__ == "__main__":
    init_scene()
    options = option_parser()

    if options.filename is None:
        data = fake_data()
    else:
        filename = sys.argv[1]
        data = np.fromfile(filename, dtype=float)

    run_sim(data)
