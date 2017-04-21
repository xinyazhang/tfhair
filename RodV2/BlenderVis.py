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
from queue import Queue
from optparse import OptionParser
from math import sin, cos, acos, atan2, pi

sys.path.append(os.path.abspath("./"))
import BlenderUtil

scene = bpy.context.scene
scene.render.engine = 'CYCLES'

# simulation settings
timestep = 0.1
fps = scene.render.fps
keyframe = 0

# display settings
axis_radius = 0.002
axis_length = 0.5
axis_group = None

# render settings
rod_material = None

rods = None
data = Queue()      # queue for storing keyframe data, i.e. rod states

def compute_keyframe(frame):
    simtime = timestep * frame
    return int(simtime * fps)

def delete_object_by_name(name):
    if name in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects[name].select = True
        bpy.ops.object.delete()

def init_bevel_circle():
    bpy.ops.curve.primitive_bezier_circle_add()

def init_viewport_shading():
    for area in bpy.context.screen.areas: # iterate through areas in current screen
        if area.type == 'VIEW_3D':
            for space in area.spaces: # iterate through spaces in current VIEW_3D area
                if space.type == 'VIEW_3D': # check if space is a 3D view
                    space.viewport_shade = 'MATERIAL' # set the viewport shading to rendered

def init_texture_material(mat_name, tex_name, tex_file):
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    texture = nodes.new("ShaderNodeTexImage")
    texcoord = nodes.new("ShaderNodeGeometry")

    texture.image = bpy.data.images.load(os.path.abspath(tex_file))
    texture.projection = "TUBE"

    links.new(texture.inputs['Vector'], texcoord.outputs['Normal'])
    links.new(diffuse.inputs['Color'], texture.outputs['Color'])
    links.new(output.inputs['Surface'], diffuse.outputs['BSDF'])
    global rod_material
    rod_material = mat

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

def create_frame(name):
    instance = bpy.data.objects.new("axis.%s" % name, None)
    instance.show_name = True
    instance.dupli_type = "GROUP"
    instance.dupli_group = axis_group
    instance.hide = True
    instance.empty_draw_size = 0.0
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
        self.bishops = [ create_frame(i) for i in range(knots) ]
        n_dis_step = 1 if knots < 10 else int(knots / 10)
        for i in range(0, knots, n_dis_step):
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
        dot = max(-1.0, min(1.0, segment1_dir.dot(segment2_dir)))
        angle = acos(dot)
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
    init_viewport_shading()
    init_bevel_circle()
    init_axis_group()
    init_texture_material("RodMaterial", "Checkerboard", "./textures/checkerboard.jpg")
    bpy.data.objects["BezierCircle"].hide = True

def run_sim(data):
    n_rods, n_centerpoints, n_dim = data.shape
    radius = 0.02

    # init rods
    global rods
    if rods is None:
        rods = [ RodState("rod-%d" % i, n_centerpoints, radius, "rod") for i in range(n_rods) ]

    # update rods
    for i, rod in enumerate(rods):
        xs = data[i,:,0:3]
        ts = data[i,:,3]
        rod.update(xs, ts)

def save_blend(filename):
    bpy.ops.wm.save_as_mainfile(filepath=filename)

def callback_finish(message):
    receiver.stop()
    scene.frame_set(0)

def callback_frame(message):
    global keyframe
    frame = int(message)
    keyframe = compute_keyframe(frame)

def callback_update(message):
    filename = message.strip()
    data.put((keyframe, filename))

callbacks = {
    "finish": callback_finish,
    "frame": callback_frame,
    "update": callback_update
}

def callback_load_keyframes(scene):
    global keyframe
    while not data.empty():
        keyframe, filename = data.get()
        run_sim(np.load(filename))
        data.task_done()
        # print("load file %s at keyframe %d" % (filename, keyframe))

def option_parser():
    parser = OptionParser()
    parser.add_option("", "--simdir", dest="simdir",
            help="write report to FILE")

    index = 0
    for i, arg in enumerate(sys.argv):
        if "BlenderVis.py" in arg:
            index = i
            break
    (options, args) = parser.parse_args(sys.argv[index:])
    return options

if __name__ == "__main__":
    init_scene()
    bpy.app.handlers.frame_change_post.append(callback_load_keyframes)

    options = option_parser()
    if options.simdir is None:
        receiver = BlenderUtil.Receiver(callbacks)
        receiver.receive()
    else:
        global data
        path = os.path.abspath(options.simdir)
        for name in os.listdir(path):
            frame = int(name.strip(".npy"))
            keyframe = compute_keyframe(frame)
            filepath = os.path.join(path, name)
            data.put((keyframe, filepath))
        scene.frame_set(0)
