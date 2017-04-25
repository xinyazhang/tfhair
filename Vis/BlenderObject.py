import bpy
import mathutils
import numpy as np

from math import sin, cos, acos, atan2, pi

def CreateBevelCircle():
    if "BezierCircle" not in bpy.data.objects:
        bpy.ops.curve.primitive_bezier_circle_add()
        bpy.data.objects["BezierCircle"].hide = True
    return bpy.data.objects["BezierCircle"]

class AxisFrame(object):

    def __init__(self, name, radius=0.002, length=0.5):
        self.name = name
        self.radius = radius
        self.length = length
        self._InitObject()
        self._InitGroup("%sGroup" % name)

    def _InitObject(self):
        cylinder_x, cone_x = self._CreateAxisArrow("x")
        cylinder_y, cone_y = self._CreateAxisArrow("y")
        cylinder_z, cone_z = self._CreateAxisArrow("z")
        self.obj_cylinders = [ cylinder_x, cylinder_y, cylinder_z ]
        self.obj_cones = [ cone_x, cone_y, cone_z ]

    def _InitGroup(self, name):
        self.axis_group = bpy.data.groups.new(name=name)
        for cylinder, cone in zip(self.obj_cylinders, self.obj_cones):
            self.axis_group.objects.link(cylinder)
            self.axis_group.objects.link(cone)
            bpy.context.scene.objects.unlink(cylinder)
            bpy.context.scene.objects.unlink(cone)

    def CreateInstance(self, name):
        instance = bpy.data.objects.new("axis.%s" % name, None)
        instance.show_name = True
        instance.dupli_type = "GROUP"
        instance.dupli_group = self.axis_group
        instance.hide = True
        instance.empty_draw_size = 0.0
        bpy.context.scene.objects.link(instance)
        return instance

    def _CreateAxisArrow(self, name):
        return self._CreateAxisCylinder(name), self._CreateAxisCone(name)

    def _CreateAxisCylinder(self, name):
        # add axis cylinder
        bpy.ops.mesh.primitive_cylinder_add(
            radius=self.radius, depth=self.length, location = (0, 0, 0))
        ob = bpy.context.object
        ob.name = "axis.%s.cylinder" % name
        me = ob.data
        me.name = "axis.%s.cylinder.mesh" % name
        if name == "x":
            bpy.ops.transform.rotate(value=pi/2.0, axis=(0, 1, 0))
            bpy.ops.transform.translate(value=(self.length/2.0, 0, 0))
        elif name == "y":
            bpy.ops.transform.rotate(value=pi/2.0, axis=(1, 0, 0))
            bpy.ops.transform.translate(value=(0, self.length/2.0, 0))
        elif name == "z":
            bpy.ops.transform.translate(value=(0, 0, self.length/2.0))
        self._SetAxisColor(ob, name)
        return ob

    def _CreateAxisCone(self, name):
        # add axis cone
        bpy.ops.mesh.primitive_cone_add(
            radius1=self.radius * 10, depth=self.length * 0.1, location = (0, 0, 0))
        ob = bpy.context.object
        ob.name = "axis.%s.cone" % name
        me = ob.data
        me.name = "axis.%s.cone.mesh" % name
        if name == "x":
            bpy.ops.transform.rotate(value=pi/2.0, axis=(0, 1, 0))
            bpy.ops.transform.translate(value=(self.length*1.05, 0, 0))
        elif name == "y":
            bpy.ops.transform.rotate(value=-pi/2.0, axis=(1, 0, 0))
            bpy.ops.transform.translate(value=(0, self.length*1.05, 0))
        elif name == "z":
            bpy.ops.transform.translate(value=(0, 0, self.length*1.05))
        self._SetAxisColor(ob, name)
        return ob

    def _SetAxisColor(self, obj, name):
        obj.active_material = bpy.data.materials.new(name="AxisMaterial.%s" % name)
        colors = { "x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1) }
        obj.active_material.diffuse_color = colors[name]

class Rod(object):

    axis = AxisFrame("Axis")

    def __init__(self, name, knots, radius=0.02):
        self.name = name
        self.knots = knots
        self.radius = radius
        self._InitObject()

    def _InitObject(self):
        knots = self.knots
        # generate curve
        curve_data = bpy.data.curves.new("%s.centerline.curve" % self.name, type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.bevel_object = CreateBevelCircle()
        curve_data.use_fill_caps = True
        # curve type
        self.polyline = curve_data.splines.new("POLY")
        self.polyline.points.add(knots - len(self.polyline.points))
        self.polyline.use_endpoint_u = True     # connect start and stop points
        # add curve to scene
        self.centerline = bpy.data.objects.new("%s.centerline.object" % self.name, curve_data)
        # self.centerline.data.materials.append(rod_material)
        bpy.context.scene.objects.link(self.centerline)
        # add subdivision modifier
        self.centerline.modifiers.new(name="%s.modifier" % self.name, type="SUBSURF")
        mod = self.centerline.modifiers[0]
        mod.levels = 4
        mod.render_levels = 4
        # add material_frame frames
        self.material_frames = [ Rod.axis.CreateInstance(i) for i in range(knots-1) ]
        n_dis_step = 1 if knots < 10 else int(knots / 10)
        for i in range(0, knots-1, n_dis_step):
            self.material_frames[i].hide = False

    def _ComputeMaterialFrame(self, theta, pt0, pt1, refd1, refd2):
        axis1 = (pt1 - pt0).normalized()
        q_twist = mathutils.Quaternion(axis1, theta)
        axis2 = q_twist * mathutils.Vector(refd1)
        axis3 = q_twist * mathutils.Vector(refd2)
        return  [ axis1, axis2, axis3 ]

    def _ComputeEulerAngleFromFrame(self, material_frame):
        e1, e2, e3 = material_frame
        mat = mathutils.Matrix()
        mat[0][0:3] = e1
        mat[1][0:3] = e2
        mat[2][0:3] = e3
        mat.invert()
        return mat.to_euler()

    # def _ComputeInitialMaterialFrame(self, pt1, pt2, theta):
    #     axis1 = (pt2 - pt1).normalized()
    #     axis2 = mathutils.Vector([-axis1.y, axis1.x, 0]).normalized()
    #     axis3 = axis1.cross(axis2)
    #     q_twist = mathutils.Quaternion(axis1, theta)
    #     axis2 = q_twist * axis2
    #     axis3 = q_twist * axis3
    #     material_frame = [ axis1, axis2, axis3 ]
    #     return material_frame
    #
    # def _ComputeNextMaterialFrame(self, material_frame, pt0, pt1, pt2, theta):
    #     segment1_dir = (pt1 - pt0).normalized()
    #     segment2_dir = (pt2 - pt1).normalized()
    #     axis = segment1_dir.cross(segment2_dir)
    #     dot = max(-1.0, min(1.0, segment1_dir.dot(segment2_dir)))
    #     angle = acos(dot)
    #     q_bend = mathutils.Quaternion(axis, angle)
    #     next_material_frame = [ q_bend * d for d in material_frame ]
    #     q_twist = mathutils.Quaternion(segment1_dir, theta)
    #     next_material_frame[1] = q_twist * next_material_frame[1]
    #     next_material_frame[2] = q_twist * next_material_frame[2]
    #     return next_material_frame

    def Select(self):
        scene.objects.active = self.centerline
        self.centerline.select = True

    def Update(self, keyframe, xs, thetas, refd1s, refd2s):
        # compute material_frame frame
        vec_xs = list(map(lambda x : mathutils.Vector(x), xs))
        material_frames = []
        for i, theta in enumerate(thetas):
            material_frames.append(self._ComputeMaterialFrame(theta, vec_xs[i], vec_xs[i+1], refd1s[i,:], refd2s[i,:]))

        thetas = np.pad(thetas, [0, 1], 'edge')
        # update inter control points
        for i, (x, y, z) in enumerate(xs):
            pt = self.polyline.points[i]
            pt.co = (x, y, z, 1.0)
            pt.radius = self.radius
            pt.tilt = thetas[i]

        # add keyframes for control points
        for point in self.polyline.points:
            point.keyframe_insert('co', frame=keyframe)
            point.keyframe_insert('tilt', frame=keyframe)

        # update material_frame frame
        for i in range(len(xs)-1):
            x1, y1, z1 = xs[i,:]
            x2, y2, z2 = xs[i + 1,:]
            v1 = mathutils.Vector([x1, y1, z1])
            v2 = mathutils.Vector([x2, y2, z2])
            self.material_frames[i].location = (v1 + v2) / 2.0
            self.material_frames[i].rotation_euler = self._ComputeEulerAngleFromFrame(material_frames[i])

        # add keyframe for material_frame frame
        for i in range(len(xs)-1):
            self.material_frames[i].keyframe_insert('location', frame=keyframe)
            self.material_frames[i].keyframe_insert('rotation_euler', frame=keyframe)
