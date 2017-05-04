import os
import bpy
import scipy.io
import numpy as np

from BlenderObject import *

class Scene(object):

    def __init__(self, fps, scene=0):
        super(Scene, self).__init__()
        self.fps = fps
        self.timestep = 1.0 / fps
        while scene >= len(bpy.data.scenes):
            bpy.ops.scene.new(type="EMPTY")
        self.scene = bpy.data.scenes[scene]
        self.scene.render.fps = fps
        if scene == 0: self.Init()
        # scene loading
        self.loaded_frames = set()
        self.last_loaded = None
        self.cache_path = None

    def IsFrameLoaded(self, frame):
        return frame in self.loaded_frames

    def SetCachePath(self, path):
        self.cache_path = path

    # switch this scene as active scene
    def Activate(self):
        bpy.data.screens['Default'].scene = self.scene
        bpy.context.screen.scene = self.scene

    # deletes the everything in the default scene
    def Init(self):
        self.Activate()
        self.DeleteObjectByName("Cube")
        # self.DeleteObjectByName("Camera")
        # self.DeleteObjectByName("Lamp")

    def SetViewportShading(self, view="VIEW_3D", shading="MATERIAL"):
        self.Activate()
        for area in bpy.context.screen.areas: # iterate through areas in current screen
            if area.type == view:
                for space in area.spaces: # iterate through spaces in current VIEW_3D area
                    if space.type == view:
                        space.viewport_shade = shading

    def SelectObjectByName(self, name):
        self.Activate()
        if name in bpy.data.objects:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects[name].select = True
            return bpy.data.objects[name]
        return None

    def DeleteObjectByName(self, name):
        self.Activate()
        if name in bpy.data.objects:
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects[name].select = True
            bpy.ops.object.delete()

    def HideObject(self, obj):
        obj.hide = True

    def ShowObject(self, obj):
        obj.hide = False

    def LinkObject(self, obj):
        self.scene.objects.link(obj)

    def UnlinkObject(self, obj):
        self.scene.objects.unlink(obj)

    def ComputeKeyframe(self, frame):
        simtime = self.timestep * frame
        return int(simtime * self.fps)

    def SetFrame(self, frame):
        self.scene.frame_set(frame)

    def SetDuration(self, keyframe):
        self.scene.frame_end = max(self.scene.frame_end, keyframe)

    def SetEndFrame(self, keyframe):
        self.scene.frame_end = keyframe

    def Load(self, frame, filename):
        # check frame loaded
        if self.IsFrameLoaded(frame):
            return
        # actual loading
        mdict = {}
        scipy.io.loadmat(filename, mdict)
        self.Update(frame, mdict)

    def Update(self, frame, mdict):
        pass

def _expand_to(tensor, to):
    if len(tensor.shape) == to:
        return tensor
    return np.expand_dims(tensor, 0)

class RodScene(Scene):

    def __init__(self, fps, scene=0):
        super(RodScene, self).__init__(fps, scene)
        self.number = scene
        self.rods =None
        self.obstacles = Sphere()

    def Update(self, frame, data):
        # mark frame as loaded
        self.loaded_frames.add(frame)

        keyframe = self.ComputeKeyframe(frame)
        self.SetDuration(frame)

        xs = _expand_to(data["cpos"], 4)
        ts = _expand_to(data["thetas"], 3)
        refd1s = _expand_to(data["refd1s"], 4)
        refd2s = _expand_to(data["refd2s"], 4)
        n_batch, n_rods, n_centerpoints, _ = xs.shape
        radius = data.get("radius", 0.02)

        spheres = None
        if "spheres" in data:
            spheres = data["spheres"]

        # init rods
        if self.rods is None:
            self.rods = [[
                Rod("{scene}-rod-{batch}-{id}".format(scene=self.number, batch=i, id=j), n_centerpoints, radius)
                for j in range(n_rods) ]
                for i in range(n_batch) ]

        # update rods
        for i in range(n_batch):
            for j in range(n_rods):
                self.rods[i][j].Update(keyframe, xs[i][j], ts[i][j], refd1s[i][j], refd2s[i][j])

        # update obstacle
        if spheres is not None:
            self.obstacles.Update(keyframe, spheres)


class HairScene(Scene):

    def __init__(self, fps, scene=0):
        super(HairScene, self).__init__(fps, scene)
        self.number = scene
        self.scene.tool_settings.use_keyframe_insert_auto = True
        self.anchors = None

    def SetMetadata(self, metadata):
        mdict = dict()
        scipy.io.loadmat(metadata, mdict)
        self.anchors = mdict["anchor"]

    def Load(self, frame, filename):
        if self.last_loaded == frame:
            return
        self.last_loaded = frame
        super(HairScene, self).Load(frame, filename)

    def Update(self, frame, data):
        print("Update frame {}".format(frame))
        keyframe = self.ComputeKeyframe(frame)

        xs = _expand_to(data["cpos"], 4)
        ts = _expand_to(data["thetas"], 3)
        refd1s = _expand_to(data["refd1s"], 4)
        refd2s = _expand_to(data["refd2s"], 4)
        n_batch, n_rods, n_centerpoints, _ = xs.shape

        obj = bpy.data.objects["Head"]
        particle_system = obj.particle_systems[0]
        hairs = particle_system.particles
        n_rods = len(hairs)
        n_segs = len(hairs[0].hair_keys) - 1

        world2local = obj.matrix_world.inverted()
        for i, h in enumerate(hairs):
            for j, hv in enumerate(h.hair_keys):
                if j <= 2: continue
                hv.co = world2local * mathutils.Vector(xs[0,i,j,:])

    def Dump(self, filename):
        obj = bpy.data.objects["Head"]
        hairs = obj.particle_systems[0].particles
        n_rods = len(hairs)
        n_segs = len(hairs[0].hair_keys) - 1

        # create numpy tensors
        cpos = np.zeros(shape=(n_rods, n_segs+1, 3), dtype=np.float32)
        cvel = np.zeros(shape=(n_rods, n_segs+1, 3), dtype=np.float32)
        theta = np.zeros(shape=(n_rods, n_segs), dtype=np.float32)
        omega = np.zeros(shape=(n_rods, n_segs), dtype=np.float32)
        initds = np.zeros(shape=(n_rods, 3), dtype=np.float32)

        def wc(vertex):
            return obj.matrix_world * vertex

        # assign cpos and theta (theta not available here)
        for i, h in enumerate(hairs):
            # assign cpos
            for j, hv in enumerate(h.hair_keys):
                cpos[i,j,:] = np.array(wc(hv.co))
            # compute initd1
            d1 = (wc(hairs[i].hair_keys[1].co) - wc(hairs[i].hair_keys[0].co)).normalized()
            d2 = mathutils.Vector([1, 0, 0])
            if d1.cross(d2).length == 0:
                d2 = mathutils.Vector([0, 1, 0])
            d3 = d1.cross(d2).normalized()
            initds[i,:] = np.array(d3)

        # assign anchor points
        anchors = np.zeros(shape=(self.scene.frame_end, n_rods * 2, 3), dtype=np.float32)
        for frame, anchor in enumerate(anchors):
            print("Generate frame %d" % frame)
            bpy.context.scene.frame_set(frame)
            for i, h in enumerate(hairs):
                anchor[i,:] = np.array(obj.matrix_world * h.hair_keys[0].co)

        for frame, anchor in enumerate(anchors):
            print("Generate frame %d" % frame)
            bpy.context.scene.frame_set(frame)
            for i, h in enumerate(hairs):
                anchor[i + n_rods,:] = np.array(obj.matrix_world * h.hair_keys[1].co)

        # create anchor indices
        indices = np.array([(i, 0) for i in range(n_rods)] + [(i, 1) for i in range(n_rods)], dtype=np.int32)

        print("Done")

        filename = os.path.abspath(filename)
        mdict = {
            "cpos"   : cpos,
            "cvel"   : cvel,
            "theta"  : theta,
            "omega"  : omega,
            "initd"  : initds,
            "anchor" : anchors,
            "anchor_indices" : indices,
        }
        if "Obstacle" in bpy.data.objects:
            obj = bpy.data.objects["Obstacle"]
            obstacle_centers = [ np.array(obj.location) ]
            obstacle_radii = [ np.array([obj.scale.x]) ]
            mdict["obstacle_centers"] = obstacle_centers,
            mdict["obstacle_radii"] = obstacle_radii,

        scipy.io.savemat(filename, mdict, appendmat=True)
