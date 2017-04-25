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

    # switch this scene as active scene
    def Activate(self):
        bpy.data.screens['Default'].scene = self.scene
        bpy.context.screen.scene = self.scene

    # deletes the everything in the default scene
    def Init(self):
        self.Activate()
        self.DeleteObjectByName("Cube")
        self.DeleteObjectByName("Camera")
        self.DeleteObjectByName("Lamp")

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

    def Load(self, frame, filename):
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

    def Load(self, frame, filename):
        mdict = {}
        scipy.io.loadmat(filename, mdict)
        self.Update(frame, mdict)

    def Update(self, frame, data):
        keyframe = self.ComputeKeyframe(frame)
        self.SetDuration(frame)

        xs = _expand_to(data["cpos"], 4)
        ts = _expand_to(data["thetas"], 3)
        refd1s = _expand_to(data["refd1s"], 4)
        refd2s = _expand_to(data["refd2s"], 4)
        n_batch, n_rods, n_centerpoints, _ = xs.shape
        radius = data.get("radius", 0.02)

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

class HairScene(Scene):

    def __init__(self, fps, scene=0):
        super(HairScene, self).__init__(fps, scene)
        self.number = scene

    def Load(self, frame, filename):
        pass

    def Update(self, frame, data):
        pass

    def Dump(self, filename):
        obj = bpy.context.object
        hairs = obj.particle_systems[0].particles
        n_rods = len(hairs)
        n_segs = len(hairs[0].hair_keys) - 1
        # create numpy tensors
        cpos = np.zeros(shape=(n_rods, n_segs+1, 3), dtype=np.float32)
        cvel = np.zeros(shape=(n_rods, n_segs+1, 3), dtype=np.float32)
        theta = np.zeros(shape=(n_rods, n_segs), dtype=np.float32)
        omega = np.zeros(shape=(n_rods, n_segs), dtype=np.float32)
        initds = np.zeros(shape=(n_rods, 3), dtype=np.float32)
        # assign cpos and theta (theta not available here)
        for i, h in enumerate(hairs):
            # assign cpos
            for j, hv in enumerate(h.hair_keys):
                cpos[i,j,:] = np.array(hv.co)
            # compute initd1
            d1 = (hairs[i].hair_keys[1].co - hairs[i].hair_keys[0].co).normalized()
            d2 = mathutils.Vector([1, 0, 0])
            if d1.cross(d2).length == 0:
                d2 = mathutils.Vector([0, 1, 0])
            d3 = d1.cross(d2)
            initds[i,:] = np.array(d3)

        filename = os.path.abspath(filename)
        mdict = {
            "cpos"   : cpos,
            "cvel"   : cvel,
            "theta"  : theta,
            "omega"  : omega,
            "initd"  : initds
        }
        scipy.io.savemat(filename, mdict, appendmat=True)
