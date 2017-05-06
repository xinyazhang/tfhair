def restore():
    import mathutils
    obj = bpy.data.objects["Head"]
    hairs = obj.particle_systems[0].particles
    hair_step = bpy.data.particles["ParticleSettings"].hair_step + 1
    for i, h in enumerate(hairs):
        restl = []
        for j in range(hair_step-1):
            restl.append((hairs[i].hair_keys[j+1].co - hairs[i].hair_keys[j].co).length)
        unit = hairs[i].hair_keys[0].co / hairs[i].hair_keys[0].co.length
        hairs[i].hair_keys[0].co_local = (0, 0, 0)
        for j in range(1, hair_step):
            factor = sum(restl[:j])
            hairs[i].hair_keys[j].co_local = (0, 0, factor)
            hairs[i].hair_keys[j].co = hairs[i].hair_keys[0].co + unit * factor

restore()
