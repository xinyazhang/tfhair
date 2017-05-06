# Visualizer

This directory contains the visualizers for our simulation project.

## RodVis

RodVis.py is used to visualize regression tests. Use the following
command to visualize each test:

```
./RodVis.py -- <path/to/simulation/out/directory>
```

We save configuration of each frame under tmp directory.
Please check the following directories. Asteriks are wildcards.

```
/tmp/tftest*
/tmp/tfccd*

e.g.

# loading single simulation
./RodVis.py -- /tmp/tfccd6

# loading multiple simulations
# each simulation will be stored in its own scene, e.g. scene.001, scene.002
# check out blender outliner for scene switch
./RodVis.py -- /tmp/tftest0 /tmp/tftest1 ...
```

Blender will usually stay black until it is fully responsive.
Once scenes are loaded, press alt+a, or play button to visualize
the simulation result.

## HairVis

In order to visualize hair simulation, one needs to generate
initial conditions. We provide two hair models: short hair and long hair.

```
# This command will not terminate because blender is currently open.
# Leave blender open, because you will need it for visualization later

./HairVis.py -- --dump=ShortHair.mat --template=ShortHair.blend /tmp/tfhair/ShortHair/
./HairVis.py -- --dump=LongHair.mat --template=LongHair.blend /tmp/tfhair/LongHair/

# Open a new terminal, change to Rod directory.
# run bench_hair, takes
# a. about 2 minutes on Nvidia GeForce 970
# b. about 7 minutes on CPU

cd ../Rod
python2 bench_hair.py ../Vis/ShortHair.mat
python2 bench_hair.py ../Vis/LongHair.mat

# Once bench_hair.py finishes, go back to blender, press alt+a to play.
# Or just click on play button for simulation result.
# Partial results can be viewed while bench_hair simulation is running.
```
