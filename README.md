# Elastic-Rod-based Hair Simulation

## Setup and Test

```
# Create venv directory for essential tensorflow support
# This only needs to be run by once
./bootstrap.sh
source venv/bin/activate # Enable python environment with tensorflow
```

Now the unit tests can be started with ``./Rod/test.py``. There is no visual results by far, however.

## Notes on System Requirements

bootstrap.sh will install AVX2-enabled tensorflow package if supported.
Please use systems with AVX2 support for best performance.

Check AVX2 support: run ``grep avx2 /proc/cpuinfo`` in command line. If there
are outputs from this command, then AVX2 is supported by you CPU. Otherwise
nothing will be printed.

## Notes on Visualization

Please do not run the script in Vis directory with python, because blender internally uses
its own version of python. Instead, invoke python script either with

```
blender -P /path/to/script
```

Or, just directly execute it

```
/path/to/script
```

## Visualization of Unit Tests

For visualization of the motion data generated into /tmp/tftest# directories
(# is a number) by ``test_M.py``, please run

``RodVis.py /tmp/tftest#`` under Rod/ directory.

Note: You don't need to call ``test_M.py`` explicitly if ``test.py`` has been
executed, since ``test.py`` runs ``test_M.py`` as part of unit tests.

Example:

```
cd /path/to/repo/Rod
python test_M.py

cd /path/to/repo/Vis
./RodVis.py /tmp/tftest0 /tmp/tftest1 /tmp/tftest2 /tmp/tftest3 /tmp/tftest4 ...
```

If multiple test directories are specified, Blender will load all tests and
will likely stay unresponsive (black screen). Once loaded, you should see each
test loaded as different scenes in the outliner panel (in default view, outliner
should be located in the top right corner, where you select objects in the scene).

Switching through each scene, you can visualize each test individually.

Similarly, ``test_MCCD.py`` generates data files under ``/tmp/tfccd0`` to
``/tmp/tfccd8/``. They can also be visualized.

## Visualization of Hair Simulation

We provide two hair models for simulation. One for short hair, and
one for long hair. In the following commands, you just need run one
from each paragraph, i.e. just run the commands for short hair model,
or just for the long hair model.

```
# Launch a new terminal, change to Vis directory.
# This command will not terminate because blender will stay open.
# Leave blender OPEN, because you will need it for visualization later

cd /path/to/repo/Vis
./HairVis.py -- --dump=ShortHair.mat --template=ShortHair.blend /tmp/tfhair/ShortHair/
./HairVis.py -- --dump=LongHair.mat --template=LongHair.blend /tmp/tfhair/LongHair/

# Launch a new terminal, change to Rod directory.
# Run hair simulation bench using the following commands
# Takes ~1.5 min on Nvidia Titan X
# Takes ~3 min on UTCS lab machine

cd /path/to/repo/Rod
python2 bench_hair.py ../Vis/ShortHair.mat
python2 bench_hair.py ../Vis/LongHair.mat

# run bench_hair with CCD
# Takes ~5 min on Nvidia Titan X
# Takes ~9 min on UTCS lab machine

cd /path/to/repo/Rod
python2 bench_hair.py --enable-collision ../Vis/ShortHair.mat
python2 bench_hair.py --enable-collision ../Vis/LongHair.mat

# Once bench_hair.py finishes, go back to blender, press alt+a to play.
# Or just click on play button for simulation result.
# Partial results can be viewed while bench_hair simulation is running.
```

## Authors

+ [Xinya Zhang](xinyazhang@utexas.edu)
+ [Tianyu Cheng](tianyu.cheng@utexas.edu)
