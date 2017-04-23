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

## Visualization of Unit Tests

For visualization of the motion data generated into /tmp/tftest# directories
(# is a number) by ``test_M.py``, please run ``.BlenderVis.py
--simdir=/tmp/tftest#`` under Rod/ directory.

Note: You don't need to call ``test_M.py`` explicitly if ``test.py`` has been
executed, since ``test.py`` runs ``test_M.py`` as part of unit tests.

## Authors

+ [Xinya Zhang](xinyazhang@utexas.edu)
+ [Tianyu Cheng](tianyu.cheng@utexas.edu)
