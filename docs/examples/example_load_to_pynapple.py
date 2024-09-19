"""
# Load an Allen Brain Map NWB into pynapple

In this example, we show how to load into pynapple an NWB directly downloaded from the Allen Brain Map.

!!! warning "Intracellular recordings NWBs and pynapple"
    NWB files containing intracellular recordings cannot be imported directly into pynapple.
    This is because pynapple requires a continuous time axis, but these experiments are typically divided into sweeps,
    with each sweep starting at t = 0 seconds.

    [At this link](https://github.com/pynapple-org/single-cell-neuropython-workshop-2024/blob/main/docs/examples/load_allen.py),
    we provide the `load_allen.py` module for loading intracellular recordings into `pynapple`.
     It works by concatenating all sweeps together and adding a fixed inter-trial interval between sweeps to
     create a continuous time axis.


For this example, let's assume that you download the dataset used in the tutorial directly from the Allen website:

[https://celltypes.brain-map.org/experiment/electrophysiology/478498617](https://celltypes.brain-map.org/experiment/electrophysiology/478498617)

To load it into pynapple, you can run the following code,
"""

# The "load_allen.py" module is in
# https://github.com/pynapple-org/single-cell-neuropython-workshop-2024/blob/main/docs/examples/load_allen.py
from load_allen import load_to_pynapple

# replace this with the path to the nwb data
path = "478498615_ephys.nwb"

# this will return the pynapple representation of the data.
# note: that we used the same naming as in "tutorial_pynapple_nemos_single_cell_full.py"
spikes, epochs,  current, voltage, sweep_metadata = load_to_pynapple(path)


