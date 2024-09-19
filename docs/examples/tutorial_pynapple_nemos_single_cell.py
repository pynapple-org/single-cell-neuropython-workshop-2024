# -*- coding: utf-8 -*-
"""# Tutorial pynapple & NeMoS"""

# %%
# 
# Data for this notebook is a patch clamp experiment with a mouse V1 neuron, from the [Allen Brain Atlas](https://celltypes.brain-map.org/experiment/electrophysiology/478498617)
# 
# ![Allen Brain Atlas view of the data we will analyze.](../../assets/allen_data.png)
# 
# ## Learning objectives
# 
# - Learn how to explore spiking data and do basic analyses using pynapple
# - Learn how to structure data for NeMoS
# - Learn how to fit a basic Generalized Linear Model using NeMoS
# - Learn how to retrieve the parameters and predictions from a fit GLM for
#   intrepetation.

# %%
# !!! warning
#     This tutorial uses matplotlib for displaying the figure
#
#     You can install all with `pip install matplotlib requests tqdm`
#

# !pip install matplotlib requests tqdm

# %%
# In case you did not install beforehand pynapple and nemos, here is the command to install it.

# !pip install pynapple nemos

# %%
# Import everything

import math
import os

import jax
import matplotlib.pyplot as plt
import nemos as nmo
import nemos.glm
import numpy as np
import pynapple as nap
import requests
import tqdm
import workshop_utils.plotting as plotting

# %%
# configure plots some
plt.style.use("workshop_utils/nemos.mplstyle")



# %%
# ## Data Streaming
# 
# - Stream the data. Format is [Neurodata Without Borders (NWB) standard](https://nwb-overview.readthedocs.io/en/latest/)
# 

path = "allen_478498617.nwb"
if path not in os.listdir("."):
  r = requests.get(f"https://osf.io/vf2nj/download", stream=True)
  block_size = 1024*1024
  with open(path, 'wb') as f:
    for data in tqdm.tqdm(r.iter_content(block_size), unit='MB', unit_scale=True,
      total=math.ceil(int(r.headers.get('content-length', 0))//block_size)):
      f.write(data)



# %%
# ## Pynapple
# ### Data structures and preparation
# 
# - Open the NWB file with [pynapple](https://pynapple-org.github.io/pynapple/)
# 

# %%





# %%
# 
# ![Annotated view of the data we will analyze.](../../assets/allen_data_annotated.gif)
# 
# - `stimulus`: Tsd containing injected current, in Amperes, sampled at 20k Hz.
# - `response`: Tsd containing the neuron's intracellular voltage, sampled at 20k Hz.
# - `units`: Tsgroup, dictionary of neurons, holding each neuron's spike timestamps.
# - `epochs`: IntervalSet, dictionary with start and end times of different intervals,
#   defining the experimental structure, specifying when each stimulation protocol began
#   and ended.
# 

# %%




# %%
# 
# - `Noise 1`: epochs of random noise
# 

# %%




# %%
# 
# - Let's focus on the first epoch.
# 

# %%




# %%
# 
# - `current` : Tsd (TimeSeriesData) : time index + data
# 

# %%

# convert current from Ampere to pico-amperes, to match the Allen Institute figures and
# move the values to a more reasonable range.




# %%
# 
# - `restrict` : restricts a time series object to a set of time intervals delimited by an IntervalSet object
# 

# %%




# %%
# 
# - `TsGroup` : a custom dictionary holding multiple `Ts` (timeseries) objects with
#   potentially different time indices.
# 

# %%





# %%
# 
# We can index into the `TsGroup` to see the timestamps for this neuron's
# spikes:
# 

# %%




# %%
# 
# Let's restrict to the same epoch `noise_interval`:
# 

# %%






# %%
# 
# Let's visualize the data from this trial:
# 

# %%

# fig, ax = plt.subplots(1, 1, figsize=(8, 2))
# ax.plot(current, "grey")
# ax.plot(spikes.to_tsd([-5]), "|", color="k", ms = 10)
# ax.set_ylabel("Current (pA)")
# ax.set_xlabel("Time (s)")



# %%
# ### Basic analyses
# 
# The Generalized Linear Model gives a predicted firing rate. First we can use
# pynapple to visualize this firing rate for a single trial.
# 
# - `count` : count the number of events within `bin_size`
# 

# %%




# %%
# 
# Let's convert the spike counts to firing rate :
# 
# - `smooth` : convolve with a Gaussian kernel
# 

# %%
# the inputs to this function are the standard deviation of the gaussian in seconds and
# the full width of the window, in standard deviations. So std=.05 and size_factor=20
# gives a total filter size of 0.05 sec * 20 = 1 sec.

# %%
# convert from spikes per bin to spikes per second (Hz)




# %%
# we're hiding the details of the plotting function for the purposes of this
# tutorial, but you can find it in the associated github repo if you're
# interested:
# https://github.com/flatironinstitute/ccn-workshop-fens-2024/blob/main/src/workshop_utils/plotting.py

# plotting.current_injection_plot(current, spikes, firing_rate)



# %%
# 
# What is the relationship between the current and the spiking activity?
# [`compute_1d_tuning_curves`](https://pynapple-org.github.io/pynapple/reference/process/tuning_curves/#pynapple.process.tuning_curves.compute_1d_tuning_curves) : compute the firing rate as a function of a 1-dimensional feature.
# 
#





# %%
# 
# Let's plot the tuning curve of the neuron.
# 



# plotting.tuning_curve_plot(tuning_curve)



# %%
# ## NeMoS
# ### Preparing data
# 
#  Get data from pynapple to NeMoS-ready format:
# 
#   - predictors and spikes must have same number of time points
# 

# enter code here


# %%
# 
#   - predictors must be 2d, spikes 1d
# 

# enter code here


# %%
# ### Fitting the model
# 
#   - define a GLM object
# 

# enter code here


# %%
# 
#   - call fit and retrieve parameters
# 

# enter code here


# %%
# 
#   - generate and examine model predictions.
# 

# enter code here
# plotting.current_injection_plot(current, spikes, firing_rate,
#                                                smooth_predicted_fr)


# %%
# 
#   - what do we see?
# 

# enter code here


# %%
# 
#   - examine tuning curve &mdash; what do we see?
# 

# enter code here
# fig = plotting.tuning_curve_plot(tuning_curve)
# fig.axes[0].plot(tuning_curve_model, color="tomato", label="glm")
# fig.axes[0].legend()


# %%
# ### Extending the model
# 
#   - choose a length of time over which the neuron integrates the input current
# 

# enter code here


# %%
# 
#   - define a basis object
# 

# enter code here


# %%
# 
#   - create the design matrix
#   - examine the features it contains
# 

# enter code here
# in this plot, we're normalizing the amplitudes to make the comparison easier --
# the amplitude of these features will be fit by the model, so their un-scaled
# amplitudes is not informative
# plotting.plot_current_history_features(binned_current, current_history, basis,
#                                                       current_history_duration_sec)


# %%
# 
#   - create and fit the GLM
#   - examine the parameters
# 

# enter code here


# %%
# 
#   - compare the predicted firing rate to the data and the old model
#   - what do we see?
# 

# enter code here


# %%
# 
#   - examine the predicted average firing rate and tuning curve
#   - what do we see?
# 

# enter code here


# %%
# 
#   - use log-likelihood to compare models
# 

# enter code here


# %%
# ### Finishing up
# 
#   - what if you want to compare models across datasets?
# 

# enter code here


# %%
# 
#   - what about spiking?
# 

# enter code here


# %%
# ## Further Exercises
# 
#   - what else can we do?
# 
# ## Data citation
# 
# The data used in this tutorial is from the Allen Brain Map, with the
# [following
# citation](https://knowledge.brain-map.org/data/1HEYEW7GMUKWIQW37BO/summary):
# 
# **Contributors**: Agata Budzillo, Bosiljka Tasic, Brian R. Lee, Fahimeh
# Baftizadeh, Gabe Murphy, Hongkui Zeng, Jim Berg, Nathan Gouwens, Rachel
# Dalley, Staci A. Sorensen, Tim Jarsky, Uygar Sümbül Zizhen Yao
# 
# **Dataset**: Allen Institute for Brain Science (2020). Allen Cell Types Database
# -- Mouse Patch-seq [dataset]. Available from
# brain-map.org/explore/classes/multimodal-characterization.
# 
# **Primary publication**: Gouwens, N.W., Sorensen, S.A., et al. (2020). Integrated
# morphoelectric and transcriptomic classification of cortical GABAergic cells.
# Cell, 183(4), 935-953.E19. https://doi.org/10.1016/j.cell.2020.09.057
# 
# **Patch-seq protocol**: Lee, B. R., Budzillo, A., et al. (2021). Scaled, high
# fidelity electrophysiological, morphological, and transcriptomic cell
# characterization. eLife, 2021;10:e65482. https://doi.org/10.7554/eLife.65482
# 
# **Mouse VISp L2/3 glutamatergic neurons**: Berg, J., Sorensen, S. A., Miller, J.,
# Ting, J., et al. (2021) Human neocortical expansion involves glutamatergic
# neuron diversification. Nature, 598(7879):151-158. doi:
# 10.1038/s41586-021-03813-8