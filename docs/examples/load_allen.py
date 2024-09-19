"""Load data from  the Allen Brain Map into pynapple."""

import h5py
from typing import Tuple
import pynapple as nap

import numpy as np


def _get_sweep_metadata(sweep_number: int, fh: h5py.File):
    sweep_metadata = {}

    # the sweep level metadata is stored in
    # stimulus/presentation/Sweep_XX in the .nwb file

    # indicates which metadata fields to return
    metadata_fields = ['aibs_stimulus_amplitude_pa',
                       'aibs_stimulus_name',
                       'gain', 'initial_access_resistance', 'seal']
    try:
        stim_details = fh['stimulus']['presentation'][
            'Sweep_%d' % sweep_number]
        for field in metadata_fields:
            # check if sweep contains the specific metadata field
            if field in stim_details.keys():
                sweep_metadata[field] = stim_details[field][()]

    except KeyError:
        sweep_metadata = {}

    return sweep_metadata


def _get_pipeline_version(fh: h5py.File) -> Tuple[int, int]:
    """
    Returns the AI pipeline version number, stored in the
    metadata field 'generated_by'. If that field is
    missing, version 0.0 is returned.

    Returns
    -------
    :
    The pipleline version
    """
    try:
        if 'generated_by' in fh["general"]:
            info = fh["general/generated_by"]
            # generated_by stores array of keys and values
            # keys are even numbered, corresponding values are in
            #   odd indices
            for i in range(len(info)):
                if info[i] == 'version':
                    version = info[i + 1]
                    break
        toks = version.split('.')
        if len(toks) >= 2:
            major = int(toks[0])
            minor = int(toks[1])
    except Exception:
        minor = 0
        major = 0
    return major, minor


def _get_sweep(sweep_number: int, fh: h5py.File):
    """ Retrieve the stimulus, response, index_range, and sampling rate
    for a particular sweep.  This method hides the NWB file's distinction
    between a "Sweep" and an "Experiment".  An experiment is a subset of
    of a sweep that excludes the initial test pulse.  It also excludes
    any erroneous response data at the end of the sweep (usually for
    ramp sweeps, where recording was terminated mid-stimulus).

    Some sweeps do not have an experiment, so full data arrays are
    returned.  Sweeps that have an experiment return full data arrays
    (include the test pulse) with any erroneous data trimmed from the
    back of the sweep.

    Parameters
    ----------
    sweep_number:
        The sweep ID.

    fh:
        The h5py file.

    Returns
    -------
    :
        A dictionary with 'stimulus', 'response', 'index_range', and
        'sampling_rate' elements.  The index range is a 2-tuple where
        the first element indicates the end of the test pulse and the
        second index is the end of valid response data.
    """
    swp = fh['epochs']['Sweep_%d' % sweep_number]

    # fetch data from file and convert to correct SI unit
    # this operation depends on file version. early versions of
    #   the file have incorrect conversion information embedded
    #   in the nwb file and data was stored in the appropriate
    #   SI unit. For those files, return uncorrected data.
    #   For newer files (1.1 and later), apply conversion value.
    major, minor = _get_pipeline_version(fh)
    if (major == 1 and minor > 0) or major > 1:
        # stimulus
        stimulus_dataset = swp['stimulus']['timeseries']['data']
        conversion = float(stimulus_dataset.attrs["conversion"])
        stimulus = stimulus_dataset[()] * conversion
        # acquisition
        response_dataset = swp['response']['timeseries']['data']
        conversion = float(response_dataset.attrs["conversion"])
        response = response_dataset[()] * conversion
    else:  # old file version
        stimulus_dataset = swp['stimulus']['timeseries']['data']
        stimulus = stimulus_dataset[()]
        response = swp['response']['timeseries']['data'][()]

    if 'unit' in stimulus_dataset.attrs:
        unit = stimulus_dataset.attrs["unit"].decode('UTF-8')

        unit_str = None
        if unit.startswith('A'):
            unit_str = "Amps"
        elif unit.startswith('V'):
            unit_str = "Volts"
        assert unit_str is not None, Exception(
            "Stimulus time series unit not recognized")
    else:
        unit = None
        unit_str = 'Unknown'

    swp_idx_start = swp['stimulus']['idx_start'][()]
    swp_length = swp['stimulus']['count'][()]

    swp_idx_stop = swp_idx_start + swp_length - 1
    sweep_index_range = (swp_idx_start, swp_idx_stop)

    # if the sweep has an experiment, extract the experiment's index
    # range
    try:
        exp = fh['epochs']['Experiment_%d' % sweep_number]
        exp_idx_start = exp['stimulus']['idx_start'][()]
        exp_length = exp['stimulus']['count'][()]
        exp_idx_stop = exp_idx_start + exp_length - 1
        experiment_index_range = (exp_idx_start, exp_idx_stop)
    except KeyError:
        # this sweep has no experiment.  return the index range of the
        # entire sweep.
        experiment_index_range = sweep_index_range

    assert sweep_index_range[0] == 0, Exception(
        "index range of the full sweep does not start at 0.")

    return {
        'stimulus': stimulus,
        'response': response,
        'stimulus_unit': unit_str,
        'index_range': experiment_index_range,
        'sampling_rate': 1.0 * swp['stimulus']['timeseries'][
            'starting_time'].attrs['rate']
    }


def _get_sweep_numbers(fh):
    """ Get all sweep numbers in the file, including test sweeps."""
    sweeps = [int(e.split('_')[1])
              for e in fh['epochs'].keys() if e.startswith('Sweep_')]
    return sweeps


def _get_spike_times(sweep_number, fh, key="spike_times"):
    """ Return any spike times stored in the NWB file for a sweep.

    Parameters
    ----------
    sweep_number: int
        index to access
    key : string
        label where the spike times are stored (default
        NwbDataSet.SPIKE_TIMES)

    Returns
    -------
    list:
       List of spike times in seconds relative to the start of the sweep
    """
    DEPRECATED_SPIKE_TIMES = "aibs_spike_times"

    datasets = ["analysis/%s/Sweep_%d" % (key, sweep_number),
                "analysis/%s/Sweep_%d" % (
                DEPRECATED_SPIKE_TIMES, sweep_number)]

    for ds in datasets:
        if ds in fh:
            return fh[ds][()]
    return []


def _convert_to_epoch_dict(sweep_metadata, trial_interval_set) -> dict[nap.IntervalSet]:
    iset_dict = {}
    for i, meta in enumerate(sweep_metadata.values()):
        stim_type = meta["aibs_stimulus_name"].decode('utf-8')
        if stim_type in iset_dict:
            iset_dict[stim_type] = np.vstack(
                (iset_dict[stim_type], np.asarray(trial_interval_set[i]))
            )
        else:
            iset_dict[stim_type] = np.asarray(trial_interval_set[i])

    return {key: nap.IntervalSet(*iset_dict[key].T) for key in iset_dict}



def load_to_pynapple(path: str, shift_trials_by_sec: float = 5.)\
        -> Tuple[nap.TsGroup, dict[nap.IntervalSet], nap.Tsd, nap.Tsd, dict]:
    """
    Load the intracellular recording as pynapple time series.

    Parameters
    ----------
    path:
        The path to the nwb dataset.
        https://celltypes.brain-map.org/experiment/electrophysiology/
    shift_trials_by_sec:
        Shift used to concatenate trials in seconds. In the original dataset,
        every trial starts at t=0.
        Pynapple time index must be monotonically increasing
        instead. We artificially add a time shift to each trial in order
        to create a consistent, monotonic time index.

    Returns
    -------
    units
        The spike times in seconds as a TsGroup.
    epochs:
        A dictionary of interval set containing the trial start and end in seconds of each epoch type.
    stimulus
        The injected current time series as Tsd.
    responses
        The sub-threshold responses as Tsd.
    sweep_metadata
        Metadata describing the stimulation protocol.
    """
    # Load the hdf5 file
    with h5py.File(path) as fh:

        # print len trials
        sweap_nums = np.sort(_get_sweep_numbers(fh))

        # Initialize the objects that will be used to construct
        # the pynapple timeseries.
        init_trial_time = 0
        sweep_metadata = {}
        stimulus = []
        responses = []
        units = []
        time_trials = []
        starts = []
        ends = []
        for cc, num in enumerate(sweap_nums):
            # get the data for a specific trial
            dat = _get_sweep(num, fh)
            sweep_metadata[num] = _get_sweep_metadata(num, fh)

            # append metadata information
            sweep_metadata[num].update(
                {
                    "stimulus_unit": dat["stimulus_unit"],
                    "sampling_rate": dat["sampling_rate"],
                    "response_unit": "Volt"
                }
            )

            # compute the time index for the trial by dividing the number of
            # samples by the sampling rate and adding a time shift that
            # guarantees that the time index is strictly increasing.
            time_trials.append(
                np.arange(dat["stimulus"].shape[0]) / dat["sampling_rate"] + init_trial_time
            )
            # add the same time shift to the spike times of the trial
            units.append(np.asarray(_get_spike_times(num, fh)) + init_trial_time)

            # append voltage and injected current
            responses.append(dat["response"])
            stimulus.append(dat["stimulus"])

            # store the first and last timestamp of each trial
            starts.append(time_trials[-1][0])
            ends.append(time_trials[-1][-1])

            # compute the next time shift
            init_trial_time = shift_trials_by_sec + time_trials[-1][-1]

        # define the pynapple objects
        trial_interval_set = nap.IntervalSet(start=starts, end=ends)
        units = nap.TsGroup(
            {0: nap.Ts(t=np.hstack(units))}, time_support=trial_interval_set
        )
        responses = nap.Tsd(
            t=np.hstack(time_trials),
            d=np.hstack(responses),
            time_support=trial_interval_set,
        )
        stimulus = nap.Tsd(
            t=responses.t, d=np.hstack(stimulus), time_support=trial_interval_set
        )
        epochs = _convert_to_epoch_dict(sweep_metadata, trial_interval_set)
        return units, epochs, stimulus, responses, sweep_metadata

