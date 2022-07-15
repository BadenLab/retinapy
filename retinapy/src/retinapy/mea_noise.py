from typing import Tuple, List, BinaryIO, Dict
from collections import namedtuple
import logging
import pathlib
import math
import numpy as np
from numpy.random.mtrand import f
import pandas as pd
import scipy
import scipy.signal
import h5py
# TODO: switch to pickle after upgrading Python version.
import pickle5 as pickle 


ELECTRODE_FREQ = 17852.767845719834 # Hz
STIMULUS_FREQ = 20 # Hz
STIMULUS_LOOP_MINS = 20
EXPERIMENT_DURATION_MINS = 15
EXPERIMENT_DURATION_SECS = EXPERIMENT_DURATION_MINS * 60 
STIMULUS_LOOP_SECS = STIMULUS_LOOP_MINS * 60
NUM_STIMULUS_LEDS = 4
REC_NAMES_FILENAME = 'recording_names.pickle'
CLUSTER_IDS_FILENAME = 'cluster_ids.pickle'
IDS_FILE_PATTERN = '{part}_ids.npy'
SNIPPET_FILE_PATTERN = '{part}_snippets.npy'
RNG_SEED = 123


# Named tuple for stimuli.
Stimulus = namedtuple('Stimulus', ['name', 'display_hex', 'import_name'])
stimuli = [
    Stimulus('red',   '#ff0a0a', '/Red_Noise'),
    Stimulus('green', '#0aff0a', '/Green_Noise'),
    Stimulus('blue',  '#0a0aff', '/Blue_Noise'),
    Stimulus('uv',    '#303030', '/UV_Noise')]
stimulus_names = tuple(s.name for s in stimuli)


def load_fullfield_stimulus(stimulus_path: str)	-> np.ndarray:
    """
    Loads the stimulus data from the HDF5 file as a Pandas DataFrame.

    Dataframe structure
    -------------------
        - integer index, representing stimulus frames.
        - columns are ['red', 'green', 'blue', 'uv']
        - values are 0 or 1 representing whether the corresponding LED is ON
          or OFF at the given stimulus frame.
    """
    with h5py.File(stimulus_path, 'r') as f:
        # The data has shape: [4, 24000, 10374]. This corresponds to 4 lights,
        # on-off pattern for 20min at 20 Hz (24000 periods), and 10374 boxes 
        # arranged over 2D screen space. In the full-field experiment, only a
        # single box was used, and hence the [:,0] access pattern.
        colour_noise = np.array([
            f[stimuli[0].import_name][:,0],
            f[stimuli[1].import_name][:,0],
            f[stimuli[2].import_name][:,0],
            f[stimuli[3].import_name][:,0]]).transpose()
    assert colour_noise.shape[0] == STIMULUS_LOOP_SECS * STIMULUS_FREQ
    # We are doing a lot of slicing in the time-step dimension, so keep it 
    # as the row dimension. Also insure the array is contiguous for further
    # speed-ups. 
    res = np.ascontiguousarray(colour_noise)
    return res


def load_response(response_path: str, keep_kernels=False) -> pd.DataFrame:
    """Loads the spike data from a Pickle file as a Pandas DataFrame.

    The input path should point to standard pickle file (not zipped).

    Dataframe structure
    -------------------
    The dataframe uses a multiindex: ['Cell index', 'Stimulus ID', 'Recording'].
        - The cell index refers to the id assigned by the spike sorter. 
        - TODO: The stimulus ID is what?
        - The 'Recording' index is a human readable string to identify a 
            recording session.

    The Dataframe has 2 columns: ['Kernel', 'Spikes'].
        - The each row contains a (2000,4) shaped Numpy array holding the
          precomputed response kernel for the matching cell-stimulus-recording.
          The kernel represents 2000 miliseconds, with the spike located at
          1000 ms. 
        - The 'Spikes' column contains a (17993,) shaped *masked* Numpy array,
          holding a variable number of integers. Each integer reprents the
          time (in sensor readings) at which the spike occurred.
    """
    with open(response_path, 'rb') as f:
        res = pickle.load(f)
    if not keep_kernels:
        res.drop('Kernel', axis=1)
    return res


def recording_names(response: pd.DataFrame) -> list:
    """Return the list of recording names."""
    rec_list = response.index.get_level_values('Recording').unique().tolist()
    return rec_list


def _list_to_index_map(l) -> Dict[str,int]:
    """
    Returns a map from list elements to their index in the list.
    """
    return {x: i for i, x in enumerate(l)}


def cluster_ids(response: pd.DataFrame, rec_name: str) -> list:
    """Returns the list of cluster ids for a given recording.

    Stimulus ID = 7 is currently ignored (actually, anything other than 1).
    """
    # Note: I'm not sure if it's better to request by recording name or
    # the recording id.
    stimulus_id = 1 # not 7.
    cluster_ids = response.xs((stimulus_id, rec_name), 
            level=('Stimulus ID', 'Recording'), drop_level=True).index \
                    .get_level_values('Cell index').unique().tolist()
    return cluster_ids


def _butter_lowpass(order, filter_half_fq):
    """
    Returns a butterworth lowpass filter.
    """
    b, a = scipy.signal.butter(order, filter_half_fq, btype='lowpass')
    return b, a


def _lowpass_filter(stimulus: np.ndarray) -> np.ndarray:
    """
    Filter (low-pass) the stimulus.

    This is needed to prevent aliasing.

    Resources on filtering
    ----------------------
    Mini-tutorial on filtering with scipy:
        https://danielmuellerkomorowska.com/2020/06/08/filtering-data-with-scipy/
    The answer here is what is being used in our code:
        https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
    Remark on filtfilt vs lfilter:
        https://dsp.stackexchange.com/questions/19084/applying-filter-in-scipy-signal-use-lfilter-or-filtfilt
        (tl;dr) We can use filtfilt instead of lfilter, as we are offline (non-streaming).
    """
    filter_half_fq = 0.5
    # Order 1, as opposed to a higher order, was selected as it seemed nice to 
    # not have any overshoot (values >1 or <0).
    order = 1
    # Only filter time axis.
    time_axis = 0 
    b, a = _butter_lowpass(order, filter_half_fq)
    return scipy.signal.filtfilt(b, a, stimulus, axis=time_axis)


def upsample_stimulus(stimulus: np.ndarray, factor: int) -> np.ndarray:
    """
    Upsamples the stimulus by the given factor.

    Args:
        stimulus: The stimulus to upsample. This is a 1D or 2D array. If 2D,
        timestep is the first axis and color channel is the second axis.
        factor: factor by which to increase the samples of the stimulus.
    """
    if factor % 1 != 0:
        raise ValueError('Upsampling factor must be an integer.')
    if factor < 1:
        raise ValueError(f'Only positive zoom factors are supported. '
                f'Got {factor}.')
    elif factor == 1:
        logging.warning('Upsampling by 1 is a no-op.')
    # Zoom the time axis and not the color axis. Remember, shape is TC. 
    zoomed_stimulus = np.kron(stimulus, np.ones((factor, 1)))
    filtered_stimulus = _lowpass_filter(zoomed_stimulus)
    return filtered_stimulus


def stimulus_slice(sampled_stimulus: np.ndarray, sampling_rate: float, 
        spike: int, total_len: int, post_spike_len: int) -> np.ndarray:
    """
    Return a subset of the stimulus around the spike point.

    Args:
        sampled_stimulus: The stimulus, upsampled to the spike frequency.
        sampling_rate: The sampling rate of the stimulus.
        spike: The sensor sample in which the spike was detected.
        total_len: The length of the snippet in stimulus frames. 
        post_spike_len: The number of frames to pad after the spike.
    Returns:
        A Numpy array of shape (total_len, NUM_STIMULUS_LEDS).

    Note 1: The total length describes the snippet length inclusive of the post
        spike padding.
    Note 2: If the spike falls exactly on the boundary of stimulus samples,
            then the earlier stimulus sample will be used.

    Example
    =======

        frame #:  |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |
        ===========================================================================
        stim:     |   0   |   1   |   1   |   0   |   0   |   1   |   0   |   0   |
                  |   1   |   1   |   0   |   0   |   1   |   0   |   0   |   0   |
                  |   1   |   1   |   1   |   0   |   0   |   0   |   0   |   1   |
                  |   0   |   0   |   0   |   0   |   1   |   0   |   1   |   1   |
        ===========================================================================
        resp:     |----------------------------------*----------------------------|

    The slice with parameters:
        - total length = 6
        - post spike length = 1

    Would be:

                  |   1   |   1   |   0   |   0   |   1   |              
                  |   1   |   0   |   0   |   1   |   0   |              
                  |   1   |   1   |   0   |   0   |   0   |              
                  |   0   |   0   |   0   |   1   |   0   |              



    Frame calculation
     ----------------
    (I'm not 100% confident that the below sample choice is correct.)

    In the following scenario:

        stimulus frame:  |  0  |  1  |  2  |  3  |
        spike:           |-----*-----------------|

    The stimulus frame in which the spike falls is:

        stimulus frame of spike = floor(spike_frame * ELECTRODE_FREQ / 
                                                       stimulus_sampling_rate)
                                = 1  (in the above example)

    In the following scenario all of the following spikes fall within the same 
    stimulus frame:

        stimulus frame: |   0   |   1   |   2   |
        spike time:     |-------********--------|
    """
    if total_len < post_spike_len + 1:
        raise ValueError(f'Total length must be at least 1 greater than the '
                f'post-spike lengths + 1. Got total_len ({total_len}) and '
                f'post_spike_len ({post_spike_len}).')
    # 1. Select the sample to represent the spike.
    stimulus_frame_of_spike = int(spike * (sampling_rate / ELECTRODE_FREQ))
    # 2. Calculate the snippet start and end.
    # The -1 appears as we include the spike sample in the snippet.
    win_start = (stimulus_frame_of_spike + post_spike_len) - (total_len - 1)
    win_end = stimulus_frame_of_spike + post_spike_len + 1 
    left_pad = max(-win_start, 0)
    right_pad = max(win_end - sampled_stimulus.shape[0], 0)
    win_start = max(win_start, 0)
    win_end = min(win_end, sampled_stimulus.shape[0])
    # 3. Extract the slice. 
    # The next line is a performance bottleneck.
    sample_slice = sampled_stimulus[win_start:win_end]
    # 4. Pad in case the spike is too close to the start or end of the stimulus.
    # Constant or mean padding? Probably not a big deal, I guess.
    if left_pad or right_pad:
        sample_slice = np.pad(sample_slice, ((left_pad, right_pad), (0, 0)), 
                mode='constant', constant_values=0)
    return sample_slice


def _save_recording_names(rec_list, save_dir) -> pathlib.Path:
    """
    Saves a list of recording names. 

    The list functions as an id to name map.
    """
    save_path = pathlib.Path(save_dir) / REC_NAMES_FILENAME
    # Create out file
    with save_path.open('wb') as f:
        pickle.dump(rec_list, f)
    return save_path


def _save_cluster_ids(cell_cluster_ids, rec_id, data_root_dir) -> pathlib.Path:
    """
    Saves a list of cluster ids. 

    The list functions as an id to name map.
    """
    parent_dir = pathlib.Path(data_root_dir) / str(rec_id)
    # Create if not exists
    parent_dir.mkdir(parents=False, exist_ok=True)
    save_path = pathlib.Path(data_root_dir) / str(rec_id) / 'cluster_ids.pickle'
    # Create out file
    with save_path.open('wb') as f:
        pickle.dump(cell_cluster_ids, f)
    return save_path


def create_snippet_training_data(
        stimulus: pd.DataFrame, 
        response: pd.DataFrame,
        save_dir,
        snippet_len: int = 1000,
        snippet_pad: int = 200,
        stimulus_freq: float = STIMULUS_FREQ,
        stimulus_zoom: int = 5,
        empty_snippets: float = 0,
        snippets_per_file: int = 1024):
    save_dir = pathlib.Path(save_dir)
    _save_recording_names(recording_names(response), save_dir)
    up_stimulus = upsample_stimulus(np.array(stimulus), stimulus_zoom)
    # TODO: only use stimulus 1? What is stimulus 7?
    by_rec = response.xs(1, level='Stimulus ID', 
            drop_level=True).groupby('Recording')
    for rec_id, (rec_name, df) in enumerate(by_rec):
        _write_rec_snippets(up_stimulus, df, rec_name, rec_id, save_dir, 
                snippet_len, snippet_pad, stimulus_freq, empty_snippets, 
                snippets_per_file)


def _write_rec_snippets(
        up_stimulus: pd.DataFrame, 
        response: pd.DataFrame,
        rec_name: str, 
        rec_id: int,
        data_root_dir, 
        snippet_len: int, 
        snippet_pad: int, 
        stimulus_freq: float, 
        empty_snippets: float,
        snippets_per_file: int):
    """
    Generate, shuffle and write the snippets for a single recording.

    The snippets may be split across multiple files.
    """
    # Create directory for recording.
    rec_dir = pathlib.Path(data_root_dir) / str(rec_id)
    rec_dir.mkdir(parents=False, exist_ok=True)
    # Extract the stimulus snippet around the spikes.
    snippets, cluster_ids = spike_snippets(up_stimulus, response, rec_name, 
            snippet_len, snippet_pad, stimulus_freq)
    # Save the cluster ids.
    _save_cluster_ids(cluster_ids, rec_id, data_root_dir)
    assert len(cluster_ids) == len(snippets)
    # Shuffle the cluster ids and snippets together.
    rng = np.random.default_rng(RNG_SEED)
    shuffle_idxs = np.arange(len(cluster_ids))
    rng.shuffle(shuffle_idxs)
    cluster_ids = cluster_ids[shuffle_idxs]
    snippets = snippets[shuffle_idxs]
    # Fill array with rec_id
    rec_ids = np.full(len(cluster_ids), rec_id)
    # Create 2 column array, recording ids and cluster ids.
    # ([1, 1, 1, 1], [1, 2, 3, 4]) -> [[1, 1], [1, 2], [1, 3], [1, 4]]
    rec_clusters = np.vstack((rec_ids, cluster_ids)).T 

    # Save the snippets and ids across multiple files.
    def save_splits():
        ids_split = np.array_split(rec_clusters, snippets_per_file)
        snippets_split = np.array_split(snippets, snippets_per_file)
        assert len(ids_split) == len(snippets_split)
        for i, (ids, ks) in enumerate(zip(ids_split, snippets_split)):
            _write_snippet_data_part(ids, ks, rec_dir, i)
    save_splits()


def _write_snippet_data_part(ids, snippets, save_dir, part_id):
    """
    Saves a subset of the spike snippets.

    This is used internally to spread the snippets data over multiple files.
    """
    ids_filename = IDS_FILE_PATTERN.format(part=part_id)
    win_filename = SNIPPET_FILE_PATTERN.format(part=part_id)
    ids_save_path = pathlib.Path(save_dir) / ids_filename
    win_save_path = pathlib.Path(save_dir) / win_filename
    with ids_save_path.open('wb') as f:
        np.save(f, ids)
    with win_save_path.open('wb') as f:
        np.save(f, snippets)
    return ids_save_path, win_save_path


def spike_snippets(up_stimulus: pd.DataFrame, response: pd.DataFrame, 
        rec_name: str, snippet_len: int = 1000, snippet_pad: int = 200, 
        sampling_freq: float=1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract all spike snippets for a single recording.

    Args:
        up_stimulus: the stimulus. If you want the stimulus upsampled, do it 
            before calling this method, as it isn't done here.
        response: the response dataframe.
        rec_name: the name of the recording to extract snippets for.
        snippet_len: the length of the snippet.
        snippet_pad: the number of timesteps to include after the spike.
        sampling_freq: the sampling frequency of the *stimulus*.
    
    Returns: two np.ndarrays as a tuple. The first element contains 
    the spike snippets, and the second element contains ids of the clusters.
    """
    response = response.xs(rec_name, level='Recording').reset_index(
            level='Cell index').drop('Kernel', axis=1, errors='ignore')

    def create_snippets(spikes) -> List[np.ndarray]:
        """Create the snippets for a list of spikes."""
        snips = []
        for spike_frame in spikes.compressed():
            w = stimulus_slice(up_stimulus, sampling_freq, spike_frame, 
                    snippet_len, snippet_pad)
            snips.append(w)
        return snips

    # Gather the snippets and align them with a matching array of cluster ids.
    snippets = []
    cluster_ids = []
    for row_as_tuple in response.itertuples(index=False):
        cluster_id, spikes = row_as_tuple
        snips = create_snippets(spikes)
        cluster_ids.extend([cluster_id,]*len(snips))
        snippets.extend(snips)
    snippets = np.stack(snippets)
    cluster_ids = np.array(cluster_ids)
    return snippets, cluster_ids


