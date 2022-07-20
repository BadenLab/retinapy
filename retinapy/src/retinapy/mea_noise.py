from collections import namedtuple
from collections import defaultdict
import itertools
import logging
import math
import pathlib
import pickle
import tempfile
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import scipy.signal

import h5py


ELECTRODE_FREQ = 17852.767845719834  # Hz
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


def load_stimulus_pattern(file_path: str) -> np.ndarray:
    """
    Loads the stimulus data from the HDF5 file as a Pandas DataFrame.

    Dataframe structure
    -------------------
        - integer index, representing stimulus frames.
        - columns are ['red', 'green', 'blue', 'uv']
        - values are 0 or 1 representing whether the corresponding LED is ON
          or OFF at the given stimulus frame.
    """
    with h5py.File(file_path, 'r') as f:
        # The data has shape: [4, 24000, 10374]. This corresponds to 4 lights,
        # on-off pattern for 20min at 20 Hz (24000 periods), and 10374 boxes
        # arranged over 2D screen space. In the full-field experiment, only a
        # single box was used, and hence the [:,0] access pattern.
        colour_noise = np.array([
            f[stimuli[0].import_name][:, 0],
            f[stimuli[1].import_name][:, 0],
            f[stimuli[2].import_name][:, 0],
            f[stimuli[3].import_name][:, 0]]).transpose()
    stimulus_switch_freq = 20
    stimulus_loop_mins = 20
    stimulus_loop_secs = stimulus_loop_mins * 60
    assert colour_noise.shape[0] == stimulus_loop_secs * stimulus_switch_freq
    # Optional, if bottleneck encountered
    # ===================================
    # We are doing a lot of slicing in the time-step dimension, so keep it
    # as the row dimension. Also insure the array is contiguous for further
    # speed-ups.
    # res = np.ascontiguousarray(colour_noise)
    return colour_noise


def load_response(file_path: str, keep_kernels=False) -> pd.DataFrame:
    """Loads the spike data from a Pickle file as a Pandas DataFrame.

    The input path should point to standard pickle file (zipped or not).

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
    res = pd.read_pickle(file_path)
    if not keep_kernels:
        res.drop('Kernel', axis=1)
    return res


def load_recorded_stimulus(file_path: str) -> pd.DataFrame:
    res = pd.read_pickle(file_path, compression='infer')
    return res


def decompress_stimulus(stimulus_pattern: np.ndarray, 
        stimulus_df: pd.DataFrame, rec_name: str, 
        downsample_factor: int = 1) -> Tuple[np.ndarray, float]:
    row = stimulus_df.xs(rec_name, level='Recording')
    if len(row) != 1:
        raise ValueError('Expected exactly one row for recording ({rec_name}).'
                         f' Got ({len(row)}).')
    row = row.iloc[0]
    # TODO: is 'End_Fr' inclusive? If so, add 1 below. Assuming yes for now.
    num_samples = row['End_Fr'] - row['Begin_Fr'] + 1
    if num_samples != int(num_samples):
        raise ValueError('Expected integer number of samples. Got '
                         f'({num_samples}).')
    trigger_events = row['Trigger_Fr_relative']
    # Two very similar code paths:
    res = np.empty((num_samples, stimulus_pattern.shape[1]))
    _decompress_stimulus(stimulus_pattern, trigger_events, res)
    res = downsample_stimulus(res, downsample_factor)

    # Optional, if bottleneck encountered
    # ===================================
    # We are doing a lot of slicing in the time-step dimension, so keep it
    # as the row dimension. Also insure the array is contiguous for further
    # speed-ups.
    # res = np.ascontiguousarray(res)

    # Note: the sampling frequency in the Pandas dataframe isn't as accurate
    # as the ELECTRODE_FREQ value.
    new_freq = row['Sampling_Freq'] / downsample_factor
    return res, new_freq


def _decompress_stimulus(
        stimulus_pattern: np.ndarray,
        trigger_events: np.ndarray,
        out_array: np.ndarray):
    """TODO: take in the stimulus trigger points."""
    logging.info('Starting: decompressing stimulus. Resulting shape '
                 f'({out_array.shape}).')
    # If this becomes a bottleneck, there are some tricks to reach for:
    # https://stackoverflow.com/questions/60049171/fill-values-in-numpy-array-that-are-between-a-certain-value
    slices = np.stack((trigger_events[:-1], trigger_events[1:]), axis=1)
    for idx, s in enumerate(slices):
        out_array[np.arange(*s)] = stimulus_pattern[idx]
    logging.info(f'Finished: decompressing stimulus. '
                 f'The last trigger was at ({trigger_events[-1]}) making its '
                 f'duration ({out_array.shape[0]} - {trigger_events[-1]} = '
                 f'{out_array.shape[0] - trigger_events[-1]}) samples.')
    return out_array


def decompress_spikes(spikes: np.ma.MaskedArray, num_sensor_samples: int, 
        downsample_factor: int = 1) -> np.ndarray:
    spikes = spikes.compressed()
    out = np.empty(
            shape=[math.ceil(num_sensor_samples/downsample_factor),], dtype=int)
    spikes = np.floor_divide(spikes, downsample_factor)
    out[spikes] = 1
    return spikes


def factors_sorted_by_count(n, limit) -> List[Tuple[int,...]]:
    """
    Calulates factor decomposition with sort and limit.

    This method is used to choose downsampling factors when a single factor
    is too large. The decompositions are sorted by the number of factors in a
    decomposition.  
    """
    def _factors(n):
        res = [(n,)]
        f1 = n // 2
        while(f1 > 1):
            f2, mod = divmod(n,f1)
            if not mod:
                res.append((f1, f2)) 
                sub_factors = [a+b for (a,b) in
                        itertools.product(_factors(f1), _factors(f2))]
                res.extend(sub_factors)
            f1 -= 1
        return res
    factors = list(set(_factors(n)))
    factors_under = [f for f in factors if max(f) <= limit]
    sorted_by_count = sorted(factors_under, key = lambda x : len(x))
    return sorted_by_count


def downsample_stimulus(stimulus: np.ndarray, factor: int) -> np.ndarray:
    """
    Filter (low-pass) a stimulus and then decimate by a factor.

    This is needed to prevent aliasing.

    Resources on filtering
    ----------------------
    https://dsp.stackexchange.com/questions/45446/pythons-tt-resample-vs-tt-resample-poly-vs-tt-decimate
    https://dsp.stackexchange.com/questions/83696/downsample-a-signal-by-a-non-integer-factor
    https://dsp.stackexchange.com/questions/83889/decimate-a-signal-whose-values-are-calculated-not-stored?noredirect=1#comment176944_83889
    """
    if factor == 1:
        return stimulus
    time_axis = 0
    logging.info('Starting: downsampling')
    # SciPy recommends to never exceed 13 on a single decimation call.
    # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html
    MAX_SINGLE_DECIMATION = 13
    sub_factors = factors_sorted_by_count(factor, 
            limit=MAX_SINGLE_DECIMATION)[0]
    for sf in sub_factors:
        logging.info(f'Starting: decimating by {sf}')
        stimulus = scipy.signal.decimate(stimulus, sf, ftype='fir',
                axis=time_axis)
        logging.info(f'Finished: decimating by {sf}')
    logging.info('Finished: downsampling')
    return stimulus


def recording_names(response: pd.DataFrame) -> list:
    """Return the list of recording names."""
    rec_list = response.index.get_level_values('Recording').unique().tolist()
    return rec_list


def _list_to_index_map(l) -> Dict[str, int]:
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
    stimulus_id = 1  # not 7.
    cluster_ids = response.xs((stimulus_id, rec_name), 
            level=('Stimulus ID', 'Recording'), drop_level=True) \
                    .index.get_level_values('Cell index').unique().tolist()
    return cluster_ids


def spike_window(spikes: Union[int, np.ndarray], total_len: int,
                 post_spike_len: int, stimulus_sample_rate: float,
                 sensor_sample_rate: float):
    """
    Calculate the window endpoints around a spike, in samples of the stimulus.
    """
    if sensor_sample_rate < stimulus_sample_rate:
        logging.warn(f'The sensor sample rate ({sensor_sample_rate}) is lower '
                     f'than the stimulus sample rate ({stimulus_sample_rate}).')
    if total_len < post_spike_len + 1:
        raise ValueError(f'Total length must be at least 1 greater than the '
                         f'post-spike lengths + 1. Got total_len ({total_len}) '
                         f'and post_spike_len ({post_spike_len}).')
    # 1. Select the sample to represent the spike.
    stimulus_frame_of_spike =  \
        np.floor(spikes * (stimulus_sample_rate / sensor_sample_rate)) \
        .astype('int')
    # 2. Calculate the snippet start and end.
    # The -1 appears as we include the spike sample in the snippet.
    win_start = (stimulus_frame_of_spike + post_spike_len) - (total_len - 1)
    win_end = stimulus_frame_of_spike + post_spike_len + 1
    return win_start, win_end


def spike_snippets(stimulus: np.ndarray, spikes: np.ndarray, 
                   stimulus_sample_rate: float,
                   sensor_sample_rate: float, total_len: int, 
                   post_spike_len: int) -> np.ndarray:
    """
    Return a subset of the stimulus around the spike points.

    Args:
        stimulus: The stimulus.
        spikes: The sensor samples in which the spikes were detected.
        total_len: The length of the snippet in stimulus frames. 
        post_spike_len: The number of frames to pad after the spike.
        stimulus_sample_rate: The sampling rate of the stimulus.
        sensor_sample_rate: The sampling rate of the sensor.
    Returns:
        A Numpy array of shape (spikes.shape[0], total_len, NUM_STIMULUS_LEDS).

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
    # 1. Get the spike windows.
    win_start, _ = spike_window(spikes, total_len, post_spike_len,
                                stimulus_sample_rate, sensor_sample_rate)
    # 2. Pad the stimulus incase windows go out of range.
    if np.any(win_start < 0) or \
            np.any(win_start >= (stimulus.shape[0] - total_len)):
        stimulus = np.pad(stimulus, ((total_len, total_len), (0, 0)),
                          'constant')
        # 3. Offset the windows, which is needed due to the padding.
        win_start += total_len
    # 4. Extract the slice.
    # The padded_stimulus is indexed by a list arrays of the form:
    #    (win_start[0], win_start[0]+1, win_start[0]+2, ..., win_start[0]+total_len)
    #    (win_start[1], win_start[1]+1, win_start[1]+2, ..., win_start[1]+total_len)
    #    ...
    #    (win_start[num_spikes-1], win_start[num_spikes-1]+1, win_start[num_spikes-1]+2, ..., win_start[num_spikes-1]+total_len)
    snippets = stimulus[np.asarray(win_start)[:, None] + np.arange(total_len)]
    return snippets


def labeled_spike_snippets(
        stimulus: np.ndarray, 
        response: pd.DataFrame,
        rec_name: str, 
        stimulus_sample_rate,
        sensor_sample_rate,
        snippet_len: int = 1000,
        snippet_pad: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Caluclate spike snippets for a recording, paired with the cluster ids.

    Args:
        stimulus: the stimulus, sampled at `stimulus_sample_rate`.
        response: the response dataframe.
        rec_name: the name of the recording to extract snippets for.
        snippet_len: the length of the snippet.
        snippet_pad: the number of timesteps to include after the spike.
        stimulus_sample_rate: the sampling frequency of the stimulus.
        sensor_sample_rate: the sampling frequency of the sensor.
    
    Returns: two np.ndarrays as a tuple. The first element contains 
    the spike snippets, and the second element contains ids of the clusters.
    """
    response = response.xs(rec_name, level='Recording').reset_index(
        level='Cell index').drop('Kernel', axis=1, errors='ignore')

    def create_snippets(spikes) -> List[np.ndarray]:
        """Create the snippets for a list of spikes."""
        snips = []
        w = spike_snippets(stimulus, spikes.compressed(), stimulus_sample_rate,
                           sensor_sample_rate, snippet_len, snippet_pad)
        snips.extend(w)
        return snips

    # Gather the snippets and align them with a matching array of cluster ids.
    snippets = []
    cluster_ids = []
    for row_as_tuple in response.itertuples(index=False):
        cluster_id, spikes = row_as_tuple
        snips = create_snippets(spikes)
        cluster_ids.extend([cluster_id, ]*len(snips))
        snippets.extend(snips)
    snippets = np.stack(snippets)
    cluster_ids = np.array(cluster_ids)
    return snippets, cluster_ids


def generate_fake_spikes(spikes, num_fake_per_real,
                         target_sample_rate,
                         sensor_sample_rate,
                         min_dist_to_real_spike,
                         rng_seed=123) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genrate fake spikes.

    Spread the spikes over the duration of the recording. Don't generate
    a fake spike too close to a real spike.

    Between every two spikes, `num_fake_per_real` spikes will be generated.
    Thus, total number of fake spikes will be 

        len(spikes) * `num_fake_per_real` -1.
    
    Some details:

        spikes:     |--------*------------*----------------------*-----------|
        intervals shown by |-----| in which we can create fake spikes:
                     |       * |--------| * |------------------| *           |

    There is a buffer between the spikes and the intervals, this is
    the `min_dist_to_real_spike`.

    Within the intervals, we will try to create `snippets_per_spike` number
    of fake spikes. From these spikes, snippets will be created. 

    The spikes are not chosen evenly within the intervals, but rather
    there is some jitter. This is what the `rng_seed` parameter is for.

    Args:
        spikes: a sorted (ascending) list of spikes, in sensor samples.
        num_fake_per_real: the number of fake spikes to generate between every
            two real spikes.
        target_sample_rate: the target sampling rate.
        sensor_sapmle_rate: the sampling rate of the sensor.
    """
    if len(spikes) < 2:
        raise ValueError(f'Need at least two spikes. Got ({len(spikes)}).')
    # Convert spikes to the target sample space.

    spikes = np.ndarray(spikes) * (target_sample_rate / sensor_sample_rate)

    intervals = np.stack(spikes[:-1] + min_dist_to_real_spike,
                         spikes[1:] - min_dist_to_real_spike)
    sub_intervals = []

    def split_interval(interval, n):
        """Split an interval into n subintervals, of roughly equal size."""
        d, m = divmod(interval, n)
        return [(i*d+min(i, m), (i+1)*d+min(i+1, m)) for i in range(n)]
    for i in intervals:
        sub_intervals.extend(split_interval(i, num_fake_per_real))

    rng = np.random.default_rng(rng_seed)
    fake_spikes = rng.random.integers(*np.stack(sub_intervals, axis=1),
                                      endpoints=True)
    return fake_spikes


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
    save_path = pathlib.Path(data_root_dir) / \
        str(rec_id) / 'cluster_ids.pickle'
    # Create out file
    with save_path.open('wb') as f:
        pickle.dump(cell_cluster_ids, f)
    return save_path


def create_snippet_training_data(
        stimulus_pattern: np.ndarray,
        recorded_stimulus: pd.DataFrame,
        response: pd.DataFrame,
        save_dir,
        downsample_factor: int = 18,
        snippet_len: int = 1000,
        snippet_pad: int = 200,
        empty_snippets: float = 0,
        snippets_per_file: int = 1024):
    # TODO: WIP
    save_dir = pathlib.Path(save_dir)
    _save_recording_names(recording_names(response), save_dir)
    # TODO: only use stimulus 1? What is stimulus 7?
    by_rec = response.xs(1, level='Stimulus ID',
                         drop_level=True).groupby('Recording')
    for rec_id, (rec_name, df) in enumerate(by_rec):
        stimulus, freq = decompress_and_decimate(stimulus_pattern,
                recorded_stimulus, rec_name, downsample_factor)
        _write_rec_snippets(stimulus, df, rec_name, rec_id, save_dir,
                            snippet_len, snippet_pad, freq, empty_snippets,
                            snippets_per_file)


def _write_rec_snippets(
        stimulus: np.ndarray,
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
    snippets, cluster_ids = labeled_spike_snippets(
        stimulus, response, rec_name, stimulus_freq, ELECTRODE_FREQ, 
        snippet_len, snippet_pad)
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
