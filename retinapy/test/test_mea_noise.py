from numpy.random.mtrand import sample
import pytest
import retinapy.mea_noise as mea
import numpy as np
import numpy.ma as ma
import numpy.testing
import pandas as pd
import pickle


FF_STIMULUS_PATH = './data/ff_noise.h5'
FF_SPIKE_RESPONSE_PATH = './data/ff_spike_response.pickle'


def test_load_fullfield_stimulus():
    noise = mea.load_fullfield_stimulus(FF_STIMULUS_PATH)
    known_shape = (24000, 4)
    assert noise.shape == known_shape


def test_load_response():
    response = mea.load_response(FF_SPIKE_RESPONSE_PATH)
    known_index_names = ['Cell index', 'Stimulus ID', 'Recording']
    assert response.index.names == known_index_names
    known_shape = (4417, 2)
    assert response.shape == known_shape


@pytest.fixture
def stimulus_data():
    return mea.load_fullfield_stimulus(FF_STIMULUS_PATH)


@pytest.fixture
def response_data():
    return mea.load_response(FF_SPIKE_RESPONSE_PATH)


def test_recording_names(response_data):
    known_list = [
            'Chicken_04_08_21_Phase_01',
            'Chicken_04_08_21_Phase_02',
            'Chicken_05_08_21_Phase_00',
            'Chicken_05_08_21_Phase_01',
            'Chicken_06_08_21_2nd_Phase_00',
            'Chicken_06_08_21_Phase_00',
            'Chicken_11_08_21_Phase_00',
            'Chicken_12_08_21_Phase_00',
            'Chicken_12_08_21_Phase_02',
            'Chicken_13_08_21_Phase_00',
            'Chicken_13_08_21_Phase_01',
            'Chicken_14_08_21_Phase_00',
            'Chicken_17_08_21_Phase_00',
            'Chicken_19_08_21_Phase_00',
            'Chicken_19_08_21_Phase_01',
            'Chicken_20_08_21_Phase_00',
            'Chicken_21_08_21_Phase_00']  
    rec_list = mea.recording_names(response_data)
    assert rec_list == known_list


def test_cluster_ids(response_data):
    known_list = [12, 13, 14, 15, 17, 25, 28, 29, 34, 44, 45, 50, 60, 61, 80, 
				  82, 99, 114, 119, 149, 217, 224, 287, 317, 421, 553, 591]
    recording_name = 'Chicken_21_08_21_Phase_00'
    cluster_ids = mea.cluster_ids(response_data, recording_name)
    assert cluster_ids == known_list


def test_upsample_stimulus_exceptions():
    # Expect exception if zoom is less than 1:
    with pytest.raises(ValueError):
        mea.upsample_stimulus(np.array([0, 1, 0, 0, 1]), 0)
    with pytest.raises(ValueError):
        mea.upsample_stimulus(np.array([0, 1, 0, 0, 1]), -1)
    # Expect exception if zoom is not integer.
    with pytest.raises(ValueError):
        mea.upsample_stimulus(np.array([0, 1, 0, 0, 1]), 1.5)


def test_upsample_stimulus():
    square_stimulus_RGBU = np.array(
            [[0, 0, 1, 1], 
             [1, 1, 0, 0], 
             [0, 0, 1, 1], 
             [1, 1, 0, 0], 
             [0, 0, 1, 1], 
             [1, 1, 0, 0], 
             [0, 0, 1, 1], 
             [1, 1, 0, 0]])
    up_stimulus = mea.upsample_stimulus(square_stimulus_RGBU, factor=2)
    expected_shape = (16, 4)
    assert up_stimulus.shape == expected_shape
    # No hard edges:
    for i in range(up_stimulus.shape[0]-1):
        mdiff = up_stimulus[i] - up_stimulus[i+1]
        assert np.max(up_stimulus[i] - up_stimulus[i+1]) < 0.9


def test_spike_snippets():
    """
    Test stimulus_slice function.

    The focus is on testing the shape, position and padding of the slice.

    Note that the slicing function doesn't do any filtering, so we can 
    use numpy.assert_equals, as the array values will not be modified.
    """
    # Setup
    # -----
    stim_frame_of_spike = [4, 1, 0, 6, 7]
    # The numbers in comments refer to the 5 tests below.
    stimulus = np.array(
            [[1, 1, 1, 1], #     |  2
             [1, 1, 1, 1], #     1  |
             [0, 1, 1, 1], #  -  |  - 
             [0, 0, 1, 1], #  |  - 
             [0, 0, 0, 1], #  0       -
             [0, 0, 0, 0], #  |       |  -
             [1, 0, 0, 0], #  -       3  |
             [1, 1, 0, 0]] #          |  4
        )
    total_len = 5
    pad = 2
    stimulus_sample_rate = 40
    # A bit of reverse engineering to get the sensor frame of the spike.
    def stimulus_frame_to_spikes(stimulus_frame):
        frame_width_in_sensor_samples = \
            mea.ELECTRODE_FREQ / stimulus_sample_rate
        first_spike = stimulus_frame * frame_width_in_sensor_samples 
        spikes_in_frame = np.arange(first_spike, 
                first_spike + frame_width_in_sensor_samples)
        return spikes_in_frame

    # Test 0
    # ------
    # Case where no padding is needed.
    snippets = mea.spike_snippets(
            stimulus,
            stimulus_frame_to_spikes(stim_frame_of_spike[0]),
            total_len,
            pad,
            stimulus_sample_rate,
            mea.ELECTRODE_FREQ)
    for s in snippets:
        expected_slice = np.array([
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [1, 0, 0, 0]])
        assert s.shape == expected_slice.shape
        numpy.testing.assert_allclose(s, expected_slice)

    # TODO: shouldn't they be filtered?

    # Test 1
    # ------
    # Sample is near the beginning and needs padding.
    snippets = mea.spike_snippets(
            stimulus,
            stimulus_frame_to_spikes(stim_frame_of_spike[1]),
            total_len,
            pad,
            stimulus_sample_rate,
            mea.ELECTRODE_FREQ)
    for s in snippets:
        expected_slice = np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1]])
        assert s.shape == expected_slice.shape
        numpy.testing.assert_allclose(s, expected_slice)

    # Test 2
    # ------
    # Sample is _at_ the beginning and needs padding.
    snippets = mea.spike_snippets(
            stimulus,
            stimulus_frame_to_spikes(stim_frame_of_spike[2]),
            total_len,
            pad,
            stimulus_sample_rate,
            mea.ELECTRODE_FREQ)
    for s in snippets:
        expected_slice = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 1]])
        assert s.shape == expected_slice.shape
        numpy.testing.assert_allclose(s, expected_slice)

    # Test 3
    # ------
    # Sample is near the end and needs padding.
    snippets = mea.spike_snippets(
            stimulus,
            stimulus_frame_to_spikes(stim_frame_of_spike[3]),
            total_len,
            pad,
            stimulus_sample_rate,
            mea.ELECTRODE_FREQ)
    for s in snippets:
        expected_slice = np.array([
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0]])
        assert s.shape == expected_slice.shape
        numpy.testing.assert_allclose(s, expected_slice)

    # Test 4
    # ------
    # Sample is _at_ the end and needs padding.
    snippets = mea.spike_snippets(
            stimulus,
            stimulus_frame_to_spikes(stim_frame_of_spike[4]),
            total_len,
            pad,
            stimulus_sample_rate,
            mea.ELECTRODE_FREQ)
    for s in snippets:
        expected_slice = np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])
        assert s.shape == expected_slice.shape
        numpy.testing.assert_equal(s, expected_slice)


def test_save_recording_names(tmp_path, response_data):
    rec_names = mea.recording_names(response_data)
    path = mea._save_recording_names(rec_names, tmp_path)
    expected_path = tmp_path / mea.REC_NAMES_FILENAME
    assert path == expected_path
    assert expected_path.is_file()
    with open(expected_path, 'rb') as f:
        contents = pickle.load(f)
    assert contents == rec_names


def test_save_cluster_ids(tmp_path, response_data):
    rec_id = 3
    cluster_ids = mea.cluster_ids(response_data, 
            mea.recording_names(response_data)[rec_id])
    path = mea._save_cluster_ids(cluster_ids, rec_id, tmp_path)
    expected_path = tmp_path / str(rec_id) / mea.CLUSTER_IDS_FILENAME
    assert path == expected_path
    assert expected_path.is_file()
    with open(expected_path, 'rb') as f:
        contents = pickle.load(f)
    assert contents == cluster_ids


def test_labeled_spike_snippets():
    """
    Create a fake response `DataFrame` and check that the spike snippets are
    calculated correctly.
    """
    # Setup
    snippet_len = 7
    snippet_pad = 2
    stim_sample_freq = mea.STIMULUS_FREQ * 2
    # Fake stimulus. Note that we are did our own nieve upsampling here. 
    # This is done to make the comparison easier.
    stimulus_up = np.array([
            [0,0,0,1], # 0
            [0,0,1,0], # 1
            [0,0,1,1], # 2
            [0,1,0,0], # 3
            [0,1,0,1], # 4
            [0,1,1,0], # 5
            [0,1,1,1], # 6
            [1,0,0,0], # 7
            [1,0,0,1], # 8
            [1,0,1,0], # 9
            [1,0,1,1], # 10
            [1,1,0,0], # 11
            ])
    # Fake response
    rec_name1 = 'Chicken_04_08_21_Phase_01'
    rec_name2 = 'Chicken_04_08_21_Phase_02'
    index = pd.MultiIndex.from_tuples(
            ((25, 1, 'Chicken_04_08_21_Phase_01'),
            (40, 1, 'Chicken_04_08_21_Phase_01'),
            (17, 1, 'Chicken_04_08_21_Phase_02'),
            (40, 1, 'Chicken_04_08_21_Phase_02')), 
            names=['Cell index', 'Stimulus ID', 'Recording'])
    data = [(None, [820, 3648]), (None, [3044, 4067]), (None, [5239,]), 
            (None, [4430,])]
    # Convert the data to use masked arrays.
    data_m = [(kernel, ma.array(spikes, mask=np.zeros(len(spikes)))) 
        for kernel, spikes in data]

    response = pd.DataFrame(data_m, index=index, columns=['Kernel', 'Spikes'])
    # The following is the predicted snippets.
    expected_spike_snippets1 = np.array([
            [
                [0,0,0,0],  # pad
                [0,0,0,0],  # pad
                [0,0,0,0],  # pad
                [0,0,0,1],  # 0
                [0,0,1,0],  # 1 <-- spike
                [0,0,1,1],  # 2 
                [0,1,0,0],  # 3 
            ],         
            [          
                [0,1,0,1], # 4
                [0,1,1,0], # 5
                [0,1,1,1], # 6
                [1,0,0,0], # 7
                [1,0,0,1], # 8 <- spike
                [1,0,1,0], # 9
                [1,0,1,1], # 10
            ],         
            [          
                [0,0,1,1], # 2 
                [0,1,0,0], # 3
                [0,1,0,1], # 4
                [0,1,1,0], # 5
                [0,1,1,1], # 6 <- spike
                [1,0,0,0], # 7
                [1,0,0,1], # 8
            ],         
            [          
                [0,1,1,0], # 5
                [0,1,1,1], # 6
                [1,0,0,0], # 7
                [1,0,0,1], # 8
                [1,0,1,0], # 9 <- spike
                [1,0,1,1], # 10
                [1,1,0,0], # 11
            ]
        ])         
    expected_spike_snippets2 = np.array([
            [          
                [1,0,0,0], # 7
                [1,0,0,1], # 8
                [1,0,1,0], # 9
                [1,0,1,1], # 10
                [1,1,0,0], # 11 <- spike
                [0,0,0,0], # pad
                [0,0,0,0], # pad
            ],                 
            [                  
                [0,1,1,0], # 5
                [0,1,1,1], # 6
                [1,0,0,0], # 7
                [1,0,0,1], # 8
                [1,0,1,0], # 9 <- spike
                [1,0,1,1], # 10
                [1,1,0,0], # 11
            ]
        ])
    expected_cluster_ids1 = np.array([25, 25, 40, 40])
    expected_cluster_ids2 = np.array([17, 40])

    # The spike times were calculated according to the following function.
    # Then, the slices of the stimulus were manually copied, and zoomed by 2,
    # as the stimulus was sampled at 40 Hz (twice the stimulus frequency).
    def create_ans():
        def spike_to_frame(sp):
            return sp * (stim_sample_freq / mea.ELECTRODE_FREQ)
        res = []
        for d in data:
            d_spikes = d[1]
            for sp in d_spikes:
                res.append(spike_to_frame(sp))
        print(res)

    # Test 1 (rec_name1)
    spike_snippets, cluster_ids = mea.labeled_spike_snippets(stimulus_up, 
            response, rec_name1, snippet_len, snippet_pad, stim_sample_freq)
    for idx, (spwin, cluster_ids) in enumerate(
            zip(spike_snippets, cluster_ids)):
        np.testing.assert_equal(spwin, expected_spike_snippets1[idx])
        np.testing.assert_equal(cluster_ids, expected_cluster_ids1[idx])

    # Test 2 (rec_name2)
    spike_snippets, cluster_ids = mea.labeled_spike_snippets(stimulus_up, 
            response, rec_name2, snippet_len, snippet_pad, stim_sample_freq)
    for idx, (spwin, cluster_ids) in enumerate(
            zip(spike_snippets, cluster_ids)):
        np.testing.assert_equal(spwin, expected_spike_snippets2[idx])
        np.testing.assert_equal(cluster_ids, expected_cluster_ids2[idx])


def test_write_rec_snippets(tmp_path, response_data, stimulus_data):
    # Setup
    stimulus_zoom = 2
    orig_stimulus_freq = 20 # Hz
    stimulus_freq = orig_stimulus_freq * stimulus_zoom
    rec_id = 2
    snippet_len = 7
    snippet_pad = 1
    empty_snippets = 0
    snippets_per_file = 100
    up_stim = mea.upsample_stimulus(stimulus_data, stimulus_zoom)
    rec_name = mea.recording_names(response_data)[rec_id]
    # Test
    # TODO: add some more checks.
    # Currently, the test only checks that the method runs to completion.
    mea._write_rec_snippets(up_stim, response_data, rec_name, rec_id, tmp_path, 
            snippet_len, snippet_pad, stimulus_freq, empty_snippets, 
            snippets_per_file)

