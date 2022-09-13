import logging
import math
import pickle

import numpy as np
import numpy.ma as ma
import numpy.testing
import pandas as pd

import pytest
import retinapy.mea as mea


FF_NOISE_PATTERN_PATH = "./data/ff_noise.h5"
FF_SPIKE_RESPONSE_PATH = "./data/ff_spike_response.pickle"
FF_SPIKE_RESPONSE_PATH_ZIP = "./data/ff_spike_response.pickle.zip"
FF_RECORDED_NOISE_PATH = "./data/ff_recorded_noise.pickle"
FF_RECORDED_NOISE_PATH_ZIP = "./data/ff_recorded_noise.pickle.zip"


def test_load_stimulus_pattern():
    noise = mea.load_stimulus_pattern(FF_NOISE_PATTERN_PATH)
    known_shape = (24000, 4)
    assert noise.shape == known_shape


def test_load_response():
    for path in (FF_SPIKE_RESPONSE_PATH, FF_SPIKE_RESPONSE_PATH_ZIP):
        response = mea.load_response(path)
        known_index_names = ["Cell index", "Stimulus ID", "Recording"]
        assert response.index.names == known_index_names
        known_shape = (4417, 2)
        assert response.shape == known_shape


def test_load_recorded_stimulus():
    for path in (FF_RECORDED_NOISE_PATH, FF_RECORDED_NOISE_PATH_ZIP):
        res = mea.load_recorded_stimulus(path)
        known_index_names = ["Stimulus_index", "Recording"]
        assert res.index.names == known_index_names
        known_shape = (18, 8)
        assert res.shape == known_shape


@pytest.fixture
def stimulus_pattern():
    return mea.load_stimulus_pattern(FF_NOISE_PATTERN_PATH)


@pytest.fixture
def recorded_stimulus():
    return mea.load_recorded_stimulus(FF_RECORDED_NOISE_PATH)


@pytest.fixture
def response_data():
    return mea.load_response(FF_SPIKE_RESPONSE_PATH)


@pytest.fixture
def comp_exp0(stimulus_pattern, recorded_stimulus, response_data):
    exp = mea.single_3brain_recording(
        "Chicken_04_08_21_Phase_01",
        stimulus_pattern,
        recorded_stimulus,
        response_data,
    )
    return exp


@pytest.fixture
def comp_exp12(stimulus_pattern, recorded_stimulus, response_data):
    exp = mea.single_3brain_recording(
        "Chicken_17_08_21_Phase_00",
        stimulus_pattern,
        recorded_stimulus,
        response_data,
    )
    return exp


@pytest.fixture
def exp0(comp_exp0):
    downsample = 18
    exp = mea.decompress_recording(comp_exp0, downsample)
    return exp


@pytest.fixture
def exp12(comp_exp12):
    downsample = 18
    exp = mea.decompress_recording(comp_exp12, downsample)
    return exp


def test_recording_names(response_data):
    known_list = [
        "Chicken_04_08_21_Phase_01",
        "Chicken_04_08_21_Phase_02",
        "Chicken_05_08_21_Phase_00",
        "Chicken_05_08_21_Phase_01",
        "Chicken_06_08_21_2nd_Phase_00",
        "Chicken_06_08_21_Phase_00",
        "Chicken_11_08_21_Phase_00",
        "Chicken_12_08_21_Phase_00",
        "Chicken_12_08_21_Phase_02",
        "Chicken_13_08_21_Phase_00",
        "Chicken_13_08_21_Phase_01",
        "Chicken_14_08_21_Phase_00",
        "Chicken_17_08_21_Phase_00",
        "Chicken_19_08_21_Phase_00",
        "Chicken_19_08_21_Phase_01",
        "Chicken_20_08_21_Phase_00",
        "Chicken_21_08_21_Phase_00",
    ]
    rec_list = mea.recording_names(response_data)
    assert rec_list == known_list


def test_cluster_ids(response_data):
    # fmt: off
    known_list = [12, 13, 14, 15, 17, 25, 28, 29, 34, 44, 45, 50, 60, 61, 80,
                  82, 99, 114, 119, 149, 217, 224, 287, 317, 421, 553, 591]
    # fmt: on
    recording_name = "Chicken_21_08_21_Phase_00"
    cluster_ids = mea._cluster_ids(response_data, recording_name)
    assert cluster_ids == known_list


def test_load_3brain_recordings():
    # Test
    res = mea.load_3brain_recordings(
        FF_NOISE_PATTERN_PATH,
        FF_RECORDED_NOISE_PATH,
        FF_SPIKE_RESPONSE_PATH,
        include=[
            "Chicken_04_08_21_Phase_01",
            "Chicken_04_08_21_Phase_02",
        ],
    )
    assert len(res) == 2


def test_split(exp12):
    """Tests splitting a recording into multiple parts.

    Tests that:
        1. A split works.
        2. zero-value ratio causes an error.
    """
    # Test 1
    # Setup
    splits = (3, 1, 1)
    expected_len = 892863
    assert len(exp12) == expected_len
    expected_split_lens = [535716 + 3, 178572, 178572]
    expected_split_lens_reversed = [178572 + 3, 178572, 535716]
    assert len(exp12) == sum(expected_split_lens)
    # Test
    res = mea.split(exp12, splits)
    assert len(res) == 3, "There should be 3 splits."
    assert [
        len(s) for s in res
    ] == expected_split_lens, "Splits should be the correct length."
    # Do it again but reversed.
    res = mea.split(exp12, splits[::-1])
    assert len(splits) == 3, "There should be 3 splits."
    assert [
            len(s) for s in res
            ] == expected_split_lens_reversed, "Splits should be the correct length."

    # Test 2
    with pytest.raises(ValueError):
        mea.split(exp12, (0, 1, 1))


def test_mirror_split(exp12):
    """Tests splitting a recording into multiple parts (the mirrored version).

    Tests that:
        1. A split works.
        2. zero-value ratio causes an error.
    """
    # Test 1
    # Setup
    splits = (3, 1, 1)
    expected_len = 892863
    assert len(exp12) == expected_len
    # Note how the remainders fall in different places compared to the 
    # non-mirrored split. This is due to the mirrored split calling split
    # twice, under the hood.
    expected_split_lens = [535716 + 1, 178572, 178572 + 2]
    expected_split_lens_reversed = [178572 + 1, 178572, 535716 + 2]
    assert len(exp12) == sum(expected_split_lens)
    # Test
    res = mea.mirror_split(exp12, splits)
    assert len(res) == 3, "There should be 3 splits."
    assert [
        len(s) for s in res
    ] == expected_split_lens, "Splits should be the correct length."
    # Do it again but reversed.
    res = mea.mirror_split(exp12, splits[::-1])
    assert len(splits) == 3, "There should be 3 splits."
    assert [
            len(s) for s in res
            ] == expected_split_lens_reversed, "Splits should be the correct length."

    # Test 2
    with pytest.raises(ValueError):
        mea.split(exp12, (0, 1, 1))


def test_decompress_recording(caplog, comp_exp12):
    """
    Test decompressing a recording.

    Tests that:
        1. Simple decompression works (basic checks).
        2. Multiple spikes in the same bucket is allowed.
    """
    # Test 1
    downsample = 18
    orig_freq = comp_exp12.sensor_sample_rate
    expected_sample_rate = orig_freq / downsample
    res = mea.decompress_recording(comp_exp12, downsample)
    assert res.sample_rate == pytest.approx(expected_sample_rate)

    # Test 2
    downsample = 100
    res = mea.decompress_recording(comp_exp12, downsample)
    max_per_bucket = np.max(res.spikes)
    assert max_per_bucket == 3


def test_single_3brain_recording(
    stimulus_pattern, recorded_stimulus, response_data
):
    """
    Tests that:
        1. A recording can be loaded without errors (very basic checks).
        2. Filtering by cluster id works.
        3. Requesting non-existing clusters raises an error.
    """
    # Test 1
    # Setup
    expected_num_samples = 16071532
    # Test
    rec = mea.single_3brain_recording(
        "Chicken_17_08_21_Phase_00",
        stimulus_pattern,
        recorded_stimulus,
        response_data,
    )
    assert rec.num_sensor_samples == expected_num_samples
    assert rec.stimulus_pattern.shape[1] == mea.NUM_STIMULUS_LEDS
    # Note: the sampling frequency in the Pandas dataframe isn't as accurate
    # as the ELECTRODE_FREQ value.
    assert rec.sensor_sample_rate == pytest.approx(mea.ELECTRODE_FREQ)

    # Test 2
    # Setup
    cluster_ids = {13, 14, 202, 1485}
    # Test
    rec = mea.single_3brain_recording(
        "Chicken_17_08_21_Phase_00",
        stimulus_pattern,
        recorded_stimulus,
        response_data,
        cluster_ids,
    )
    assert len(rec.spike_events) == len(cluster_ids)
    assert len(rec.cluster_ids) == len(cluster_ids)
    assert set(rec.cluster_ids) == cluster_ids

    # Test 3
    # Setup
    partially_existing_clusters = {
        13,  # Exists
        14,  # Exists
        18,  # Does not exist
    }
    # Test
    with pytest.raises(ValueError):
        mea.single_3brain_recording(
            "Chicken_17_08_21_Phase_00",
            stimulus_pattern,
            recorded_stimulus,
            response_data,
            partially_existing_clusters,
        )


def test_decompress_stimulus():
    """
    Tests that:
        1. Decompression with no downsample works.
        2. Decompression with downsample works.
            - This test overlaps a bit with test_downsample_stimulus
        3. Invalid trigger start should cause an exception.
    """
    # Setup
    # fmt: off
    stimulus_pattern = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0]]).T
    trigger_events = np.array([0, 4, 5, 7, 12])
    num_sensor_samples = 20
    expected_output = np.array([
    # idx: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
    # val: 0  0  0  0  1  2  2  3  3  3  3  3  4  4  4  4  4  4  4  4
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]).T
    expected_downsampled = np.array([
    # idx: 0     2     4     6     8     10    12    14    16    18
    # val: 0     0     1     2     3     3     4     4     4     4 
          [0.7, 1.1,   1,    1,    1,    1,  0.3,    0,    0,    0],
          [0.7, 1.1, 0.7,    0,    0,    0,  0.7,    1.1,  1,    1],
          [0.7,   1, 0.5,  0.7,    0,    0,  0.7,    1.1,  1,    1]]).T
    # fmt: on
    # Notice how the first downsampled value is pulled towards zero. This
    # is due to the "constant" option sent to scipy.resample_poly, which is
    # called by scipy.resample with "0" as the constant value. There are other
    # options, such as "mean" too. If we want a different constant value
    # or want "mean" padding, we should call scipy.resapmle_poly directly.

    # Test 1
    downsample = 1
    res = mea.decompress_stimulus(
        stimulus_pattern, trigger_events, num_sensor_samples, downsample
    )
    np.testing.assert_array_equal(res, expected_output)

    # Test 2
    downsample = 2
    res = mea.decompress_stimulus(
        stimulus_pattern, trigger_events, num_sensor_samples, downsample
    )
    # Test approx with numpy
    numpy.testing.assert_allclose(res, expected_downsampled, atol=0.1)

    # Test 3
    # Non-zero starting trigger is invalid.
    trigger_events_invalid = np.array([2, 4, 5, 7, 12])
    with pytest.raises(ValueError):
        mea.decompress_stimulus(
            stimulus_pattern,
            trigger_events_invalid,
            num_sensor_samples,
            downsample,
        )


def test_factors_sorted_by_count():
    """
    Tests that:
        1. A simple case:
            1. There are no duplicates.
            2. The factors are sorted by count.
        2. Setting a limit will cause the list to be truncated.
        3. If no factorizations meet the limit requirement, a fallback is
            returned.
        4. For a prime, the parameter is returned as is.
    """
    # Test 1
    num = 12
    expected = ((2, 2, 3), (2, 6), (3, 4), (12,))
    # Test 1
    res = mea.factors_sorted_by_count(num)
    assert set(res) == set(expected)

    # Test 2
    num = 12
    expected = ((2, 2, 3), (3, 4))
    res = mea.factors_sorted_by_count(num, limit=5)
    assert set(res) == set(expected)
    expected = ((2, 2, 3), (2, 6), (3, 4))
    res = mea.factors_sorted_by_count(num, limit=6)
    assert set(res) == set(expected), "The limit should be inclusive."

    # Test 3
    num = 2 * 2 * 13 * 13
    # Note: note how the result is not really ideal (13, 13, 4) would be better.
    expected = ((2, 2, 13, 13),)
    res = mea.factors_sorted_by_count(num, limit=5)
    assert set(res) == set(expected)

    # Test 4
    num = 89
    expected = ((89,),)
    res = mea.factors_sorted_by_count(num)
    assert res == expected


def test_downsample_stimulus():
    # Setup
    # fmt: off
    orig_signal = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1])
    # TODO: is this satisfactory?
    expected_decimate_by_2 = np.array(
            [0.012, -0.019,  0.029, -0.060,
             0.739,  1.085,  0.937,  1.081,
             0.249, -0.076,  0.058, -0.077,
             0.750,  1.077,  0.942,  1.077,
             0.250, -0.077,  0.058, -0.076,
             0.749,  1.081,  0.937,  1.085])
    expected_decimate_by_4 = np.array(
            [0.030, -0.065,  0.595,  1.148,
             0.364, -0.120,  0.620,  1.136,
             0.369, -0.120,  0.614,  1.148])
    # fmt: on

    # Test
    decimated_by_2 = mea.downsample_stimulus(orig_signal, 2)
    numpy.testing.assert_allclose(
        decimated_by_2, expected_decimate_by_2, atol=0.002
    )
    decimated_by_4 = mea.downsample_stimulus(orig_signal, 4)
    numpy.testing.assert_allclose(
        decimated_by_4, expected_decimate_by_4, atol=0.002
    )


def test_decompress_spikes1():
    # Setup
    downsample_by = 9
    num_sensor_samples = 123
    # fmt: off
    spike_times1 = np.array([8, 9, 30, 40, 50, 70, 80, 90, 100, 110])
    spike_times2 = np.array([0, 1, 8, 9, 10, 27, 30, 40, 50, 70, 80, 90, 100, 110])

    spike_counts1 = np.array([1, 1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  1,  1,  0])
    spike_counts2 = np.array([3, 2,  0,  2,  1,  1,  0,  1,  1,  0,  1,  1,  1,  0])
    # fmt: on
    expected_output_len = math.ceil(num_sensor_samples / downsample_by)
    assert len(spike_counts1) == len(spike_counts2) == expected_output_len

    # Test 1
    spikes = mea.decompress_spikes(
        spike_times1, num_sensor_samples, downsample_by
    )
    assert np.array_equal(spikes, spike_counts1)

    # Test 2
    # Test the case where two spikes land in the same bucket.
    # There should *not* be an error thrown, even though two samples land in
    # the same bucket.
    spikes = mea.decompress_spikes(
        spike_times2, num_sensor_samples, downsample_by
    )
    assert np.array_equal(spikes, spike_counts2)


def test_decompress_spikes2(response_data):
    """
    Test decompress_spikes on actual recordings.

    Not much is actually checked though.
    """
    # Setup
    cluster_id = 36
    spikes_row = response_data.xs(
        (cluster_id, "Chicken_17_08_21_Phase_00"),
        level=("Cell index", "Recording"),
    ).iloc[0]
    spikes = spikes_row["Spikes"].compressed()

    assert spikes.shape == (2361,)
    num_sensor_samples = spikes[-1] + 1000

    # Test
    mea.decompress_spikes(spikes, num_sensor_samples)
    mea.decompress_spikes(spikes, num_sensor_samples, downsample_factor=18)


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
    # fmt: off
    stimulus = np.array(
        [
            [1, 1, 1, 1],  #     |  2
            [1, 1, 1, 1],  #     1  |
            [0, 1, 1, 1],  #  -  |  -
            [0, 0, 1, 1],  #  |  -
            [0, 0, 0, 1],  #  0       -
            [0, 0, 0, 0],  #  |       |  -
            [1, 0, 0, 0],  #  -       3  |
            [1, 1, 0, 0],  #          |  4
        ]              
    )
    # fmt: on
    total_len = 5
    pad = 2
    stimulus_sample_rate = 40
    # A bit of reverse engineering to get the sensor frame of the spike.
    def stimulus_frame_to_spikes(stimulus_frame):
        frame_width_in_sensor_samples = (
            mea.ELECTRODE_FREQ / stimulus_sample_rate
        )
        first_spike = stimulus_frame * frame_width_in_sensor_samples
        spikes_in_frame = np.arange(
            first_spike, first_spike + frame_width_in_sensor_samples
        )
        return spikes_in_frame

    # Test 0
    # ------
    # Case where no padding is needed.
    snippets = mea.spike_snippets(
        stimulus,
        stimulus_frame_to_spikes(stim_frame_of_spike[0]),
        stimulus_sample_rate,
        mea.ELECTRODE_FREQ,
        total_len,
        pad,
    )
    for s in snippets:
        expected_slice = np.array(
            [
                [0, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        )
        assert s.shape == expected_slice.shape
        numpy.testing.assert_allclose(s, expected_slice)

    # TODO: shouldn't they be filtered?

    # Test 1
    # ------
    # Sample is near the beginning and needs padding.
    snippets = mea.spike_snippets(
        stimulus,
        stimulus_frame_to_spikes(stim_frame_of_spike[1]),
        stimulus_sample_rate,
        mea.ELECTRODE_FREQ,
        total_len,
        pad,
    )
    for s in snippets:
        expected_slice = np.array(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 0, 1, 1],
            ]
        )
        assert s.shape == expected_slice.shape
        numpy.testing.assert_allclose(s, expected_slice)

    # Test 2
    # ------
    # Sample is _at_ the beginning and needs padding.
    snippets = mea.spike_snippets(
        stimulus,
        stimulus_frame_to_spikes(stim_frame_of_spike[2]),
        stimulus_sample_rate,
        mea.ELECTRODE_FREQ,
        total_len,
        pad,
    )
    for s in snippets:
        expected_slice = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 1],
            ]
        )
        assert s.shape == expected_slice.shape
        numpy.testing.assert_allclose(s, expected_slice)

    # Test 3
    # ------
    # Sample is near the end and needs padding.
    snippets = mea.spike_snippets(
        stimulus,
        stimulus_frame_to_spikes(stim_frame_of_spike[3]),
        stimulus_sample_rate,
        mea.ELECTRODE_FREQ,
        total_len,
        pad,
    )
    for s in snippets:
        expected_slice = np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        assert s.shape == expected_slice.shape
        numpy.testing.assert_allclose(s, expected_slice)

    # Test 4
    # ------
    # Sample is _at_ the end and needs padding.
    snippets = mea.spike_snippets(
        stimulus,
        stimulus_frame_to_spikes(stim_frame_of_spike[4]),
        stimulus_sample_rate,
        mea.ELECTRODE_FREQ,
        total_len,
        pad,
    )
    for s in snippets:
        expected_slice = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        assert s.shape == expected_slice.shape
        numpy.testing.assert_equal(s, expected_slice)


def test_save_recording_names(tmp_path, response_data):
    rec_names = mea.recording_names(response_data)
    path = mea._save_recording_names(rec_names, tmp_path)
    expected_path = tmp_path / mea.REC_NAMES_FILENAME
    assert path == expected_path
    assert expected_path.is_file()
    with open(expected_path, "rb") as f:
        contents = pickle.load(f)
    assert contents == rec_names


def test_save_cluster_ids(tmp_path, response_data):
    rec_name = "Chicken_13_08_21_Phase_00"
    cluster_ids = mea._cluster_ids(response_data, rec_name)
    path = mea._save_cluster_ids(cluster_ids, tmp_path)
    expected_path = tmp_path / mea.CLUSTER_IDS_FILENAME
    assert path == expected_path
    assert expected_path.is_file()
    with open(expected_path, "rb") as f:
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
    sensor_sample_rate = 1
    # Fake stimulus pattern.
    stimulus_pattern = np.array(
        [
            [0, 0, 0, 1],  # 0
            [0, 0, 1, 0],  # 1
            [0, 0, 1, 1],  # 2
            [0, 1, 0, 0],  # 3
            [0, 1, 0, 1],  # 4
            [0, 1, 1, 0],  # 5
            [0, 1, 1, 1],  # 6
            [1, 0, 0, 0],  # 7
            [1, 0, 0, 1],  # 8
            [1, 0, 1, 0],  # 9
            [1, 0, 1, 1],  # 10
            [1, 1, 0, 0],  # 11
        ]
    )
    # fmt: off

    # And fake stimulus events, each exactly on the sensor event.
    stimulus_events = np.arange(0, len(stimulus_pattern))
    num_sensor_samples = len(stimulus_events) 
    # Fake response
    rec_name1 = "Chicken_04_08_21_Phase_01"
    rec_name2 = "Chicken_04_08_21_Phase_02"
    cluster_ids1 = [25, 40]
    cluster_ids2 = [17, 40]
    spike_events1 = [
        np.array([1, 8]),
        np.array([6, 9])]
    spike_events2 = [
        np.array([11], dtype=int),
        np.array([9], dtype=int)]

    # Finally, make a CompressedSpikeRecording.
    recording1 = mea.CompressedSpikeRecording(
            rec_name1,
            stimulus_pattern,
            stimulus_events,
            spike_events1,
            cluster_ids1,
            sensor_sample_rate,
            num_sensor_samples)
    recording2 = mea.CompressedSpikeRecording(
            rec_name2,
            stimulus_pattern,
            stimulus_events,
            spike_events2,
            cluster_ids2,
            sensor_sample_rate,
            num_sensor_samples)

    # The following is the predicted snippets.
    expected_spike_snippets1 = np.array(
        [
            [
                [0, 0, 0, 0],  # pad
                [0, 0, 0, 0],  # pad
                [0, 0, 0, 0],  # pad
                [0, 0, 0, 1],  # 0
                [0, 0, 1, 0],  # 1 <-- spike
                [0, 0, 1, 1],  # 2
                [0, 1, 0, 0],  # 3
            ],
            [
                [0, 1, 0, 1],  # 4
                [0, 1, 1, 0],  # 5
                [0, 1, 1, 1],  # 6
                [1, 0, 0, 0],  # 7
                [1, 0, 0, 1],  # 8 <- spike
                [1, 0, 1, 0],  # 9
                [1, 0, 1, 1],  # 10
            ],
            [
                [0, 0, 1, 1],  # 2
                [0, 1, 0, 0],  # 3
                [0, 1, 0, 1],  # 4
                [0, 1, 1, 0],  # 5
                [0, 1, 1, 1],  # 6 <- spike
                [1, 0, 0, 0],  # 7
                [1, 0, 0, 1],  # 8
            ],
            [
                [0, 1, 1, 0],  # 5
                [0, 1, 1, 1],  # 6
                [1, 0, 0, 0],  # 7
                [1, 0, 0, 1],  # 8
                [1, 0, 1, 0],  # 9 <- spike
                [1, 0, 1, 1],  # 10
                [1, 1, 0, 0],  # 11
            ],
        ]
    )
    expected_spike_snippets2 = np.array(
        [
            [
                [1, 0, 0, 0],  # 7
                [1, 0, 0, 1],  # 8
                [1, 0, 1, 0],  # 9
                [1, 0, 1, 1],  # 10
                [1, 1, 0, 0],  # 11 <- spike
                [0, 0, 0, 0],  # pad
                [0, 0, 0, 0],  # pad
            ],
            [
                [0, 1, 1, 0],  # 5
                [0, 1, 1, 1],  # 6
                [1, 0, 0, 0],  # 7
                [1, 0, 0, 1],  # 8
                [1, 0, 1, 0],  # 9 <- spike
                [1, 0, 1, 1],  # 10
                [1, 1, 0, 0],  # 11
            ],
        ]
    )
    # fmt: on
    expected_cluster_ids1 = np.array([25, 25, 40, 40])
    expected_cluster_ids2 = np.array([17, 40])

    # Test 1 (rec_name1)
    spike_snippets, cluster_ids, _ = mea.labeled_spike_snippets(
        recording1,
        snippet_len,
        snippet_pad,
        downsample=1,
    )
    for idx, (spwin, cluster_ids) in enumerate(
        zip(spike_snippets, cluster_ids)
    ):
        np.testing.assert_equal(spwin, expected_spike_snippets1[idx])
        np.testing.assert_equal(cluster_ids, expected_cluster_ids1[idx])

    # Test 2 (rec_name2)
    spike_snippets, cluster_ids, _ = mea.labeled_spike_snippets(
        recording2, snippet_len, snippet_pad, downsample=1
    )
    for idx, (spwin, cluster_ids) in enumerate(
        zip(spike_snippets, cluster_ids)
    ):
        np.testing.assert_equal(spwin, expected_spike_snippets2[idx])
        np.testing.assert_equal(cluster_ids, expected_cluster_ids2[idx])


def test_write_rec_snippets(tmp_path, comp_exp12):
    # Setup
    snippet_len = 7
    snippet_pad = 1
    empty_snippets = 0
    snippets_per_file = 100
    downsample = 18
    id_for_folder_name = 5

    # Test
    # TODO: add some more checks.
    # Currently, the test only checks that the method runs to completion.
    mea._write_rec_snippets(
        comp_exp12,
        tmp_path,
        id_for_folder_name,
        downsample,
        snippet_len,
        snippet_pad,
        empty_snippets,
        snippets_per_file,
    )
