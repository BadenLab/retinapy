from collections import namedtuple
import math
import pickle

import numpy as np
import numpy.ma as ma
import numpy.testing
import pandas as pd

import pytest
import retinapy.mea_noise as mea


FF_NOISE_PATTERN_PATH = "./data/ff_noise.h5"
FF_SPIKE_RESPONSE_PATH = "./data/ff_spike_response.pickle"
FF_SPIKE_RESPONSE_PATH_ZIP = "./data/ff_spike_response.pickle.zip"
FF_RECORDED_NOISE_PATH = "./data/ff_recorded_noise.pickle"
FF_RECORDED_NOISE_PATH_ZIP = "./data/ff_recorded_noise.pickle.zip"


class ExperimentData:
    def __init__(
        self,
        rec_name,
        stimulus_pattern,
        response,
        recorded_stimulus,
        downsample_factor=18,
    ):
        self.rec_name = rec_name
        self.stimulus_pattern = stimulus_pattern
        self.response = response
        self.recorded_stimulus = recorded_stimulus
        self.downsample_factor = downsample_factor
        self._rec_id = None
        self._stimulus = None
        self._stimulus_sample_rate = None
        self._sensor_sample_rate = None

    @property
    def stimulus(self) -> np.ndarray:
        if not self._stimulus:
            stimlus, freq = mea.decompress_stimulus(
                self.stimulus_pattern,
                self.recorded_stimulus,
                self.rec_name,
                self.downsample_factor,
            )
            self._stimulus = stimlus
            self._stimulus_sample_rate = freq
        return self._stimulus

    @property
    def stimulus_sample_rate(self) -> float:
        if not self._stimulus_sample_rate:
            assert not self._stimulus
            self.stimulus
            assert self._stimulus_sample_rate
        return self._stimulus_sample_rate

    @property
    def rec_id(self) -> int:
        if not self._rec_id:
            self._rec_id = mea.recording_names(self.response).index(
                self.rec_name
            )
            assert self._rec_id
        return self._rec_id

    @property
    def sensor_sample_rate(self) -> float:
        if not self._sensor_sample_rate:
            self._sensor_sample_rate = self.recorded_stimulus.xs(
                self.rec_name, "Recording"
            ).iloc[0]["Sampling_Freq"]
            assert self._sensor_sample_rate
        self._sensor_sample_rate
        return self._sensor_sample_rate

    @property
    def num_sensor_samples(self) -> int:
        row = self.recorded_stimulus.xs(self.rec_name, level="Recording").iloc[
            0
        ]
        num_samples = row["End_Fr"] - row["Begin_Fr"] + 1
        return num_samples

    def spikes(self, cluster_id) -> np.ndarray:
        spikes = (
            self.response.xs(self.rec_name, level="Recording")
            .reset_index("Stimulus ID")
            .loc[cluster_id]["Spikes"]
            .compressed()
        )
        return spikes


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
def exp0(stimulus_pattern, recorded_stimulus, response_data):
    exp = ExperimentData(
        "Chicken_04_08_21_Phase_01",
        stimulus_pattern,
        response_data,
        recorded_stimulus,
    )
    return exp


@pytest.fixture
def exp12(stimulus_pattern, recorded_stimulus, response_data):
    exp = ExperimentData(
        "Chicken_17_08_21_Phase_00",
        stimulus_pattern,
        response_data,
        recorded_stimulus,
    )
    return exp


@pytest.fixture
def decimated_stimulus_freq():
    stim, freq = mea.decompress_stimulus(
        mea.load_stimulus_pattern(FF_NOISE_PATTERN_PATH),
        mea.load_recorded_stimulus(FF_RECORDED_NOISE_PATH),
        "Chicken_17_08_21_Phase_00",
        18,
    )
    return stim, freq


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
                  82, 99, 114, 119, 149, 217, 224, 287, 317, 421, 553, 591,]
    # fmt: on
    recording_name = "Chicken_21_08_21_Phase_00"
    cluster_ids = mea.cluster_ids(response_data, recording_name)
    assert cluster_ids == known_list


def test_decompress_stimulus(stimulus_pattern, recorded_stimulus):
    # Test 1
    # No downsampling.
    decomp, new_freq = mea.decompress_stimulus(
        stimulus_pattern,
        recorded_stimulus,
        "Chicken_17_08_21_Phase_00",
        downsample_factor=1,
    )
    known_length = 16071532
    assert decomp.shape == (known_length, mea.NUM_STIMULUS_LEDS)
    assert new_freq == pytest.approx(mea.ELECTRODE_FREQ)

    # Test 2
    # Downsample at x18.
    downsample_factor = 18
    orig_len = 16071532
    expected_decimated_len = math.ceil(orig_len / downsample_factor)
    orig_freq = mea.ELECTRODE_FREQ
    expected_downsampled_freq = orig_freq / downsample_factor
    decomp, new_freq = mea.decompress_stimulus(
        stimulus_pattern,
        recorded_stimulus,
        "Chicken_17_08_21_Phase_00",
        downsample_factor,
    )
    assert decomp.shape == (expected_decimated_len, mea.NUM_STIMULUS_LEDS)
    # Note: the sampling frequency in the Pandas dataframe isn't as accurate
    # as the ELECTRODE_FREQ value.
    assert new_freq == pytest.approx(expected_downsampled_freq)


def test_downsample_stimulus():
    # Setup
    # fmt: off
    orig_signal = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1,])
    # TODO: is this satisfactory?
    expected_decimate_by_2 = np.array(
            [0.012, -0.019,  0.029, -0.060,
             0.739,  1.085,  0.937,  1.081,
             0.249, -0.076,  0.058, -0.077,
             0.750,  1.077,  0.942,  1.077,
             0.250, -0.077,  0.058, -0.076,
             0.749,  1.081,  0.937,  1.085,])
    expected_decimate_by_4 = np.array(
            [0.030, -0.065,  0.595,  1.148,
             0.364, -0.120,  0.620,  1.136,
             0.369, -0.120,  0.614,  1.148,])
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


def test_decompress_stimulus():
    # Setup
    # fmt: off
    stimulus_pattern = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0],]).T
    trigger_events = np.array([0, 4, 5, 7, 12])
    # Out array should be overridden, so check this by filling with some value.
    out_arr = np.full(shape=(20, stimulus_pattern.shape[1]), fill_value=7)
    expected_output = np.array([
        # idx: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
        # val: 0  0  0  0  1  2  2  3  3  3  3  3  4  4  4  4  4  4  4  4
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],]).T
    # fmt: on

    # Test 1
    mea._decompress_stimulus(stimulus_pattern, trigger_events, out_arr)
    np.testing.assert_array_equal(out_arr, expected_output)

    # Test 2
    # Non-zero starting trigger is invalid.
    trigger_events_invalid = np.array([2, 4, 5, 7, 12])
    with pytest.raises(ValueError):
        mea._decompress_stimulus(
            stimulus_pattern, trigger_events_invalid, out_arr
        )


def test_decompress_spikes1():
    # Setup
    downsample_by = 9
    num_sensor_samples = 123
    # fmt: off
    spike_times1 = np.array([0, 1, 8, 9, 10, 27, 30, 40, 50, 70, 80, 90, 100, 110])
    spike_times2 = np.array([8, 9, 30, 40, 50, 70, 80, 90, 100, 110])

    # input index:          [0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117]
    # output index          [0, 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,  12,  13]
    spike_counts1 = np.array([3, 2,  0,  2,  1,  1,  0,  1,  1,  0,  1,  1,  1,  0])
    spike_counts2 = np.array([1, 1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  1,  1,  0])
    # fmt: on

    expected_output_len = math.ceil(num_sensor_samples / downsample_by)
    expected_ones_mask = np.zeros(expected_output_len, dtype=bool)
    expected_ones_mask[spike_times1 // downsample_by] = True
    expected_zeros_mask = ~expected_ones_mask

    # Test 1
    # There should be an error thrown, as two samples land in the same bucket.
    with pytest.raises(ValueError):
        mea.decompress_spikes(spike_times1, num_sensor_samples, downsample_by)

    # Test 2
    spikes = mea.decompress_spikes(
        spike_times2, num_sensor_samples, downsample_by
    )
    assert spikes.shape == (expected_output_len,)
    assert np.all(spikes[expected_ones_mask] == True)
    assert np.all(spikes[expected_zeros_mask] == False)


def test_decompress_spikes2(exp12):
    """
    Test decompress_spikes on actual recordings.

    Not much is actually checked though.
    """
    # Setup
    cluster_id = 36
    spikes = exp12.spikes(cluster_id)
    assert spikes.shape == (2361,)

    # Test
    spikes_decomp1 = mea.decompress_spikes(spikes, exp12.num_sensor_samples)
    spikes_decomp18 = mea.decompress_spikes(
        spikes, exp12.num_sensor_samples, downsample_factor=18
    )


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
        [
            [1, 1, 1, 1],  #     |  2
            [1, 1, 1, 1],  #     1  |
            [0, 1, 1, 1],  #  -  |  -
            [0, 0, 1, 1],  #  |  -
            [0, 0, 0, 1],  #  0       -
            [0, 0, 0, 0],  #  |       |  -
            [1, 0, 0, 0],  #  -       3  |
            [1, 1, 0, 0],
        ]  #          |  4
    )
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
    rec_id = 3
    cluster_ids = mea.cluster_ids(
        response_data, mea.recording_names(response_data)[rec_id]
    )
    path = mea._save_cluster_ids(cluster_ids, rec_id, tmp_path)
    expected_path = tmp_path / str(rec_id) / mea.CLUSTER_IDS_FILENAME
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
    stim_sample_freq = 40
    # Fake stimulus. Note that we are did our own nieve upsampling here.
    # This is done to make the comparison easier.
    stimulus = np.array(
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
    # Fake response
    rec_name1 = "Chicken_04_08_21_Phase_01"
    rec_name2 = "Chicken_04_08_21_Phase_02"
    index = pd.MultiIndex.from_tuples(
        (
            (25, 1, "Chicken_04_08_21_Phase_01"),
            (40, 1, "Chicken_04_08_21_Phase_01"),
            (17, 1, "Chicken_04_08_21_Phase_02"),
            (40, 1, "Chicken_04_08_21_Phase_02"),
        ),
        names=["Cell index", "Stimulus ID", "Recording"],
    )
    # Spike data has format: [(Kernel, Spikes),...]
    # We only care about the spikes, so we can leave the kernels as None.
    # fmt: off
    spike_data = [
        (None, [820, 3648]),
        (None, [3044, 4067]),
        (None, [5239,],),
        (None, [4430,],)]
    # fmt: on
    # Convert the data to use masked arrays.
    spike_data_m = [
        (kernel, ma.array(spikes, mask=np.zeros(len(spikes))))
        for kernel, spikes in spike_data
    ]

    response = pd.DataFrame(
        spike_data_m, index=index, columns=["Kernel", "Spikes"]
    )
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
    expected_cluster_ids1 = np.array([25, 25, 40, 40])
    expected_cluster_ids2 = np.array([17, 40])

    # The spike times were calculated according to the following function.
    # Then, the slices of the stimulus were manually copied, and zoomed by 2,
    # as the stimulus was sampled at 40 Hz (twice the stimulus frequency).
    # Leaving the function here in case the test arrays need to be regenerated.
    def create_ans():
        def spike_to_frame(sp):
            return sp * (stim_sample_freq / mea.ELECTRODE_FREQ)

        res = []
        for d in spike_data:
            d_spikes = d[1]
            for sp in d_spikes:
                res.append(spike_to_frame(sp))
        print(res)

    # Test 1 (rec_name1)
    spike_snippets, cluster_ids = mea.labeled_spike_snippets(
        stimulus,
        response,
        rec_name1,
        stim_sample_freq,
        mea.ELECTRODE_FREQ,
        snippet_len,
        snippet_pad,
    )
    for idx, (spwin, cluster_ids) in enumerate(
        zip(spike_snippets, cluster_ids)
    ):
        np.testing.assert_equal(spwin, expected_spike_snippets1[idx])
        np.testing.assert_equal(cluster_ids, expected_cluster_ids1[idx])

    # Test 2 (rec_name2)
    spike_snippets, cluster_ids = mea.labeled_spike_snippets(
        stimulus,
        response,
        rec_name2,
        stim_sample_freq,
        mea.ELECTRODE_FREQ,
        snippet_len,
        snippet_pad,
    )
    for idx, (spwin, cluster_ids) in enumerate(
        zip(spike_snippets, cluster_ids)
    ):
        np.testing.assert_equal(spwin, expected_spike_snippets2[idx])
        np.testing.assert_equal(cluster_ids, expected_cluster_ids2[idx])


def test_write_rec_snippets(tmp_path, exp12):
    # Setup
    snippet_len = 7
    snippet_pad = 1
    empty_snippets = 0
    snippets_per_file = 100

    # Test
    # TODO: add some more checks.
    # Currently, the test only checks that the method runs to completion.
    mea._write_rec_snippets(
        exp12.stimulus,
        exp12.response,
        exp12.rec_name,
        exp12.rec_id,
        tmp_path,
        snippet_len,
        snippet_pad,
        exp12.stimulus_sample_rate,
        empty_snippets,
        snippets_per_file,
    )
