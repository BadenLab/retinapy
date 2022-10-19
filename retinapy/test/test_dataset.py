import pytest
import retinapy.dataset as dataset
import retinapy.mea as mea
import numpy as np


FF_NOISE_PATTERN_PATH = "./data/ff_noise.h5"
FF_SPIKE_RESPONSE_PATH = "./data/ff_spike_response.pickle"
FF_RECORDED_NOISE_PATH = "./data/ff_recorded_noise.pickle"


@pytest.fixture
def exp12_1kHz():
    stimulus_pattern = mea.load_stimulus_pattern(FF_NOISE_PATTERN_PATH)
    response_data = mea.load_response(FF_SPIKE_RESPONSE_PATH)
    recorded_stimulus = mea.load_recorded_stimulus(FF_RECORDED_NOISE_PATH)
    rec = mea.single_3brain_recording(
        "Chicken_17_08_21_Phase_00",
        stimulus_pattern,
        recorded_stimulus,
        response_data,
    )
    rec = mea.decompress_recording(rec, downsample=18)
    return rec


def test_DistFieldDataset(exp12_1kHz):
    # Setup
    snippet_len = 2000
    mask_begin = 1000
    mask_end = 1500
    max_dist = 500
    pad = 500
    mask_shape = (mask_end - mask_begin,)

    # Test
    # 1. The dataset should be created correctly.
    exp12_1kHz_cluster_13 = exp12_1kHz.clusters({13})
    ds = dataset.DistFieldDataset(
        exp12_1kHz_cluster_13, snippet_len, mask_begin, mask_end, pad, max_dist
    )
    # 2. The dataset should have the correct length.
    assert len(ds) == len(exp12_1kHz) - snippet_len - pad + 1

    # 3. The dataset should return the correct snippet and distance fields.
    sample = ds[0]
    masked_snippet = sample["snippet"]
    target_spikes = sample["target_spikes"]
    dist_field = sample["dist"]
    # 3.1. The shapes should be correct.
    # The first dimension: LEDs (4), spikes (1), and position encoding (1).
    assert masked_snippet.shape == (mea.NUM_STIMULUS_LEDS + 2, snippet_len)
    assert target_spikes.shape == mask_shape
    assert dist_field.shape == mask_shape
    # 3.2 The target spikes should be float array, with mostly zeros.
    # Note: this was changed from int. float resulted in higher examples/sec
    # when training.
    assert target_spikes.dtype == float
    known_spike_count = 10
    assert np.sum(target_spikes) == known_spike_count
    # 3.3 The distance fields should be zero where there are spikes, and
    #     non-zero where there are no spikes.
    assert not np.any(dist_field[np.where(target_spikes > 0)])
    assert np.all(dist_field[np.where(target_spikes == 0)])
    # 3.4 No distance in the distance fields should be larger than max_dist.
    assert np.max(dist_field) <= max_dist


@pytest.fixture
def two_exps():
    recs = mea.load_3brain_recordings(
        FF_NOISE_PATTERN_PATH,
        FF_RECORDED_NOISE_PATH,
        FF_SPIKE_RESPONSE_PATH,
        include=["Chicken_04_08_21_Phase_01", "Chicken_20_08_21_Phase_00"],
    )
    dc_recs = mea.decompress_recordings(recs, downsample=18)
    return dc_recs


@pytest.mark.parametrize("stride", [1, 3, 17])
def test_ConcatDistFieldDataset(two_exps, np_rng, stride):
    """Test the concatenated distfield dataset.

    Tests that:
        1. Construction from two DistField datasets throws no errors and has 
            expected length.
        2. Samples with same timestep but different cluster have equal stimulus.
        3. All samples for a single cluster have the correct (and same) 
            "cluster_id" and "rec_id" value.

    These tests are run parameterized for different strides.
    """
    # Setup
    num_timestep_trials = 1000
    num_cluster_trials = 50
    snippet_len = 1000
    mask_begin = 800
    mask_end = 1000
    pad = 100
    dist_clamp = 500
    distfield_ds1 = dataset.DistFieldDataset(
        two_exps[0],
        snippet_len,
        mask_begin,
        mask_end,
        pad,
        dist_clamp,
        stride=stride,
        enable_augmentation=False,
    )
    distfield_ds2 = dataset.DistFieldDataset(
        two_exps[1],
        snippet_len,
        mask_begin,
        mask_end,
        pad,
        dist_clamp,
        stride=stride,
        enable_augmentation=False,
    )

    # Test
    # 1. The dataset should be created correctly.
    concat_ds = dataset.ConcatDistFieldDataset([distfield_ds1, distfield_ds2])
    assert len(concat_ds) == len(distfield_ds1) + len(distfield_ds2)

    # 2. Two clusters from the same recording should share the same stimulus.
    # 2.1 For the first recording.
    num_timesteps1 = distfield_ds1.ds.num_strided_timesteps
    test_idxs = np_rng.integers(0, num_timesteps1, num_timestep_trials)
    for idx in test_idxs:
        s1 = concat_ds[idx]["snippet"][0 : mea.NUM_STIMULUS_LEDS, :]
        s2 = concat_ds[idx + num_timesteps1]["snippet"][
            0 : mea.NUM_STIMULUS_LEDS, :
        ]
        np.testing.assert_allclose(s1, s2, err_msg=f"idx={idx}")
    # 2.2 For the second recording.
    num_timesteps2 = distfield_ds2.ds.num_strided_timesteps
    ds_2_start_idx = len(distfield_ds1)
    test_idxs = np_rng.integers(
        ds_2_start_idx, ds_2_start_idx + num_timesteps2, num_timestep_trials
    )
    for idx in test_idxs:
        s1 = concat_ds[idx]["snippet"][0 : mea.NUM_STIMULUS_LEDS, :]
        s2 = concat_ds[idx + num_timesteps2]["snippet"][
            0 : mea.NUM_STIMULUS_LEDS, :
        ]
        np.testing.assert_allclose(s1, s2, err_msg=f"idx={idx}")

    # 3. A cluster's snippets should have the same 'rec_id' and 'cluster_id'.
    # 3.1 For the first recording.
    cluster_idxs = np_rng.integers(
        0, len(distfield_ds1.recording.cluster_ids), num_cluster_trials
    )
    for c_idx in cluster_idxs:
        offset = c_idx * distfield_ds1.ds.num_strided_timesteps
        test_idxs = np_rng.integers(0, num_timesteps1, num_timestep_trials)
        for idx in test_idxs:
            c_idx_from_ds = concat_ds[idx + offset]["cluster_id"]
            rec_idx_from_ds = concat_ds[idx + offset]["rec_id"]
            assert c_idx_from_ds == c_idx
            assert rec_idx_from_ds == 0
    # 3.2 For the second recording.
    cluster_idxs = np_rng.integers(
        0, len(distfield_ds2.recording.cluster_ids), num_cluster_trials
    )
    for c_idx in cluster_idxs:
        offset = (
            len(distfield_ds1) + c_idx * distfield_ds2.ds.num_strided_timesteps
        )
        test_idxs = np_rng.integers(0, num_timesteps2, num_timestep_trials)
        for idx in test_idxs:
            c_idx_from_ds = concat_ds[idx + offset]["cluster_id"]
            rec_idx_from_ds = concat_ds[idx + offset]["rec_id"]
            assert c_idx_from_ds == c_idx
            assert rec_idx_from_ds == 1
