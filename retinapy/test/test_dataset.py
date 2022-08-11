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


def test_SpikeDistanceFieldDataset(exp12_1kHz):
    # Setup
    snippet_len = 2000
    mask_begin = 1000
    mask_end = 1500
    max_dist = 500
    mask_shape = (mask_end - mask_begin,)

    # Test
    # 1. The dataset should be created correctly.
    ds = dataset.SpikeDistanceFieldDataset(
        exp12_1kHz, snippet_len, mask_begin, mask_end, max_dist
    )
    # 2. The dataset should have the correct length.
    assert len(ds) == len(exp12_1kHz) - snippet_len + 1

    # 3. The dataset should return the correct snippet and distance fields.
    masked_snippet, target_spikes, before_dist, after_dist = ds[0]
    # 3.1. The shapes should be correct.
    assert masked_snippet.shape == (mea.NUM_STIMULUS_LEDS + 1, snippet_len)
    assert target_spikes.shape == mask_shape
    assert before_dist.shape == mask_shape
    assert after_dist.shape == mask_shape
    # 3.2 The target spikes should be a boolean array, with mostly False.
    assert target_spikes.dtype == np.bool
    known_spike_count = 8
    assert np.sum(target_spikes) == known_spike_count
    # 3.3 The distance fields should be zero where there are spikes, and 
    #     non-zero where there are no spikes.
    assert np.all(~before_dist[target_spikes])
    assert np.all(~after_dist[target_spikes])
    assert np.all(before_dist[~target_spikes])
    assert np.all(after_dist[~target_spikes])
    # 3.4 No distance in the distance fields should be larger than max_dist.
    assert np.max(before_dist) <= max_dist
    assert np.max(after_dist) <= max_dist
