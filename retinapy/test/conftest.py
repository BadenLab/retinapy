import pytest
import numpy as np
import random as rand

import retinapy.mea as mea


@pytest.fixture
def seed_random():
    rand.seed(123)
    np.random.seed(123)


@pytest.fixture
def np_rng():
    return np.random.default_rng(123)


FF_NOISE_PATTERN_PATH = "./data/ff_noise.h5"
FF_RECORDED_NOISE_PATH = "./data/ff_recorded_noise.pickle"
FF_SPIKE_RESPONSE_PATH = "./data/ff_spike_response.pickle"


@pytest.fixture
def rec0(stimulus_pattern, recorded_stimulus, response_data):
    exp = mea.single_3brain_recording(
        "Chicken_04_08_21_Phase_01",
        stimulus_pattern,
        recorded_stimulus,
        response_data,
    )
    return exp


@pytest.fixture
def rec12(stimulus_pattern, recorded_stimulus, response_data):
    exp = mea.single_3brain_recording(
        "Chicken_17_08_21_Phase_00",
        stimulus_pattern,
        recorded_stimulus,
        response_data,
    )
    return exp


@pytest.fixture
def dc_rec0(rec0):
    downsample = 18
    exp = mea.decompress_recording(rec0, downsample)
    return exp


@pytest.fixture
def dc_rec12(rec12):
    downsample = 18
    exp = mea.decompress_recording(rec12, downsample)
    return exp


@pytest.fixture
def five_recs():
    """Load 5 recordings.

    5 is just a small number for testing.
    """
    return mea.load_3brain_recordings(
        FF_NOISE_PATTERN_PATH,
        FF_RECORDED_NOISE_PATH,
        FF_SPIKE_RESPONSE_PATH,
        include=[
            "Chicken_04_08_21_Phase_01",
            "Chicken_04_08_21_Phase_02",
            "Chicken_05_08_21_Phase_00",
            "Chicken_05_08_21_Phase_01",
            "Chicken_06_08_21_2nd_Phase_00",
        ],
        num_workers=5,
    )

@pytest.fixture
def five_dc_recs(five_recs):
    """Load 5 decompressed recordings."""
    dc_recs = mea.decompress_recordings(five_recs, downsample=18, 
                                        num_workers=5)
    return dc_recs
