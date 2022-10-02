import logging
import pytest
import retinapy.spikeprediction as sp
import retinapy.mea as mea
import itertools


FF_NOISE_PATTERN_PATH = "./data/ff_noise.h5"
FF_SPIKE_RESPONSE_PATH = "./data/ff_spike_response.pickle"
FF_RECORDED_NOISE_PATH = "./data/ff_recorded_noise.pickle"


# TODO: move this to common test utils.
@pytest.fixture
def rec0():
    rec = mea.single_3brain_recording(
        "Chicken_04_08_21_Phase_01",
        mea.load_stimulus_pattern(FF_NOISE_PATTERN_PATH),
        mea.load_recorded_stimulus(FF_RECORDED_NOISE_PATH),
        mea.load_response(FF_SPIKE_RESPONSE_PATH)
    )
    return rec


def test_trainable_factories(rec0):
    """
    Tests multiple functions in one go (so as to speed up tests).

    Tests that: 
        1. both create_distfield_trainable and create_lnl_trainable
           run without errors for a number of configurations.
    """
    # Setup
    downsample_factors = [89, 178]
    input_lengths_ms = [992, 1586]
    output_lenghts_ms = [1, 50]
    configs = tuple(
        sp.Configuration(*tple)
        for tple in itertools.product(
            downsample_factors,
            input_lengths_ms,
            output_lenghts_ms,
        )
    )
    cluster40_rec0 = rec0.clusters({40})
    # Test
    for config in configs:
        t1 = sp.DistFieldCnnTGroup.create_trainable(cluster40_rec0, 
                                                            config)
        t2 = sp.LinearNonLinearTGroup.create_trainable(cluster40_rec0, 
                                                               config)
        assert t1 is not None
        assert t2 is not None

