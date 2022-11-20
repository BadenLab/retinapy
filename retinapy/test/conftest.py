import pytest
import numpy as np
import random as rand

import retinapy.mea as mea
import pathlib


@pytest.fixture
def seed_random():
    rand.seed(123)
    np.random.seed(123)


@pytest.fixture
def np_rng():
    return np.random.default_rng(123)


DATA_DIR = pathlib.Path("./data/ff_noise_recordings")


@pytest.fixture
def rec0():
    exp = mea.single_3brain_recording("Chicken_04_08_21_Phase_01", DATA_DIR)
    return exp


@pytest.fixture
def rec1():
    exp = mea.single_3brain_recording("Chicken_04_08_21_Phase_02", DATA_DIR)
    return exp


@pytest.fixture
def rec2():
    exp = mea.single_3brain_recording("Chicken_05_08_21_Phase_01", DATA_DIR)
    return exp


@pytest.fixture
def rec12():
    exp = mea.single_3brain_recording("Chicken_17_08_21_Phase_00", DATA_DIR)
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
        DATA_DIR,
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
