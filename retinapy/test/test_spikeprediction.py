import logging
import pytest
import retinapy.spikeprediction as sp
import retinapy.mea as mea
import itertools
import pathlib

DATA_DIR = pathlib.Path("./data/ff_noise_recordings")


# TODO: move this to common test utils.
@pytest.fixture
def recs():
    res = mea.load_3brain_recordings(
        DATA_DIR,
        include=["Chicken_04_08_21_Phase_01", "Chicken_04_08_21_Phase_02"],
    )
    # Filter out some clusters
    res[0] = res[0].clusters({40, 41})
    res[1] = res[1].clusters({46, 352, 703})
    return res


def test_trainable_factories(recs):
    """
    Tests multiple functions in one go (so as to speed up tests).

    Tests that:
        1. for each tranable group, a trainable is created without error for
        a small set of different configurations.
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
    # Get default options from arg parser.
    _, parser = sp.arg_parsers()
    default_opts = parser.parse_args([])
    # For the models that only support a single cluster:
    single_cluster = [recs[0].clusters({40})]

    # Test
    for config in configs:
        assert (
            sp.LinearNonLinearTGroup.create_trainable(
                single_cluster, config, default_opts
            )
        ) is not None
        assert (
            sp.DistFieldCnnTGroup.create_trainable(
                single_cluster, config, default_opts
            )
        ) is not None
        assert (
            sp.MultiClusterDistFieldTGroup.create_trainable(
                recs, config, default_opts
            )
        ) is not None
        assert (
            sp.TransformerTGroup.create_trainable(recs, config, default_opts)
        ) is not None
        assert (
            sp.ClusteringTGroup.create_trainable(recs, config, default_opts)
        ) is not None
