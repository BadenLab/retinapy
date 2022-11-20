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


def test_create_multi_cluster_df_datasets(rec0, rec1, rec2):
    # Setup
    # Note down the expected number of clusters. These were calculated once
    # in a Jupyter notebook. It's not a very good ground truth, as it was using
    # the function in question; however, it does work as a check against any
    # unexpected changes.
    #
    # Filtered clusters:
    #  - 24 filtered from rec2 for having too few spikes.
    #  - 66 filtered from rec1 for having too few spikes.
    #  - 17 filtered from rec0 for having too few spikes.
    #  - 3 filtered from rec2 for being > 19 Hz.
    # Total: 110
    expected_num_filtered = 110
    expected_num_clusters = (
        rec0.num_clusters()
        + rec1.num_clusters()
        + rec2.num_clusters()
        - expected_num_filtered
    )

    # Test
    train_ds, val_ds, test_ds = sp.create_multi_cluster_df_datasets(
        [rec0, rec1, rec2],
        input_len=992,
        output_len=100,
        downsample=18,
        stride=17,
        num_workers=5,
    )
    assert (
        expected_num_clusters
        == train_ds.num_clusters
        == val_ds.num_clusters
        == test_ds.num_clusters
    )


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
