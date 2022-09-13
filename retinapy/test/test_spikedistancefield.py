from math import dist
import pytest
import retinapy.spikedistancefield as sdf
import numpy as np
import torch


@pytest.fixture
def spike_data1():
    spike_batch = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    MAX_DIST = 6
    M = MAX_DIST  # Used to make the below array literal tidier.

    after_field = np.array(
        [
            [0, 1, 2, 3, 4, 5, M, M, M, 0],
            [M, M, 0, 1, 2, 3, 4, 5, M, M],
            [M, M, M, M, M, M, M, 0, 1, 2],
            [M, 0, 1, 0, 1, 2, 3, 0, 1, 2],
            [M, 0, 0, 0, 1, 2, 3, 4, 5, M],
            [M, M, M, M, M, M, M, M, M, M],
        ]
    )

    before_field = np.array(
        [
            [0, M, M, M, 5, 4, 3, 2, 1, 0],
            [2, 1, 0, M, M, M, M, M, M, M],
            [M, M, 5, 4, 3, 2, 1, 0, M, M],
            [1, 0, 1, 0, 3, 2, 1, 0, M, M],
            [1, 0, 0, 0, M, M, M, M, M, M],
            [M, M, M, M, M, M, M, M, M, M],
        ]
    )

    return (spike_batch, MAX_DIST, before_field, after_field)


@pytest.mark.skip(reason="Broken. See: GitHub issue #2.")
def test_bi_distance_field(spike_data1):
    spike_batch, max_dist, before_field, after_field = spike_data1
    for i in range(spike_batch.shape[0]):
        dist_before, dist_after = sdf.bi_distance_field(
            spike_batch[i], max_dist, max_dist
        )
        assert np.array_equal(dist_before, before_field[i])
        assert np.array_equal(dist_after, after_field[i])


@pytest.fixture
def spike_data2():
    spike_batch = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )  # just for reference

    MAX_DIST = 100
    M = MAX_DIST  # Used to make the below array literal tidier.

    after_field = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
            [M, M, 0, 1, 2, 3, 4, 5, 6, 7],
            [M, M, M, M, M, M, M, 0, 1, 2],
            [M, 0, 1, 0, 1, 2, 3, 0, 1, 2],
            [M, 0, 0, 0, 1, 2, 3, 4, 5, 6],
            [M, M, M, M, M, M, M, M, M, M],
        ]
    )

    before_field = np.array(
        [
            [0, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [2, 1, 0, M, M, M, M, M, M, M],
            [7, 6, 5, 4, 3, 2, 1, 0, M, M],
            [1, 0, 1, 0, 3, 2, 1, 0, M, M],
            [1, 0, 0, 0, M, M, M, M, M, M],
            [M, M, M, M, M, M, M, M, M, M],
        ]
    )

    spike_counts = np.array([2, 1, 1, 3, 3, 0])

    return (spike_counts, MAX_DIST, before_field, after_field)


@pytest.mark.skip(reason="Broken. See: GitHub issue #1.")
def test_count_inference_from_bi_df(spike_data2):
    spike_counts, max_dist, before_field, after_field = spike_data2
    before_field = torch.from_numpy(before_field).to(dtype=torch.float32)
    after_field = torch.from_numpy(after_field).to(dtype=torch.float32)
    for i in range(len(spike_counts)):
        num_spikes = sdf.count_inference_from_bi_df(
            before_field[i],
            after_field[i],
            lhs_spike=-max_dist,
            rhs_spike=len(before_field[i]) + max_dist - 1,
            spike_pad=1,
            target_interval=(0, len(before_field[i])),
            max_num_spikes=max_dist,
        )
        assert num_spikes == spike_counts[i]

@pytest.mark.skip(reason="Broken. See: GitHub issue #1.")
def test_count_inference_from_bi_df2(spike_data2):
    spike_counts, max_dist, before_field, after_field = spike_data2
    before_field = torch.from_numpy(before_field).to(dtype=torch.float32)
    after_field = torch.from_numpy(after_field).to(dtype=torch.float32)
    num_spikes = sdf.count_inference_from_bi_df2(
        before_field,
        after_field,
        torch.full((6,1), -max_dist),
        torch.full((6,1), before_field.shape[1] + max_dist - 1),
        spike_pad=1,
        target_interval=(0, before_field.shape[1]),
    )
    assert np.array_equal(num_spikes.numpy(), spike_counts)


@pytest.fixture
def distance_field_data():
    MAX_DIST = 100
    M = MAX_DIST  # Used to make the below array literal tidier.
    spike_batch = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    distance_field = np.array(
        [
            [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
            [2, 1, 0, 1, 2, 3, 4, 5, 6, 7],
            [7, 6, 5, 4, 3, 2, 1, 0, 1, 2],
            [1, 0, 1, 0, 1, 2, 1, 0, 1, 2],
            [1, 0, 0, 0, 1, 2, 3, 4, 5, 6],
            [M, M, M, M, M, M, M, M, M, M],
        ]
    )
    return MAX_DIST, spike_batch, distance_field


def test_distance_field(distance_field_data):
    M, spikes, dist_fields = distance_field_data
    for spike, known_df in zip(spikes, dist_fields):
        dist_field = sdf.distance_field(spike, M)
        dist_field_cpu = sdf.distance_field2(spike, M)
        assert np.array_equal(known_df, dist_field)
        assert np.array_equal(dist_field_cpu, dist_field)


@pytest.fixture
def spike_interval_data():
    MAX_COUNT = 100
    M = MAX_COUNT  # Used to make the below array literal tidier.
    spike_batch = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    spike_intervals = np.array(
        [
            [0, 8, 8, 8, 8, 8, 8, 8, 8, 0],
            [M, M, 0, M, M, M, M, M, M, M],
            [M, M, M, M, M, M, M, 0, M, M],
            [M, 0, 1, 0, 3, 3, 3, 0, M, M],
            [M, 0, 0, 0, M, M, M, M, M, M],
            [M, M, M, M, M, M, M, M, M, M],
        ]
    )
    return MAX_COUNT, spike_batch, spike_intervals

def test_spike_interval(spike_interval_data):
    M, spikes, spike_intervals = spike_interval_data
    for spike, known_si in zip(spikes, spike_intervals):
        si = sdf.spike_interval(spike, M)
        assert np.array_equal(known_si, si)


