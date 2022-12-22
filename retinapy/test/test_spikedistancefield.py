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


@pytest.fixture
def model_out():
    # fmt: off
    m_out = torch.tensor(
        [ 0.4894,  0.4011,  0.2615,  0.2934,  0.2920,  0.4351,  0.3792,  0.2756,
          0.2254,  0.2130,  0.3810,  0.2311,  0.3143,  0.1473,  0.1911,  0.0621,
          0.1182,  0.2855,  0.1616,  0.1891,  0.1915,  0.0125, -0.0554, -0.0989,
         -0.0705,  0.0066, -0.1032,  0.0580, -0.1414, -0.1264, -0.3183, -0.1703,
         -0.1188, -0.2844, -0.2782, -0.3319, -0.3468, -0.1175, -0.2886, -0.3774,
         -0.2710, -0.2036, -0.2079, -0.4266, -0.4779, -0.3084, -0.4457, -0.7059,
         -0.3227, -0.5177, -0.4833, -0.6246, -0.2733, -0.3722, -0.3221, -0.5164,
         -0.5055, -0.5312, -0.5755, -0.3454, -0.6374, -0.4992, -0.4382, -0.5223,
         -0.8060, -0.5486, -0.3585, -0.5843, -0.4683, -0.5209, -0.5460, -0.6472,
         -0.7897, -0.6615, -0.5655, -0.7873, -0.6966, -0.7216, -0.8772, -0.6974,
         -0.6149, -0.5308, -0.8156, -0.8241, -0.8226, -0.6119, -0.7701, -0.8313,
         -0.7065, -0.8450, -0.7924, -0.7698, -0.7925, -0.7792, -0.9590, -0.8655,
         -0.7848, -0.9146, -0.8254, -0.7527, -0.7260, -0.8419, -0.6392, -0.9389,
         -1.0958, -0.7896, -0.6737, -0.7596, -0.6505, -0.6735, -0.6042, -0.7249,
         -0.6186, -0.7381, -0.8281, -0.4862, -0.7551, -0.9013, -0.7199, -0.6987,
         -0.8099, -0.7832, -0.6781, -0.6754, -0.6618, -0.6769, -0.6471, -0.6859,
         -0.8223, -0.7526, -0.7328, -0.6135, -0.5127, -0.6113, -0.4793, -0.4298,
         -0.6625, -0.7032, -0.6157, -0.6887, -0.5769, -0.4660, -0.4941, -0.5800,
         -0.4354, -0.5422, -0.3904, -0.4516, -0.5311, -0.4200, -0.5390, -0.3285,
         -0.4625, -0.3773, -0.5106, -0.6114, -0.4547, -0.4823, -0.4969, -0.2066,
         -0.4920, -0.3165, -0.4899, -0.3512, -0.1178, -0.4567, -0.4579, -0.3514,
         -0.2120, -0.3493, -0.3918, -0.1780, -0.3574, -0.3294, -0.2956, -0.1519,
         -0.2250, -0.2273, -0.0325, -0.2153, -0.1886, -0.0219, -0.2240, -0.0705,
         -0.2690, -0.1702, -0.1968,  0.0284, -0.1367, -0.2778, -0.2963, -0.3697,
         -0.2111, -0.1242, -0.2309, -0.2367, -0.1295, -0.1725, -0.1652, -0.1543,
         -0.1757, -0.1950, -0.1479, -0.1138, -0.1335,  0.0070, -0.0735, -0.0121,
         -0.0505,  0.2243,  0.0186, -0.1186, -0.0888,  0.1576, -0.0675, -0.2259,
          0.1117, -0.1274,  0.0042,  0.0855,  0.0160,  0.1925,  0.0262,  0.1184,
         -0.0223,  0.0583,  0.1262,  0.1019,  0.2555,  0.0884,  0.0761,  0.1445,
          0.2110,  0.2663,  0.1752,  0.2272,  0.3009,  0.2622,  0.1527,  0.1774,
         -0.0359,  0.2301,  0.0512,  0.1823,  0.2320,  0.1454,  0.1736,  0.2462,
          0.1392,  0.2303,  0.2476,  0.2392,  0.2508,  0.1602,  0.3934,  0.2380,
          0.2492,  0.3875,  0.3098,  0.3362,  0.3215,  0.2594,  0.3919,  0.1859,
          0.3256,  0.3779,  0.4607,  0.4322,  0.5226,  0.2696,  0.4180,  0.5693,
          0.7369,  0.6615,  0.5092,  0.4336,  0.4846,  0.6086,  0.6781,  0.5601,
          0.5698,  0.6761,  0.6295,  0.3697,  0.7786,  0.5129,  0.6587,  0.8219,
          0.6926,  0.4487,  0.5463,  0.6973,  0.6021,  0.5258,  0.7146,  0.8207,
          0.5809,  0.4910,  0.7438,  0.6263,  0.7713,  0.8162,  0.7625,  0.8670,
          0.8140,  0.8140,  0.7516,  0.8940,  0.8027,  0.8454,  0.9290,  0.7825,
          0.8352,  0.8767,  1.0313,  0.7601,  0.8175,  0.9062,  1.0742,  0.8811,
          0.9060,  0.7844,  1.0226,  0.9159,  1.0629,  1.0483,  0.8602,  1.0611,
          1.0423,  1.1778,  1.1160,  1.0138,  1.2130,  1.2146,  1.2061,  1.0018,
          1.1067,  1.1410,  1.0872,  0.9375,  1.1646,  1.4330,  1.1668,  1.2610,
          1.1011,  1.2249,  1.1975,  1.3248,  1.0587,  1.2501,  1.1901,  1.3800,
          1.3391,  1.4126,  1.4046,  1.2834,  1.3593,  1.4846,  1.2942,  1.1529,
          1.3288,  1.2906,  1.3560,  1.3603,  1.5383,  1.5052,  1.6077,  1.2861,
          1.2679,  1.3342,  1.6458,  1.2148,  1.3937,  1.4690,  1.3483,  1.3835,
          1.4845,  1.3828,  1.5018,  1.2510,  1.5238,  1.2786,  1.5751,  1.3669,
          1.5252,  1.4066,  1.3523,  1.4890,  1.4622,  1.3569,  1.4364,  1.4126,
          1.5315,  1.6878,  1.5042,  1.3435,  1.5752,  1.6419,  1.4947,  1.5581])
    # fmt: on
    return m_out


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
        torch.full((6, 1), -max_dist),
        torch.full((6, 1), before_field.shape[1] + max_dist - 1),
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


def test_mle_inference_from_df(distance_field_data):
    M, spikes, dist_fields = distance_field_data
    for spike, dist_field in zip(spikes, dist_fields):
        num_spikes = sdf.mle_inference_via_dp(
            torch.Tensor(dist_field),
            lhs_spike=-M,
            rhs_spike=len(spike) + M - 1,
            spike_pad=1,
            max_clamp=M * 2,
            max_num_spikes=5,
            resolution=1,
        )


def test_predict(model_out):
    pred = sdf.predict_sum(
        model_out, lhs_spike=-200, max_dist=1200, output_offset=3.4
    )
    print(pred)
    assert pred > 1
