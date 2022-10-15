import retinapy.mea as mea
import retinapy.spikedistancefield as sdf
import numpy as np
import torch
import bisect
import math
from typing import Tuple, Optional, List, Iterable


def _num_snippets(num_timesteps: int, snippet_len: int, stride: int) -> int:
    """
    Returns the number of snippets that can be extracted from a recording with
    the given number of timesteps, snippet length, and stride.
    """
    # The most likely place for insidious bugs to hide is here. So try two
    # ways.
    # 1. Find the last strided timestep, then divide by stride.
    last_nonstrided = num_timesteps - snippet_len
    last_strided_timestep = last_nonstrided - (last_nonstrided % stride)
    num_snippets1 = last_strided_timestep // stride + 1
    # 2. Same deal, but mindlessly copying the formula from Pytorch docs.
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    # Padding is 0, and dilation is 1.
    num_snippets2 = math.floor(
        (num_timesteps - (snippet_len - 1) - 1) / stride + 1
    )
    assert num_snippets1 == num_snippets2, (
        f"Strided snippet calculation is wrong. {num_snippets1} != "
        f"{num_snippets2}."
    )
    return num_snippets1


class SnippetDataset(torch.utils.data.Dataset):
    def __init__(
        self, recording: mea.SpikeRecording, snippet_len: int, stride: int = 1
    ):
        if snippet_len > len(recording):
            raise ValueError(
                f"Snippet length ({snippet_len}) is larger than "
                f"the recording length ({len(recording)})."
            )
        self.recording = recording
        self.snippet_len = snippet_len
        self.stride = stride
        self.num_clusters = len(recording.cluster_ids)
        self.num_timesteps = len(recording) - snippet_len + 1
        assert (
            self.num_timesteps > 0
        ), "Snippet length is longer than the recording."
        self.num_strided_timesteps = _num_snippets(
            len(recording), snippet_len, stride
        )

    def __len__(self):
        """
        Calculates the number of samples in the dataset.
        """
        return self.num_strided_timesteps * self.num_clusters

    def _decode_index(self, index: int) -> Tuple[int, int]:
        """
        Decodes the index into the timestep and cluster id.

        The data is effectively a 2D array with dimensions (time, cluster).
        The index is the flattened index of this array, and so the timestep
        increases as the index increases and wraps to the next cluster id when
        it reaches the end of the recording.
        """
        timestep_idx = self.stride * (index % self.num_strided_timesteps)
        cluster_idx = index // self.num_strided_timesteps
        return timestep_idx, cluster_idx

    def __getitem__(self, idx):
        """
        Returns the snippet at the given index.
        """
        start_time_idx, cluster_idx = self._decode_index(idx)
        end_time_idx = start_time_idx + self.snippet_len
        assert end_time_idx <= len(self.recording)
        rec = self.recording.stimulus[start_time_idx:end_time_idx].T
        spikes = self.recording.spikes[start_time_idx:end_time_idx, cluster_idx]
        res = {"stimulus": rec, "spikes": spikes, "cluster_id": cluster_idx}
        return res

    @property
    def sample_rate(self):
        return self.recording.sample_rate


class SpikeCountDataset(torch.utils.data.Dataset):
    """
    Dataset that pairs a stimulus+spike history with a future spike count.

    The spike count is for a configurable duration after the end of the
    history snippet.

    This (X,y) style dataset is intended to for basic comparison between
    different spike count models.
    """

    def __init__(
        self, recording: mea.SpikeRecording, input_len: int, output_len: int
    ):
        self.output_len = output_len
        self.input_len = input_len
        self.total_len = input_len + output_len
        self.recording = recording
        self.ds = SnippetDataset(recording, self.total_len)

    def __len__(self):
        """
        Calculates the number of samples in the dataset.

        There will be one sample for every timestep in the recording.
        """
        res = len(self.recording) - self.total_len + 1
        assert res > 0, "Snippet length is longer than the recording."
        return res

    def __getitem__(self, idx):
        """
        Returns the (X,y) sample at the given index.

        Index is one-to-one with the timesteps in the recording.
        """
        rec, spikes = self.ds[idx]
        X_stim = rec[:, 0 : self.input_len]
        X_spikes = spikes[0 : self.input_len]
        X = np.vstack((X_stim, X_spikes))
        Y = spikes[self.input_len :]
        assert Y.shape == (self.output_len,)
        y = np.sum(Y)
        return X, y


class DistFieldDataset(torch.utils.data.Dataset):
    """
    Dataset producing a snippet (spikes and stimulus) and a spike distance
    field. A portion of the spikes in the snippet are masked. The spike distance
    field will be created for the masked portion of the snippet.

    The intended usecase of this dataset is to predict spike activity given
    stimulus and spike history.
    """

    # Mask value should be negative. Zero represents no spikes, and 1+ represent
    # a spike count which can be created than 1!
    MASK_VALUE = -1
    # Do not! set a seed within the dataset. Process forking leads to identical
    # seeds.
    # RNG_SEED = 123

    # TODO: make configurable
    NOISE_SD = 0.1
    NOISE_MU = 0
    NOISE_JITTER = 4
    DROP_RATE = 0.1

    def __init__(
        self,
        recording: mea.SpikeRecording,
        snippet_len,
        mask_begin: int,
        mask_end: int,
        pad: int,
        dist_clamp: float,
        stride: int = 1,
        enable_augmentation: bool = True,
        allow_cheating: bool = False,
    ):
        self.enable_augmentation = enable_augmentation
        self.num_stim_channels = recording.stimulus.shape[1]
        self.pad = pad
        self.dist_clamp = dist_clamp
        self.ds = SnippetDataset(
            recording, snippet_len + self.pad, stride=stride
        )
        self.mask_slice = slice(mask_begin, mask_end)
        self.allow_cheating = allow_cheating
        self.stim_mean = np.expand_dims(recording.stimulus.mean(axis=0), -1)
        self.stim_sd = np.expand_dims(recording.stimulus.std(axis=0), -1)
        # The stimulus mean will be dominated by the mask
        mask_len = mask_end - mask_begin
        self.spike_mean = mask_len * self.MASK_VALUE / snippet_len
        self.spike_sd = (
            mask_len * (self.MASK_VALUE - self.spike_mean) ** 2
            + (snippet_len - mask_len) * self.spike_mean**2
        ) / snippet_len

    def __len__(self):
        return len(self.ds)

    @property
    def recording(self):
        return self.ds.recording

    @property
    def sample_rate(self):
        return self.recording.sample_rate

    @property
    def stride(self):
        return self.ds.stride

    def _augment_stimulus(self, stimulus):
        """
        Augment a stimulus portion of a sample.
        """
        # Whole block scale.
        mu = 1.0
        sd = 0.15
        scale = np.random.normal(mu, sd, size=(1,))
        # Whole block offset.
        mu = 0.0
        sigma = 0.15
        offset_noise = np.random.normal(mu, sigma, size=(1,))
        # Per bin noise.
        max_length = stimulus.shape[1]
        center, length = np.random.randint(low=0, high=max_length, size=(2,))
        left = max(0, center - length // 2)
        right = min(max_length - 1, center + length // 2 + 1)
        bin_noise = np.random.normal(
            self.NOISE_MU,
            self.NOISE_SD,
            size=(self.num_stim_channels, (right - left)),
        )
        stimulus = stimulus * scale + offset_noise
        stimulus[:, left:right] += bin_noise
        return stimulus

    def normalize_stimulus(self, stimulus):
        """
        Normalize a stimulus portion of a sample.
        """
        res = (stimulus - self.stim_mean) / self.stim_sd
        return res

    def normalize_spikes(self, spikes):
        """
        Normalize a spike portion of a sample.
        """
        res = (spikes - self.spike_mean) / self.spike_sd
        return res

    def normalize_snippet(self, snippet):
        snippet[0 : self.num_stim_channels, :] = (
            snippet[0 : self.num_stim_channels, :] - self.stim_mean
        ) / self.stim_sd
        snippet[self.num_stim_channels :, :] = (
            snippet[self.num_stim_channels :] - self.spike_mean
        ) / self.spike_sd
        return snippet

    def _augment_spikes(self, spikes):
        """
        "Augment the spike portion of a sample.

        Call this on the model input portion of the spike data, and not the
        portion that we are trying to predict.
        """
        spike_indicies = np.nonzero(spikes)
        spikes[spike_indicies] = 0
        # Add jitter
        jitter = np.random.randint(
            -self.NOISE_JITTER, self.NOISE_JITTER, len(spike_indicies)
        )
        spike_indicies = np.clip(spike_indicies + jitter, 0, len(spikes) - 1)
        # Drop some spikes.
        new_spikes = np.random.binomial(
            1, p=(1 - self.DROP_RATE), size=len(spike_indicies)
        )
        spikes[spike_indicies] = new_spikes
        return spikes

    def __getitem__(self, idx):
        """
        Returns the sample at the given index.

        +---------------------------------+-------------+
        |  a) input stimulus                            |
        +---------------------------------+-------------+---------+
        |  b) input spike data            | c) masked   | d) pad* |
        +---------------------------------+-------------+
        |  e) position encoding                         |
        +---------------------------------+-------------+

        Note (d*): there is an extra bit of spike data used when creating
        a sample, here called a pad. The pad is used to calculate the ground
        truth distance field. This bit of data is not placed in the sample that
        is returned.
        """
        # 1. Get the snippet. Make it extra long, for the distance field calc.
        sample = self.ds[idx]
        extra_long_stimulus = sample["stimulus"]
        extra_long_spikes = sample["spikes"]
        cluster_id = sample["cluster_id"]
        # For some unknown reason, the following copy call makes
        # training about 5x faster, and it has no effect when called on the
        # stimulus array. Maybe related to the copy that is done below for
        # target_spikes?
        extra_long_spikes = np.array(extra_long_spikes, copy=True, dtype=float)

        # 2. Optional augmentation.
        if self.enable_augmentation:
            extra_long_stimulus = self._augment_stimulus(extra_long_stimulus)
            extra_long_spikes[0 : self.mask_slice.start] = self._augment_spikes(
                extra_long_spikes[0 : self.mask_slice.start]
            )
        # 3. Calculate the distance field.
        dist = sdf.distance_field(extra_long_spikes, self.dist_clamp)
        # With the distance field calculated, we can throw away the extra bit.
        dist = dist[self.mask_slice]
        target_spikes = np.array(extra_long_spikes[self.mask_slice], copy=True)
        if not self.allow_cheating:
            extra_long_spikes[self.mask_slice] = self.MASK_VALUE
        # 4. Remove the extra padding that was used to calculate the distance fields.
        stimulus = extra_long_stimulus[:, 0 : -self.pad]
        spikes = extra_long_spikes[0 : -self.pad]
        # 5. Normalize
        stimulus_norm = self.normalize_stimulus(stimulus)
        # 6. Position encoding.
        pos_enc = np.log(np.arange(len(spikes)) - self.mask_slice.start)
        # TODO: what about cheating? Why now?
        # 7. Stack 
        snippet = np.vstack((stimulus_norm, 
                             # pos_enc, 
                             spikes))
        # Returning a dictionary is more flexible than returning a tuple, as
        # we can add to the dictionary without breaking existing consumers of
        # the dataset.
        res = {
            "snippet": snippet,
            "dist": dist,
            "target_spikes": target_spikes,
            "cluster_id": cluster_id,
        }
        return res


class LabeledConcatDataset(torch.utils.data.Dataset):
    """
    Dataset that concatenates datasets and inserts a dataset label.

    This is an edited version of PyTorch's ConcatDataset, with the addition
    of a dataset index being included in each sample. This is useful for
    making a multi-recording dataset easily from a list of single recording
    datasets. The PyTorch implementation wasn't sufficient, as we want to
    include information of which recording a sample belongs to.
    """

    datasets: List[torch.utils.data.Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(
        self,
        datasets: Iterable[torch.utils.data.Dataset],
        label_key: str = "id",
    ) -> None:
        super(LabeledConcatDataset, self).__init__()
        self.datasets = list(datasets)
        self.label_key = label_key
        assert len(self.datasets) > 0, (
            "datasets should not be an empty " "iterable"
        )
        for d in self.datasets:
            assert not isinstance(
                d, torch.utils.data.IterableDataset
            ), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample = self.datasets[dataset_idx][sample_idx]
        assert isinstance(
            sample, dict
        ), "Sample must be a dictionary in order to add a label."
        sample[self.label_key] = dataset_idx
        return sample


class ConcatDistFieldDataset(LabeledConcatDataset):
    """
    A concatenation of distfield datasets, one for each recording.

    This class exists to expose some methods like sample_rate() that are both
    present and consistent among all contained datasets. Using these methods
    is tedious if using LabeledConcatDataset directly. It's also brittle, as
    the LabeledConcatDataset can't (but should) act as a drop in replacement
    for a single DistFieldDataset.

    While we are at it, we can encapsulate the setting of the "rec_id" as the
    custom label key.
    """

    def __init__(self, datasets: Iterable[DistFieldDataset]) -> None:
        super().__init__(datasets, label_key="rec_id")
        # Check for consistency
        sample_rate = self.datasets[0].sample_rate
        for ds in self.datasets:
            assert ds.sample_rate == sample_rate, (
                "All datasets must have the same sample rate. Got "
                f"({sample_rate}) and ({ds.sample_rate})."
            )

    @property
    def sample_rate(self):
        return self.datasets[0].sample_rate

    @property
    def stride(self):
        return self.datasets[0].stride

