import retinapy.mea as mea
import retinapy.spikedistancefield as sdf
import numpy as np
import torch
import bisect
import math
from typing import Tuple, Optional, List, Iterable, Dict, TypeAlias


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
    _num_strided_timesteps : int
    _stride : int

    def __init__(
        self,
        recording: mea.SpikeRecording,
        snippet_len: int,
        stride: int = 1,
        rec_cluster_ids: Optional[mea.RecClusterIds] = None,
    ):
        """
        Args:
            rec_cluster_ids: an optional map to be used to determine the
                cluster id used for training.
        """
        if snippet_len > len(recording):
            raise ValueError(
                f"Snippet length ({snippet_len}) is larger than "
                f"the recording length ({len(recording)})."
            )
        self.recording = recording
        self.snippet_len = snippet_len
        self.num_clusters = len(recording.cluster_ids)
        self.num_timesteps = len(recording) - snippet_len + 1
        self.rec_cluster_ids = rec_cluster_ids
        assert (
            self.num_timesteps > 0
        ), "Snippet length is longer than the recording."
        self.stride = stride

    def __len__(self):
        """
        Calculates the number of samples in the dataset.
        """
        return self._num_strided_timesteps * self.num_clusters

    def _decode_index(self, index: int) -> Tuple[int, int, int]:
        """
        Decodes the index into the timestep and cluster id.

        The data is effectively a 2D array with dimensions (time, cluster).
        The index is the flattened index of this array, and so the timestep
        increases as the index increases and wraps to the next cluster id when
        it reaches the end of the recording.
        """
        timestep_idx = self.stride * (index % self._num_strided_timesteps)
        cluster_idx = index // self._num_strided_timesteps
        if self.rec_cluster_ids is not None:
            cluster_id = self.rec_cluster_ids[
                (self.recording.name, self.recording.cluster_ids[cluster_idx])
            ]
        else:
            cluster_id = cluster_idx
        return timestep_idx, cluster_idx, cluster_id

    def __getitem__(self, idx):
        """
        Returns the snippet at the given index.
        """
        start_time_idx, cluster_idx, cluster_id = self._decode_index(idx)
        end_time_idx = start_time_idx + self.snippet_len
        assert end_time_idx <= len(
            self.recording
        ), f"{end_time_idx} <= {len(self.recording)}"
        rec = self.recording.stimulus[start_time_idx:end_time_idx].T
        spikes = self.recording.spikes[start_time_idx:end_time_idx, cluster_idx]
        res = {"stimulus": rec, "spikes": spikes, "cluster_id": cluster_id}
        return res

    @property
    def sample_rate(self):
        return self.recording.sample_rate

    @property
    def stride(self):
        return self._stride

    @stride.setter
    def stride(self, stride : int):
        self._stride = stride
        self._num_strided_timesteps = _num_snippets(
            len(self.recording), self.snippet_len, stride
        )

    @property
    def num_strided_timesteps(self):
        return self._num_strided_timesteps
        



class SpikeCountDataset(torch.utils.data.Dataset):
    """
    Dataset that pairs a stimulus+spike history with a future spike count.

    The spike count is for a configurable duration after the end of the
    history snippet.

    This (X,y) style dataset is intended to for basic comparison between
    different spike count models.
    """

    def __init__(
        self,
        recording: mea.SpikeRecording,
        input_len: int,
        output_len: int,
        stride: int = 1,
    ):
        self.output_len = output_len
        self.input_len = input_len
        self.total_len = input_len + output_len
        self.recording = recording
        self.ds = SnippetDataset(recording, self.total_len, stride)

    def __len__(self):
        """
        Calculates the number of samples in the dataset.

        There will be one sample for every timestep in the recording.
        """
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Returns the (X,y) sample at the given index.

        Index is one-to-one with the timesteps in the recording.
        """
        sample = self.ds[idx]
        rec = sample["stimulus"]
        spikes = sample["spikes"]
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
    MASK2_VALUE = -0.5
    MASK3_VALUE = 10
    # Do not! set a seed within the dataset. Process forking leads to identical
    # seeds.
    # RNG_SEED = 123

    # TODO: make configurable
    NOISE_SD = 0.7
    NOISE_MU = 0
    NOISE_JITTER = 4
    DROP_RATE = 0.1
    STIM_MASK_RATE = 0.0
    SPIKE_MASK_RATE = 0.2
    MIXUP_P = 0.4

    def __init__(
        self,
        recording: mea.SpikeRecording,
        snippet_len,
        mask_begin: int,
        mask_end: int,
        pad: int,
        dist_clamp: float,
        stride: int = 1,
        rec_cluster_ids: Optional[mea.RecClusterIds] = None,
        enable_augmentation: bool = True,
        allow_cheating: bool = False,
    ):
        self.enable_augmentation = enable_augmentation
        self.num_stim_channels = recording.stimulus.shape[1]
        self.pad = pad
        self.dist_clamp = dist_clamp
        self.ds = SnippetDataset(
            recording, snippet_len + self.pad, stride, rec_cluster_ids
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

    # Setter property for stride
    @stride.setter
    def stride(self, stride: int):
        self.ds.stride = stride

    @classmethod
    def mask_start(cls, spikes):
        mask_val = cls.MASK_VALUE
        mask_start_idx = np.min(np.flatnonzero(spikes == mask_val))
        return mask_start_idx

    def _augment_stimulus(self, stimulus):
        """
        Augment a stimulus portion of a sample.
        """
        # Whole block scale.
        mu = 1.0
        sd = 0.10
        scale = np.random.normal(mu, sd, size=(1,))
        # Whole block offset.
        mu = 0.0
        sigma = 0.10
        offset_noise = np.random.normal(mu, sigma, size=(1,))
        # Per bin noise.
        max_length = stimulus.shape[1]
        center, length = np.random.randint(low=0, high=max_length, size=(2,))
        #left = max(0, center - length // 2)
        #right = min(max_length - 1, center + length // 2 + 1)
        left, right = (0, max_length - 1)
        bin_noise = np.random.normal(
            self.NOISE_MU,
            self.NOISE_SD,
            size=(self.num_stim_channels, (right - left)),
        )
        stimulus = stimulus * scale + offset_noise
        stimulus[:, left:right] += bin_noise
        # Mask some parts.
        mask_indicies = np.nonzero(
            np.random.binomial(1, p=self.STIM_MASK_RATE, size=len(stimulus))
        )
        stimulus[:, mask_indicies] = self.MASK3_VALUE
        # Mixup
        mix_idx = np.random.randint(0, len(self))
        mixup = self.ds[mix_idx]['stimulus'][:, 0:stimulus.shape[1]]
        stimulus = stimulus * (1 - self.MIXUP_P) + mixup * self.MIXUP_P
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
        # Mask some parts.
        mask_indicies = np.nonzero(
            np.random.binomial(1, p=self.SPIKE_MASK_RATE, size=len(spikes))
        )
        spikes[mask_indicies] = self.MASK2_VALUE
        return spikes

    def __getitem__(self, idx):
        """
        Returns the sample at the given index.

        +---------------------------------+-------------+
        |  a) input stimulus                            |
        +---------------------------------+-------------+---------+
        |  b) input spike data            | c) masked   | d) pad* |
        +---------------------------------+-------------+---------+

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
        # TODO: what about cheating? Why now?
        # 6. Stack
        snippet = np.vstack((stimulus_norm, spikes))
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
        r = []
        s = 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(
        self,
        datasets: Iterable[torch.utils.data.Dataset],
        label_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            label_key: If not None, the dataset index will be included in each
                sample, under this key. If None, the dataset index will not be
                included. Example, label_key = "id". This was originally added
                to allow the recording id to be included in the sample when
                multiple recordings form a concatenated dataset. However,
                to allow for more flexibility in what recordings are loaded in
                train vs. test time, this responsibility has been delegated to
                the underlying per-recording dataset. We do however need to
                enable the recording identifier, as it is disabled by default.
                At the moment, this is expected to be done to each dataset
                before they are passed in here. With label_key = None, this
                class is equivalent to the PyTorch ConcatDataset. Might end
                up removing it if the label key isn't needed anymore.
        """
        super(LabeledConcatDataset, self).__init__()
        self.datasets = list(datasets)
        self.label_key = label_key
        assert len(self.datasets) > 0, (
            "datasets should not be an empty " "iterable"
        )
        self._calc_size()

    def _calc_size(self):
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
        if self.label_key:
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
        super().__init__(datasets, label_key=None)
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

    @stride.setter
    def stride(self, stride):
        for ds in self.datasets:
            ds.stride = stride
        self._calc_size()

    @property
    def num_recordings(self):
        return len(self.datasets)

    @property
    def num_clusters(self):
        res = sum(d.recording.num_clusters() for d in self.datasets)
        return res
