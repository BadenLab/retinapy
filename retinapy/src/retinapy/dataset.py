import retinapy.mea as mea
import retinapy.spikedistancefield as sdf
import numpy as np
import torch
from typing import Tuple, Optional


class SnippetDataset(torch.utils.data.Dataset):
    def __init__(self, recording: mea.SpikeRecording, snippet_len: int):
        if snippet_len > len(recording):
            raise ValueError(
                f"Snippet length ({snippet_len}) is larger than "
                f"the recording length ({len(recording)})."
            )
        self.snippet_len = snippet_len
        self.recording = recording
        assert (
            len(recording.cluster_ids) == 1
        ), "For the moment, we only support a single cluster."

    def __len__(self):
        """
        Calculates the number of samples in the dataset.

        There will be one sample for every timestep in the recording.
        """
        res = len(self.recording) - self.snippet_len + 1
        assert res > 0, "Snippet length is longer than the recording."
        return res

    def __getitem__(self, idx):
        """
        Returns the snippet at the given index.

        Index is one-to-one with the timesteps in the recording.
        """
        begin_idx = idx
        end_idx = idx + self.snippet_len
        assert end_idx <= len(self.recording)
        rec = self.recording.stimulus[begin_idx:end_idx]
        assert (
            len(self.recording.cluster_ids) == 1
        ), "For the moment, we only support a single cluster."
        spikes = self.recording.spikes[begin_idx:end_idx, 0]
        return rec.T, spikes


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


class SpikeDistanceFieldDataset(torch.utils.data.Dataset):
    """
    Dataset producing a snippet (spikes and stimulus) and a spike distance
    field. A portion of the spikes in the snippet are masked. The spike distance
    field will be created for the masked portion of the snippet.

    The intended usecase of this dataset is to predict spike activity given
    stimulus and spike history.
    """

    MASK_VALUE = 2
    RNG_SEED = 123

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
        enable_augmentation: bool = True,
        allow_cheating: bool = False,
    ):
        self.enable_augmentation = enable_augmentation
        self.num_stim_channels = recording.stimulus.shape[1]
        self.rng = np.random.default_rng(self.RNG_SEED)
        self.pad = pad
        self.dist_clamp = dist_clamp
        self.ds = SnippetDataset(recording, snippet_len + self.pad)
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

    def _augment_stimulus(self, stimulus):
        """
        Augment a stimulus portion of a sample.
        """
        # Whole block scale.
        mu = 1.0
        sd = 0.15
        scale = self.rng.normal(mu, sd, size=(1,))
        # Whole block offset.
        mu = 0.0
        sigma = 0.15
        offset_noise = self.rng.normal(mu, sigma, size=(1,))
        # Per bin noise.
        max_length = stimulus.shape[1]
        center, length = self.rng.integers(low=0, high=max_length, size=(2,))
        left = max(0, center - length // 2)
        right = min(max_length-1, center + length // 2 + 1)
        bin_noise = self.rng.normal(self.NOISE_MU, self.NOISE_SD, 
                                size=(self.num_stim_channels, (right-left)))
        stimulus = (stimulus * scale + offset_noise)
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
        jitter = self.rng.integers(
            -self.NOISE_JITTER, self.NOISE_JITTER, len(spike_indicies)
        )
        spike_indicies = np.clip(spike_indicies + jitter, 0, len(spikes) - 1)
        # Drop some spikes.
        new_spikes = self.rng.binomial(
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
        +---------------------------------+-------------+---------+

        Note (d*): there is an extra bit of spike data used when creating
        a sample, here called a pad. The pad is used to calculate the ground
        truth distance field. This bit of data is not placed in the sample that
        is returned.
        """
        # 1. Get the snippet. Make it extra long, for the distance field calc.
        extra_long_stimulus, extra_long_spikes = self.ds[idx]
        # For some unknown reason, the following copy call makes 
        # training about 5x faster, and it has no effect when called on the
        # stimulus array. Maybe related to the copy that is done below for
        # target_spikes?
        extra_long_spikes = np.array(extra_long_spikes, copy=True)

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
        #spikes_norm = self.normalize_spikes(spikes)
        # TODO: what about cheating? Why now?
        # 6. Stack stimulus and spike data to make X.
        snippet = np.vstack((stimulus_norm, spikes))
        return snippet, target_spikes, dist
