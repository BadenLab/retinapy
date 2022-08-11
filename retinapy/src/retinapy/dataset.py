import retinapy.mea as mea
import retinapy.spikedistancefield as sdf
import numpy as np
import torch
from typing import Tuple, Optional


class SnippetDataset(torch.utils.data.Dataset):
    def __init__(self, recording: mea.SpikeRecording, snippet_len: int,
                 cluster_idx: int):
        if snippet_len > len(recording):
            raise ValueError(
                f"Snippet length ({snippet_len}) is larger than "
                f"the recording length ({len(recording)})."
            )
        self.snippet_len = snippet_len
        self.recording = recording
        self.cluster_idx = cluster_idx

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
        spikes = self.recording.spikes[begin_idx:end_idx, self.cluster_idx]
        return rec.T, spikes


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
    NOISE_SD = 0.15
    NOISE_JITTER = 1
    DROP_RATE = 0.05

    def __init__(
        self,
        recording: mea.SpikeRecording,
        snippet_len,
        cluster_idx,
        mask_begin: int,
        mask_end: int,
        pad: int,  
        dist_clamp: float,
        dist_norm: float = 1,
        enable_augmentation: bool = True,
        allow_cheating: bool = False,
    ):
        self.enable_augmentation = enable_augmentation
        self.rng = np.random.default_rng(self.RNG_SEED)
        self.pad = pad
        self.dist_norm = dist_norm
        self.dist_clamp = dist_clamp
        self.ds = SnippetDataset(recording, snippet_len + self.pad, 
                                 cluster_idx)
        self.mask_slice = slice(mask_begin, mask_end)
        self.allow_cheating = allow_cheating

    def __len__(self):
        return len(self.ds)

    def _stimulus_transform(self, stimulus):
        """
        Augment a stimulus portion of a sample.
        """
        mu = 1.0
        noise = self.rng.normal(mu, self.NOISE_SD)
        res = stimulus * noise
        return res

    def _spikes_transform(self, spikes):
        """
        "Augment the spike portion of a sample.

        Call this on the model input portion of the spike data, and not the
        portion that we are trying to predict.
        """
        spike_indicies = np.nonzero(spikes)
        spikes[spike_indicies] = False
        # Add jitter
        jitter = self.rng.integers(-self.NOISE_JITTER, self.NOISE_JITTER, 
                                   len(spike_indicies))
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
        |  b) input spike data            | c) masked   |   d*    |
        +---------------------------------+-------------+---------+

        Note (d*): there is an extra bit of spike data used when creating 
        a sample. The trailing bit marked with (d*) is used to calculate the
        ground truth distance field. This bit of data is not placed in the
        sample that is returned.
        """
        # 1. Get the snippet. Make it extra long, for the distance field calc.
        extra_long_stimulus, extra_long_spikes = self.ds[idx]
        extra_long_spikes = extra_long_spikes.astype(int)
        # 2. Optional augmentation.
        if self.enable_augmentation:
            extra_long_stimulus = self._stimulus_transform(extra_long_stimulus)
            extra_long_spikes[
                0 : self.mask_slice.start
            ] = self._spikes_transform(
                extra_long_spikes[0 : self.mask_slice.start]
            )
        # 3. Calculate the distance field.
        dist = sdf.distance_field(extra_long_spikes, self.dist_clamp)
        # With the distance field calculated, we can throw away the extra bit.
        dist = dist[self.mask_slice]
        normed_dist = dist / self.dist_norm
        target_spikes = np.array(extra_long_spikes[self.mask_slice], copy=True)
        if not self.allow_cheating:
            extra_long_spikes[self.mask_slice] = self.MASK_VALUE
        extra_long_snippet = np.vstack((extra_long_stimulus, extra_long_spikes))
        # Remove the extra padding that was used to calculate the distance fields.
        snippet = extra_long_snippet[:, 0 : -self.pad]
        return snippet, target_spikes, normed_dist

