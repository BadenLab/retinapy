import argparse
from collections import defaultdict
import logging
import pathlib
from typing import Iterable
import math

import yaml

import retinapy
import retinapy._logging
import retinapy.dataset
import retinapy.vis
import retinapy.mea as mea
import retinapy.models
import retinapy.train
import retinapy.nn
import scipy
import torch
import plotly.io


DEFAULT_OUT_BASE_DIR = "./out/"
LOG_FILENAME = "train.log"
ARGS_FILENAME = "args.yaml"
TENSORBOARD_DIR = "tensorboard"

IN_CHANNELS = 4 + 1
# Quite often there are lengths in the range 300.
# The pad acts as the maximum, so it's a good candidate for a norm factor.
# Example: setting normalization to 400 would cause 400 time steps to be fit
# into the [0,1] region.
LOSS_CALC_PAD_MS = 600
DIST_CLAMP_MS = 600
SPLIT_RATIO = (3, 1, 1)

_logger = logging.getLogger(__name__)


def parse_args():
    """Parse commandline and config file arguments.

    The approach carried out here is inspired by the pytorch-image-models
    project:
        https://github.com/rwightman/pytorch-image-models

    Arguments are populated in the following order:
        1. Default values
        2. Config file
        3. Command line
    """
    config_parser = argparse.ArgumentParser(
        description="Config from YAML", add_help=False
    )
    config_parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        metavar="FILE",
        help="YAML config file to override argument defaults.",
    )

    parser = argparse.ArgumentParser(description="Spike detection training")
    # fmt: off

    # Model/config arguments
    # Using -k as a filter, just like pytest.
    parser.add_argument("-k", type=str, default=None, metavar="EXPRESSION", help="Filter configs and models to train or test.")

    # Optimization parameters
    opt_group = parser.add_argument_group("Optimizer parameters")
    opt_group.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    opt_group.add_argument("--weight-decay", type=float, default=1e-6, help="weight decay (default: 2e-5)")

    # Data
    data_group = parser.add_argument_group("Data parameters")
    data_group.add_argument("--stimulus-pattern", type=str, default=None, metavar="FILE", help="Path to stimulus pattern file.")
    data_group.add_argument("--stimulus", type=str, default=None, metavar="FILE", help="Path to stimulus recording file.")
    data_group.add_argument("--response", type=str, default=None, metavar="FILE", help="Path to response recording file.")
    data_group.add_argument("--recording-names", nargs='+', default=None, help="Names of recordings within the recording file.")
    data_group.add_argument("--cluster-id", type=int, default=None, help="Cluster ID to train on.")

    parser.add_argument("--steps-til-eval", type=int, default=None, help="Steps until validation.")
    parser.add_argument("--steps-til-log", type=int, default=100, help="How many batches to wait before logging a status update.")
    parser.add_argument("--steps-til-eval-test-ds", type=int, default=10, help="After how many validation runs with the validation data should validation be run with the training data.")
    parser.add_argument("--initial-checkpoint", type=str, default=None, help="Initialize model from the checkpoint at this path.")
    #parser.add_argument("--resume", type=str, default=None, help="Resume full model and optimizer state from checkpoint path.")
    parser.add_argument("--output", type=str, default=None, metavar="DIR", help="Path to output folder (default: current dir).")
    parser.add_argument("--labels", type=str, default=None, help="List of experiment labels. Used for naming files and/or subfolders.")
    parser.add_argument("--epochs", type=int, default=8, metavar="N", help="number of epochs to train (default: 300)")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    # fmt: on

    # First check if we have a config file to deal with.
    args, remaining = config_parser.parse_known_args()
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            # Populate the main parser's defaults.
            parser.set_defaults(**config)
    # Now the main parser.
    opt = parser.parse_args(remaining)
    # Serialize the arguments.
    opt_text = yaml.safe_dump(opt.__dict__, default_flow_style=False)
    return opt, opt_text


class Configuration:
    def __init__(self, downsample, input_len, output_len):
        self.downsample = downsample
        self.input_len = input_len
        self.output_len = output_len

    def __str__(self):
        return f"{self.downsample}ds_{self.input_len}in_{self.output_len}out"


"""
The following table is useful as a reference to see the effects of each 
downsampling factor. The downsampling rates in the table were chosen to fall
closely to some "simple" frequencies/periods, like 1 kHz, etc. 

Downsample factors
==================
Downsample : Freq (Hz)   : Timestep period (ms)
1          : 17852.768   : 
2          :  8926.389   :
4          :  4463.192   :  0.224
9          :  2231.596   :  0.504
18         :   991.820   :  1.001
36         :   495.910   :  2.016
71         :   251.447   :  3.977
89         :   200.593   :  4.985
143        :   124.845   :  8.001
179        :    99.736   : 10.026
"""
# Roughly, we are aiming for the following:
downsample_factors = [9, 18, 89, 179]
input_lengths_ms = [1000, 1600]
output_lenghts_ms = [1, 10, 50, 100]

"""
The following are the model input-output configurations we will use. The
three variables are:

    - downsampling factor
    - input length 
    - output length

Given that each model will be trained for each configuration, it is not 
feasible to have a large number of configurations. So some care has been 
given to pick an initial set that will hopefully be interesting and give us
some insight into the problem. After this, we should be able to narrow in
on a smaller set of configurations. 

There are a few considerations driving the choices for these variables.

Downsample factor
-----------------
We are not sure at what point downsampling causes loss of important
information. Ideally, we would like to work with a low sample rate. Trying with
a few different rates will help us to determine the trade-offs at each
downsampling factor. It's likely that further experiments can work with a
single sampling rate.

Input length
------------
I am told that previous experiments have found that retinal ganglion cells can
depend on the past 1200 ms of input. At least one of the papers I am trying to
benchmark against used much shorter inputs (400ms). I am setting a high input
rate of 1600 to give a decent buffer so that we can support/test an argument
that no more than X ms seem to be used, where X is currently hypothesized to be
around 1200. I am setting a low input length of 1000 ms to test if 1000 is
insufficient. The choice of 1000 is somewhat arbitrary. If 1000 is
insufficient, we can justify working with a shorter 1000 ms input, which is a
win from an engineering point of view. If 1000 is insufficient, then this is
nice evidence to support the hypothesis that >1000 ms are used. If we had
chosen 400 ms, this result would not be interesting, as I think it is widely
accepted that the ganglion cells depend on more than the last 400 ms. So 1000
was chosen to try be a win-win: either an engineering win, or a win from the
point of view of having interesting evidence.

Output length
-------------
Varying the output duration is a way to test how temporally precise a model
is. The output duration represents the duration over which spikes will be
summed to calculate the "spike count" for the output interval. In addition,
varying output duration allows us to test our evaluation metrics. For example,
what is the relationship between accuracy, false positives, false negatives,
and any correlation measures. The first set of experiments are using 
[1, 10, 50, 100] ms output durations. The 1 ms output duration is expected to
be too difficult to model, while the 100 ms output is expected to be easy.
So, these two extremes will act as a sanity check, helping to identify any
strange behaviour. The 10 ms output was chosen as I have seen it in other 
literature, so if will be useful for comparison. After these 3 durations,
I wasn't sure what to choose. Given that it's the first set of experiments,
I'm not expecting amazing models, so periods greater than 10 ms might be 
more useful for comparison than the more difficult shorter periods. I hope
we can get to the point where periods between 5 ms and 10 ms are interesting.
"""
all_configs = tuple(
    Configuration(*c)
    for c in [
        # 0.504 ms bins.
        #   1000.18 ms input.
        #        1.008 ms output
        (9, 1984, 2),
        #       10.82 ms output
        (9, 1984, 20),
        #       21.60 ms output
        (9, 1984, 100),
        #       50.41 ms output
        (9, 1984, 198),
        #       99.82 ms output
        #   1600.09 ms input.
        #        1.008 ms output
        (9, 3174, 2),
        #       10.82 ms output
        (9, 3174, 20),
        #       21.60 ms output
        (9, 3174, 100),
        #       50.41 ms output
        (9, 3174, 198),
        #       99.82 ms output
        # 1.001 ms bins.
        #   1000.18 ms input.
        #        1.008 ms output
        (18, 992, 1),
        #       10.08 ms output
        (18, 992, 10),
        #       50.41 ms output
        (18, 992, 50),
        #      100.82 ms output
        (18, 992, 100),
        #   1599.08 ms input.
        #        1.008 ms output
        (18, 1586, 1),
        #       10.08 ms output
        (18, 1586, 10),
        #       50.41 ms output
        (18, 1586, 50),
        #      100.82 ms output
        (18, 1586, 100),
        # Downsample by 89, giving 4.985 ms bins. At this rate, we can't
        # output 1 ms bins, so there are only 6 configurations for this
        # downsample factor.
        # 4.985 ms bins
        #    997.04 ms input
        #    Alternative is the closer 1.002 ms with 201 bins, but going with
        #    201 bins to try and keep the input/output bins even numbers.
        #       9.970 ms output
        (89, 200, 2),
        #       49.85 ms output
        (89, 200, 10),
        #       99.70 ms output
        (89, 200, 20),
        #   1595.27 ms input
        #   Alternative is the 1600.27 ms, with 321 bins, but going with 320
        #   bins to try and keep the input/output bins even numbers.
        #       9.970 ms output
        (89, 320, 2),
        #       49.85 ms output
        (89, 320, 10),
        #       99.70 ms output
        (89, 320, 20),
        # Downsample by 179, giving 10.026 ms bins. Same as with 89, we can't
        # output 1 ms bins, so there are only 6 configurations for this
        # downsample factor.
        # 10.026 ms bins
        #   1002.65 ms input
        #       10.037 ms output
        (179, 100, 1),
        #       20.053 ms output
        (179, 100, 2),
        #       50.132 ms output
        (179, 100, 5),
        #       10.037 ms output
        #   1604.22 ms input
        (179, 160, 1),
        #       20.053 ms output
        (179, 160, 2),
        #       50.132 ms output
        (179, 160, 5),
    ]
)

# To achieve the above, we can run the following function, get_configurations();
# however, the input and output lengths are inconvenient to work with. So,
# we will specify each configuration manually.

# The precise configurations would have been:
#
#     downsample, input bins, output bins
#     9 ds, 1983.6409 in, 1.9836 out
#     9 ds, 1983.6409 in, 19.8364 out
#     9 ds, 1983.6409 in, 99.1820 out
#     9 ds, 1983.6409 in, 198.3641 out
#     9 ds, 3173.8254 in, 1.9836 out
#     9 ds, 3173.8254 in, 19.8364 out
#     9 ds, 3173.8254 in, 99.1820 out
#     9 ds, 3173.8254 in, 198.3641 out
#     18 ds, 991.8204 in, 9.9182 out
#     18 ds, 991.8204 in, 49.5910 out
#     18 ds, 991.8204 in, 99.1820 out
#     18 ds, 1586.9127 in, 9.9182 out
#     18 ds, 1586.9127 in, 49.5910 out
#     18 ds, 1586.9127 in, 99.1820 out
#     89 ds, 200.5929 in, 2.0059 out
#     89 ds, 200.5929 in, 10.0296 out
#     89 ds, 200.5929 in, 20.0593 out
#     89 ds, 320.9486 in, 2.0059 out
#     89 ds, 320.9486 in, 10.0296 out
#     89 ds, 320.9486 in, 20.0593 out
#     etc


def ms_to_num_bins(time_ms, downsample_factor):
    res = time_ms * (mea.ELECTRODE_FREQ / 1000) / downsample_factor
    return res


def get_configurations():
    res = []
    for downsample in downsample_factors:
        for in_len in input_lengths_ms:
            in_bins = ms_to_num_bins(in_len, downsample)
            for out_len in output_lenghts_ms:
                out_bins = ms_to_num_bins(out_len, downsample)
                if in_bins < 1 or out_bins < 1:
                    # Not enough resolution at this downsample factor.
                    continue
                in_bins_int = round(in_bins)
                out_bins_int = round(out_bins)
                res.append(Configuration(downsample, in_bins_int, out_bins_int))
    return res


class LinearNonLinearTrainable(retinapy.train.Trainable):
    def __init__(self, train_ds, val_ds, test_ds, model, model_label):
        super(LinearNonLinearTrainable, self).__init__(
            train_ds, val_ds, test_ds, model, model_label
        )
        self.loss_fn = torch.nn.PoissonNLLLoss(log_input=False)

    def forward(self, sample):
        X, y = sample
        X = X.float().cuda()
        y = y.float().cuda()
        model_output = self.model(X)
        loss = self.loss_fn(model_output, target=y)
        return model_output, loss

    def evaluate(self, val_dl):
        predictions = []
        targets = []
        # Note: the loss per batch is averaged, so we are averaging this again
        # as we loop through each batch.
        loss_meter = retinapy._logging.Meter("loss")
        for (X, y) in val_dl:
            X = X.float().cuda()
            y = y.float().cuda()
            model_output = self.model(X)
            loss_meter.update(
                self.loss_fn(model_output, target=y).item(), y.shape[0]
            )
            predictions.append(model_output.cpu())
            targets.append(y.cpu())
        # Don't forget to check if the model output is log(the Poisson λ parameter)
        # or not log!
        predictions = torch.round(torch.cat(predictions))
        targets = torch.cat(targets)
        acc = (predictions == targets).float().mean().item()
        pearson_corr = scipy.stats.pearsonr(predictions, targets)[0]
        metrics = [
            retinapy._logging.Metric("loss", loss_meter.avg, increasing=False),
            retinapy._logging.Metric("accuracy", acc),
            retinapy._logging.Metric("pearson_corr", pearson_corr),
        ]
        return metrics


class DistFieldTrainable_(retinapy.train.Trainable):
    """
    A base trainable for the distance field models.

    There are quite a few things that are common to all such models, such as:
        - the conversion from distance field to model output (and vice versa)
        - the evaluation proceedure
        - the requirement to wrap properties like sample rate that would
          otherwise not be contained in one obvious place.
    """

    eval_lengths_ms: Iterable[int]
    max_eval_count: int

    """
    min_dist has duel purpose:
        a) it represents the fact that a bin that contains a spike means that
           when viewing the bin as a sample point, the actual spike could have
           occurred anywhere ± 1/2 the sample period. Thus, for a bin that
           contains a spike, zero distance is not appropriate—it should be
           the expected value of the distance, which if a uniform distribution
           is assumed, then the distance is 1/2.
        b) it caps the error for a prediction for a bin that contains a spike.
           When using the log distance as the target, a distance of zero will 
           give -∞ as a target. So we should choose some minimum distance. 
           Enter 0.5 as a reasonable minimum given the justification from
           the previous point.

    TODO: I think that due to the explanation in point a) above, the value
    of 1/2 should be set by the dataset, and not here. The reason for delaying
    this now is that quite a bit of changes would need to be made (tests etc.)
    """
    MIN_DIST: float = 0.5
    DEFAULT_EVAL_LENGTHS_MS = [10, 50, 100]
    DEFAULT_MAX_EVAL_COUNT = int(1e7)
    DEFAULT_DIST_NORM = 20

    def __init__(
        self, train_ds, val_ds, test_ds, model, model_label, eval_lengths=None
    ):
        super().__init__(train_ds, val_ds, test_ds, model, model_label)
        sample_rates_equal = (
            train_ds.sample_rate == test_ds.sample_rate == val_ds.sample_rate
        )
        if not sample_rates_equal:
            raise ValueError(
                "Train, validation and test datasets do not have "
                f"the same sample rate. Got ({train_ds.sample_rate:.5f}, "
                f"{val_ds.sample_rate:.5f}, {test_ds.sample_rate:.5f})"
            )
        if eval_lengths is None:
            self.eval_lengths_ms = self.DEFAULT_EVAL_LENGTHS_MS
        else:
            self.eval_lengths_ms = eval_lengths
        self.max_eval_count = self.DEFAULT_MAX_EVAL_COUNT
        self.dist_norm = self.DEFAULT_DIST_NORM
        self.num_plots = 16

    @property
    def in_device(self):
        res = next(self.model.parameters()).device
        return res

    def eval_bins(self) -> Iterable[int]:
        res = []
        for ms in self.eval_lengths_ms:
            num_bins = max(1, round(ms * (self.sample_rate / 1000)))
            res.append(num_bins)
        return res

    @property
    def sample_period_ms(self):
        return 1000 / self.sample_rate

    @property
    def sample_rate(self):
        sample_rates_equal = (
            self.train_ds.sample_rate
            == self.test_ds.sample_rate
            == self.val_ds.sample_rate
        )
        assert sample_rates_equal
        # We can use any of the datasets.
        return self.train_ds.sample_rate

    def distfield_to_nn_output(self, distfield):
        return torch.log((distfield + self.MIN_DIST) / self.dist_norm)

    def nn_output_to_distfield(self, nn_output):
        return torch.exp(nn_output) * self.dist_norm - self.MIN_DIST

    def quick_infer(self, dist, eval_len):
        """Quickly infer the number of spikes in the eval region.

        An approximate inference used for evaluation.

        Returns:
            the number of spikes in the region.
        """
        threshold = 0.4
        res = (dist[:, 0:eval_len] < threshold).sum(dim=1)
        return res

    def evaluate(self, val_dl):
        predictions = defaultdict(list)
        targets = defaultdict(list)
        loss_meter = retinapy._logging.Meter("loss")
        plotly_figs = []
        for i, sample in enumerate(val_dl):
            plotly.io.write_image(
                self.input_output_fig(sample, self.forward(sample)[0], idx=0),
                "./out/fig_debug.png",
                format="png",
            )
            # Don't run out of memory, or take too long.
            num_so_far = val_dl.batch_size * i
            if num_so_far > self.max_eval_count:
                break
            target_spikes = sample["target_spikes"].float().cuda()
            batch_len = target_spikes.shape[0]
            model_output, loss = self.forward(sample)
            loss_meter.update(loss.item(), batch_len)
            # Count accuracies
            for eval_len in self.eval_bins():
                pred = self.quick_infer(model_output, eval_len=eval_len)
                y = torch.sum(target_spikes[:, 0:eval_len], dim=1)
                predictions[eval_len].append(pred)
                targets[eval_len].append(y)
            # Plot some example input-outputs
            if len(plotly_figs) < self.num_plots:
                # Plot the first batch element.
                idx = 0
                plotly_figs.append(
                    self.input_output_fig(sample, model_output, idx)
                )

        metrics = [
            retinapy._logging.Metric("loss", loss_meter.avg, increasing=False)
        ]
        for eval_len in self.eval_bins():
            p = torch.cat(predictions[eval_len])
            t = torch.cat(targets[eval_len])
            acc = (p == t).float().mean().item()
            pearson_corr = scipy.stats.pearsonr(
                p.cpu().numpy(), t.cpu().numpy()
            )[0]
            metrics.append(
                retinapy._logging.Metric(f"accuracy-{eval_len}_bins", acc)
            )
            metrics.append(
                retinapy._logging.Metric(
                    f"pearson_corr-{eval_len}_bins", pearson_corr
                )
            )
        results = {
            "metrics": metrics,
            "plotly": retinapy._logging.LogData(
                "input-output-fig", plotly_figs
            ),
        }
        return results

    def input_output_fig(self, sample, model_out, idx):
        target_dist = self.distfield_to_nn_output(sample["dist"][idx])
        fig = retinapy.vis.distfield_model_in_out(
            # Stimulus takes up all channels except the last.
            stimulus=sample["snippet"][idx][0:-1].cpu().numpy(),
            in_spikes=sample["snippet"][idx][-1].cpu().numpy(),
            target_dist=target_dist.cpu().numpy(),
            model_out=model_out[idx].cpu().numpy(),
            start_ms=0,
            bin_duration_ms=self.sample_period_ms,
            cluster_label=(
                f"rec idx: {sample['rec_id'][idx]},"
                f"cluster idx: {sample['cluster_id'][idx]}"
            ),
        )
        # The legend takes up a lot of space, so disable it.
        fig.update_layout(showlegend=False)
        return fig


class DistFieldTrainableMC(DistFieldTrainable_):
    def __init__(
        self, train_ds, val_ds, test_ds, model, model_label, eval_lengths
    ):
        """Trainable for a multi-cluster distance field model.

        This is a minimal extension of the DistFieldTrainable.
        """
        super().__init__(
            train_ds, val_ds, test_ds, model, model_label, eval_lengths
        )
        self.dist_loss_fn = retinapy.models.dist_loss
        # Network output should ideally have mean,sd = (0, 1). Network output
        # 20*exp([-3, 3])  = [1.0, 402], which is a pretty good range, with
        # 20 being the mid point. Is this too low?
        self.dist_norm = 20
        # Network output should ideally have mean,sd = (0, 1). Network output
        # 20*exp([-3, 3])  = [1.0, 402], which is a pretty good range, with
        # 20 being the mid point. Is this too low?
        self.max_eval_count = int(1e5)

    def loss(self, m_dist, z_mu, z_lorvar, target):
        batch_size = m_dist.shape[0]
        # Scale to get roughly in the ballpark of 1.
        dist_loss = self.dist_loss_fn(m_dist, target)
        kl_loss = -0.5 * torch.sum(1 + z_lorvar - z_mu.pow(2) - z_lorvar.exp())
        β = 1
        # β = 0
        batch_loss = dist_loss + β * kl_loss
        batch_ave_loss = batch_loss / batch_size
        return batch_ave_loss

    def encode(
        self, rec_idxs: torch.LongTensor, cluster_idxs: torch.LongTensor
    ):
        """
        Return the latent representation of the given recording-cluster pairs.

        Args:
            rec_idxs: a batched tensor of recording indexes.
            cluster_idxs: a batch tensor of cluster indexes.
        """
        # Maybe we should make a separate non-sampling function, but it's fine
        # for now.
        rec_idxs = rec_idxs.to(self.in_device)
        cluster_idxs = cluster_idxs.to(self.in_device)
        _, z, _ = self.model.encode(rec_idxs, cluster_idxs)
        return z

    def forward(self, sample):
        masked_snippet = sample["snippet"].float().to(self.in_device)
        dist = sample["dist"].float().to(self.in_device)
        rec_id = sample["rec_id"].int().to(self.in_device)
        cluster_id = sample["cluster_id"].int().to(self.in_device)
        m_dist, z_mu, z_logvar = self.model(masked_snippet, rec_id, cluster_id)
        # Dist model
        y = self.distfield_to_nn_output(dist)
        loss = self.loss(m_dist, z_mu, z_logvar, target=y)
        return m_dist, loss


class DistFieldTrainable(DistFieldTrainable_):
    def __init__(
        self, train_ds, val_ds, test_ds, model, model_label, eval_lengths
    ):
        """Trainable for a distance field model.

        Notable is the eval_len parameter (not present in Trainable), which
        is number of bins to consider when claculating accuracy. This is
        needed as the you typically want to guess the number of spikes in a
        region by using a distance field that is bigger than and contains the
        region. So there is not a 1-1 between distance field length and the
        number of bins over which we are counting spikes.
        """
        super().__init__(
            train_ds, val_ds, test_ds, model, model_label, eval_lengths
        )
        self.loss_fn = retinapy.models.dist_loss

    def forward(self, sample):
        masked_snippet = sample["snippet"].float().cuda()
        dist = sample["dist"].float().cuda()
        masked_snippet = masked_snippet.float().cuda()
        model_output = self.model(masked_snippet)
        # Dist model
        y = self.distfield_to_nn_output(dist)
        loss = self.loss_fn(model_output, target=y)
        return model_output, loss


def create_multi_cluster_df_datasets(
    recordings: Iterable[mea.CompressedSpikeRecording],
    input_len: int,
    output_len: int,
    downsample: int,
):
    train_ds = []
    val_ds = []
    test_ds = []
    stride = 17
    for rec in recordings:
        dc_rec = mea.decompress_recording(rec, downsample=downsample)
        train_val_test_splits = mea.mirror_split(
            dc_rec, split_ratio=SPLIT_RATIO
        )
        snippet_len = input_len + output_len
        train_val_test_datasets = [
            retinapy.dataset.DistFieldDataset(
                r,
                snippet_len=snippet_len,
                mask_begin=input_len,
                mask_end=snippet_len,
                pad=round(ms_to_num_bins(LOSS_CALC_PAD_MS, downsample)),
                dist_clamp=round(ms_to_num_bins(DIST_CLAMP_MS, downsample)),
                stride=stride,
                enable_augmentation=use_augmentation,
                allow_cheating=False,
            )
            for (r, use_augmentation) in zip(
                # train_val_test_splits, [True, False, False]
                # For the moment, while the validation just takes a small
                # portion of the validation set.
                train_val_test_splits,
                [True, True, False],
            )
        ]
        train_ds.append(train_val_test_datasets[0])
        val_ds.append(train_val_test_datasets[1])
        test_ds.append(train_val_test_datasets[2])
    concat_train_ds = retinapy.dataset.ConcatDistFieldDataset(train_ds)
    concat_val_ds = retinapy.dataset.ConcatDistFieldDataset(val_ds)
    concat_test_ds = retinapy.dataset.ConcatDistFieldDataset(test_ds)
    res = (concat_train_ds, concat_val_ds, concat_test_ds)
    return res


def create_distfield_datasets(
    recording: mea.CompressedSpikeRecording,
    input_len: int,
    output_len: int,
    downsample: int,
):
    rec = mea.decompress_recording(recording, downsample=downsample)
    train_val_test_splits = mea.mirror_split(rec, split_ratio=SPLIT_RATIO)
    snippet_len = input_len + output_len
    train_val_test_datasets = [
        retinapy.dataset.DistFieldDataset(
            r,
            snippet_len=snippet_len,
            mask_begin=input_len,
            mask_end=snippet_len,
            pad=round(ms_to_num_bins(LOSS_CALC_PAD_MS, downsample)),
            dist_clamp=round(ms_to_num_bins(DIST_CLAMP_MS, downsample)),
            enable_augmentation=use_augmentation,
            allow_cheating=False,
        )
        for (r, use_augmentation) in zip(
            train_val_test_splits, [True, False, False]
        )
    ]
    return train_val_test_datasets


def create_count_datasets(
    recording: mea.CompressedSpikeRecording,
    input_len: int,
    output_len: int,
    downsample: int,
):
    """
    Creates the spike count datasets for the given recording data.

    Three datasets are returned: train, validation and test.

    These datasets take the form:
        X,y = (stimulus-spike history, num spikes)

    The length of the input history, the output binning duration and the
    downsample rate can be configured.
    """
    rec = mea.decompress_recording(recording, downsample=downsample)
    train_val_test_splits = mea.mirror_split(rec, split_ratio=SPLIT_RATIO)
    # train_val_test_splits = mea.split(rec, split_ratio=SPLIT_RATIO)
    train_val_test_datasets = [
        retinapy.dataset.SpikeCountDataset(
            r,
            input_len=input_len,
            output_len=output_len,
        )
        for r in train_val_test_splits
    ]
    return train_val_test_datasets


class TrainableGroup:
    def trainable_label(self, config):
        raise NotImplementedError

    def create_trainable(
        self, recordings: Iterable[mea.CompressedSpikeRecording], config
    ):
        raise NotImplementedError


class MultiClusterDistFieldTGroup(TrainableGroup):
    @staticmethod
    def trainable_label(config):
        return (
            f"MultiClusterDistField-{config.downsample}"
            f"ds_{config.input_len}in_{config.output_len}out"
        )

    @classmethod
    def create_trainable(cls, recordings, config):
        train_ds, val_ds, test_ds = create_multi_cluster_df_datasets(
            recordings,
            config.input_len,
            config.output_len,
            config.downsample,
        )
        max_num_clusters = max([len(r.cluster_ids) for r in recordings])
        model = retinapy.models.MultiClusterModel(
            config.input_len + config.output_len,
            config.output_len,
            cls.num_downsample_layers(config.input_len, config.output_len),
            len(recordings),
            max_num_clusters,
        )
        res = DistFieldTrainableMC(
            train_ds,
            val_ds,
            test_ds,
            model,
            DistFieldCnnTGroup.trainable_label(config),
            eval_lengths=[10, 20, 50],
        )
        return res

    @staticmethod
    def num_downsample_layers(in_len, out_len):
        res_f = math.log(in_len / out_len, 2)
        num_downsample = int(res_f)
        if res_f - num_downsample > 0.4:
            logging.warning(
                "Model input/output lengths are not well matched. "
                f"Downsample desired: ({res_f}), downsample being used: "
                f"({num_downsample})."
            )
        return num_downsample


class DistFieldCnnTGroup(TrainableGroup):
    @staticmethod
    def trainable_label(config):
        return f"DistFieldCnn-{config.downsample}" f"ds_{config.input_len}in"

    @staticmethod
    def create_trainable(recordings, config):
        rec = recordings[0]
        output_lens = {1984: 200, 992: 100, 3174: 400, 1586: 200}
        model_out_len = output_lens[config.input_len]
        train_ds, val_ds, test_ds = create_distfield_datasets(
            rec, config.input_len, model_out_len, config.downsample
        )
        model = retinapy.models.DistanceFieldCnnModel(
            DIST_CLAMP_MS,
            config.input_len + model_out_len,
            model_out_len,
        )
        res = DistFieldTrainable(
            train_ds,
            val_ds,
            test_ds,
            model,
            DistFieldCnnTGroup.trainable_label(config),
            eval_lengths=[5, 10, 20, 50, 100],
        )
        return res


class LinearNonLinearTGroup(TrainableGroup):
    @staticmethod
    def trainable_label(config):
        return (
            f"LinearNonLinear-{config.downsample}"
            f"ds_{config.input_len}in_{config.output_len}out"
        )

    @staticmethod
    def create_trainable(recordings, config):
        rec = recordings[0]
        num_inputs = IN_CHANNELS * config.input_len
        m = retinapy.models.LinearNonlinear(in_n=num_inputs, out_n=1)
        train_ds, val_ds, test_ds = create_count_datasets(
            rec, config.input_len, config.output_len, config.downsample
        )
        label = LinearNonLinearTGroup.trainable_label(config)
        return LinearNonLinearTrainable(train_ds, val_ds, test_ds, m, label)


def _train(out_dir):
    print("Models & Configurations")
    print("=======================")
    # Product of models and configs
    trainable_groups = {
        "LinearNonLinear": LinearNonLinearTGroup,
        "DistFieldCnn": DistFieldCnnTGroup,
        "MultiClusterDistField": MultiClusterDistFieldTGroup,
    }

    def run_id(model_str, config):
        return f"{model_str}-{str(config)}"

    def _match(run_id, match_str):
        return match_str in run_id

    do_trainable = dict()
    for c in all_configs:
        for _, tg in trainable_groups.items():
            t_label = tg.trainable_label(c)
            do_trainable[t_label] = _match(t_label, opt.k)

    logging.info(f"Model-configs filter: {opt.k}")
    logging.info(
        "\n".join(
            [
                t_label if do_train else t_label.ljust(40) + " (skip)"
                for t_label, do_train in do_trainable.items()
            ]
        )
    )
    total_trainables = sum(do_trainable.values())
    logging.info(f"Total: {total_trainables} models to be trained.")

    # Load the data.
    recordings = mea.load_3brain_recordings(
        opt.stimulus_pattern,
        opt.stimulus,
        opt.response,
    )
    ## Filter the recording with different sample rate
    skip_rec_names = {"Chicken_21_08_21_Phase_00"}
    recordings = [r for r in recordings if r.name not in skip_rec_names]
    
    ## Filter recordings, if requested.
    # if opt.recording_names is not None:
    #    recordings = [
    #        r for r in recordings if r.name in set(opt.recording_names)
    #    ]
    # logging.info(f"Loaded {len(recordings)} recordings, compressed:")
    # for r in recordings:
    #    logging.info(f"  {r.name}")
    # recordings = [
    #    mea.single_3brain_recording(
    #        opt.recording_names[0],
    #        mea.load_stimulus_pattern(opt.stimulus_pattern),
    #        mea.load_recorded_stimulus(opt.stimulus),
    #        mea.load_response(opt.response),
    #        # include_clusters={21, 138},
    #        # include_clusters={21, },
    #    )
    # ]

    done_trainables = set()
    for c in all_configs:
        for tg in trainable_groups.values():
            t_label = tg.trainable_label(c)
            if t_label in done_trainables:
                continue
            if not do_trainable[t_label]:
                continue
            t = tg.create_trainable(recordings, c)
            if t is None:
                logging.warning(
                    f"Skipping. Model ({t_label}) isn't yet supported."
                )
                continue
            num_done = len(done_trainables)
            logging.info(
                f"Starting model training ({num_done}/{total_trainables}): "
                f"{t_label}"
            )
            sub_dir = out_dir / str(t_label)
            logging.info(f"Output directory: ({sub_dir})")
            retinapy.train.train(
                t,
                num_epochs=opt.epochs,
                batch_size=opt.batch_size,
                lr=opt.lr,
                weight_decay=opt.weight_decay,
                out_dir=sub_dir,
                steps_til_log=opt.steps_til_log,
                steps_til_eval=opt.steps_til_eval,
                initial_checkpoint=opt.initial_checkpoint,
            )
            logging.info(f"Finished training model")
            done_trainables.add(t_label)
    logging.info("Finished training all linear non-linear models.")


def main():
    retinapy._logging.setup_logging(logging.INFO)
    # Arguments are parsed now. This is not done globally in the file scope as
    # tests are run against methods in this file, and we don't want to run
    # argument parsing when running tests.
    global opt
    opt, opt_text = parse_args()

    labels = opt.labels.split(",") if opt.labels else None
    base_dir = pathlib.Path(opt.output if opt.output else DEFAULT_OUT_BASE_DIR)
    out_dir = retinapy._logging.get_outdir(base_dir, labels)
    print("Output directory:", out_dir)
    retinapy._logging.enable_file_logging(out_dir / LOG_FILENAME)
    # Record the arguments.
    with open(str(out_dir / ARGS_FILENAME), "w") as f:
        f.write(opt_text)
    _train(out_dir)


if __name__ == "__main__":
    main()
