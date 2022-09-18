from os import stat
import torch
import retinapy
import retinapy.models
import retinapy.nn
import retinapy.dataset
import retinapy.mea as mea
import retinapy.spikedistancefield as sdf
import argparse
import pathlib
import logging
import retinapy._logging
import yaml
from typing import Union
import numpy as np
import scipy
from collections import defaultdict
from contextlib import contextmanager


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
    data_group.add_argument("--recording-name", type=str, default=None, help="Name of recording within the recording file.")
    data_group.add_argument("--cluster-id", type=int, default=None, help="Cluster ID to train on.")

    parser.add_argument("--steps-til-val", type=int, default=None, help="Steps until validation.")
    parser.add_argument("--log-interval", type=int, default=1000, help="How many batches to wait before logging a status update.")
    parser.add_argument("--initial-checkpoint", type=str, default=None, help="Initialize model from the checkpoint at this path.")
    #parser.add_argument("--resume", type=str, default=None, help="Resume full model and optimizer state from checkpoint path.")
    parser.add_argument("--output", type=str, default=None, metavar="DIR", help="Path to output folder (default: current dir).")
    parser.add_argument("--labels", type=str, default=None, help="List of experiment labels. Used for naming files and/or subfolders.")
    parser.add_argument("--epochs", type=int, default=8, metavar="N", help="number of epochs to train (default: 300)")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--val-with-train-ds-period", type=int, default=10, help="After how many validation runs with the validation data should validation be run with the training data.")
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


# TODO: checkpointing like: https://github.com/rwightman/pytorch-image-models/blob/7c4682dc08e3964bc6eb2479152c5cdde465a961/timm/utils/checkpoint_saver.py#L21


class Configuration:
    def __init__(self, downsample_factor, input_len, output_len):
        self.downsample_factor = downsample_factor
        self.input_len = input_len
        self.output_len = output_len

    def __str__(self):
        return f"{self.downsample_factor}ds_{self.input_len}in_{self.output_len}out"


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
    for downsample_factor in downsample_factors:
        for in_len in input_lengths_ms:
            in_bins = ms_to_num_bins(in_len, downsample_factor)
            for out_len in output_lenghts_ms:
                out_bins = ms_to_num_bins(out_len, downsample_factor)
                if in_bins < 1 or out_bins < 1:
                    # Not enough resolution at this downsample factor.
                    continue
                in_bins_int = round(in_bins)
                out_bins_int = round(out_bins)
                res.append(
                    Configuration(downsample_factor, in_bins_int, out_bins_int)
                )
    return res


class Trainable:
    """Encapsulates a dataset, model input-output and loss function.

    This class is needed in order to be able to train multiple models and
    configurations with the same training function. The training function
    is too general to know about how to route the data into and out of a model,
    evaluate the model or how to take a model output and create a prediction.

    Redesign from function parameters to a class
    --------------------------------------------
    The class began as a dumb grouping of the parameters to the train function:
    train_ds, val_ds, test_ds, model, loss_fn, forward_fn, val_fn, and more—there
    were so many parameters that they were grouped together into a NamedTuple.
    However, functions like forward_fn and val_fn would need to be given
    the model, loss_fn and forward_fn in order to operate. This is the exact
    encapsulation behaviour that classes achieve, so the NamedTuple was made
    into a class. Leaning into the use of classes, forward_fn and val_fn were
    made into methods of the class, while the rest became properties. This
    change is noteworthy as customizing the forward or validation functions
    now requires defining a new class, rather than simply passing in a new
    function. Although, nothing is stopping someone from defining a class that
    simply wraps a function and passes the arguments through.

    Flexibility to vary the data format
    -----------------------------------
    Why things belongs inside or outside this class can be understood by
    realizing that the nature of the datasets are encapsulated here. As such,
    any function that needs to extract the individual parts of a dataset
    sample will need to know what is in each sample. Such a function is a
    good candidate to appear in this class.

    While in some cases you can separate the datasets from the models, this
    isn't always easy or a good idea. A model for ImageNet can easily be
    separated from the dataset, as the inputs and outputs are so standard; but
    for the spike prediction, the model output is quite variable. Consider
    the distance-field model which outputs a distance field, whereas a
    Poisson-distribution model will output a single number. Training is
    done with these outputs, and involves the dataset producing sample tuples
    that have appropriate elements (distance fields, for example).
    The actual inference is an additional calculation using these outputs.

    So, the procedure of taking the model input and model output from a dataset
    sample and feeding it to the model calculating the loss and doing the
    inference—none of these steps can be abstracted to be ignorant of either
    the model or the dataset.

    Other libraries
    ---------------
    Compared to Keras and FastAI: Trainable encapsulates a lot less than
    Keras's Model or FastAI's Learner.

    At this point (2022-09-12) I'm not eager to use the FastAI API, as I don't
    want to discover later that it's too limiting in some certain way. It's
    quite possible that it's already too prescriptive. Reading the docs, it's
    not clear what parts of Learner's internals are exposed for customization.
    If all "_" prefixed methods are not meant to be customized, then it's
    already too restrictive. Notably, there seems to be an expected format for
    the elements of the dataset, which I want to avoid. The reason for this is
    that the distance fields are intermediate results, and while I want to
    train on them, I would like to evaluate based on approximations to
    actual spike count accuracy, and I would like to make predictions using
    much more expensive dynamic programming inference routines. So the data
    doesn't fall nicely into (X,y) type data, and the metrics are not
    consistent across training and evaluation.

    In addition, at least at the momemt, FastAI's library provides a lot more
    abstraction/generalization than I need, which can make it harder for
    myself (or others) to understand what is going on. This might end up being
    a mistake, as the growing code might reveal itself to provide nice
    abstraction boundaries that are already handled nicely in FastAI.
    """

    def __init__(self, train_ds, val_ds, test_ds, model, model_label):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.model = model
        self.model_label = model_label

    def forward(self, sample):
        """Run the model forward.

        Args:
            sample: a single draw from the train or validation data loader.

        Returns:
            (output, loss): the model output and the loss, as a tuple.
        """
        raise NotImplementedError("Override")

    def evaluate(self, val_dl):
        """Run the full evaluation procedure.

        Args:
            val_dl: the validation data loader.

        Returns:
            metrics: a str:float dictionary containing evaluation metrics. It
                is expected that this dictionary at least contains 'loss' and
                'accuracy' metrics.
        """
        raise NotImplementedError("Override")


class LinearNonLinearTrainable(Trainable):
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


class DistFieldTrainable(Trainable):
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
        super(DistFieldTrainable, self).__init__(
            train_ds, val_ds, test_ds, model, model_label
        )
        self.loss_fn = retinapy.models.DistLoss()
        self.eval_lengths = eval_lengths
        self.min_dist = 0.5
        self.dist_norm = 20
        self.offset = -0.5
        # Network output should ideally have mean,sd = (0, 1). Network output
        # 20*exp([-3, 3])  = [1.0, 402], which is a pretty good range, with
        # 20 being the mid point. Is this too low?

    def forward(self, sample):
        masked_snippet, _, dist = sample
        masked_snippet = masked_snippet.float().cuda()
        model_output = self.model(masked_snippet)
        # Dist model
        dist = dist.float().cuda()
        y = self.distfield_to_nn_output(dist)
        loss = self.loss_fn(model_output, target=y)
        return model_output, loss

    def quick_infer(self, dist, eval_len):
        """Quickly infer the number of spikes in the eval region.

        An approximate inference used for evaluation.

        Returns:
            the number of spikes in the region.
        """
        threshold = 30
        res = (dist[:, 0:eval_len] < threshold).sum(dim=1)
        return res

    def distfield_to_nn_output(self, distfield):
        return torch.log((distfield + self.min_dist) / self.dist_norm) - self.offset

    def nn_output_to_distfield(self, nn_output):
        return torch.exp(nn_output) * self.dist_norm - self.min_dist + self.offset


    def evaluate(self, val_dl):
        predictions = defaultdict(list)
        targets = defaultdict(list)
        loss_meter = retinapy._logging.Meter("loss")
        for (masked_snippet, target_spikes, dist) in val_dl:
            X = masked_snippet.float().cuda()
            dist = dist.float().cuda()
            model_output = self.model(X)
            target_dist = self.distfield_to_nn_output(dist)
            batch_len = X.shape[0]
            loss_meter.update(
                self.loss_fn(model_output, target=target_dist).item(),
                batch_len,
            )
            # Count accuracies
            # Unnormalize for accuracy.
            model_output = self.nn_output_to_distfield(model_output)
            for eval_len in self.eval_lengths:
                pred = self.quick_infer(model_output, eval_len=eval_len).cpu()
                y = torch.sum(target_spikes[:, 0:eval_len], dim=1)  # .float()
                predictions[eval_len].append(pred)
                targets[eval_len].append(y)

        metrics = [
            retinapy._logging.Metric("loss", loss_meter.avg, increasing=False)
        ]
        for eval_len in self.eval_lengths:
            p = torch.cat(predictions[eval_len])
            t = torch.cat(targets[eval_len])
            acc = (p == t).float().mean().item()
            pearson_corr = scipy.stats.pearsonr(p, t)[0]
            metrics.append(
                retinapy._logging.Metric(f"accuracy-{eval_len}_bins", acc)
            )
            metrics.append(
                retinapy._logging.Metric(
                    f"pearson_corr-{eval_len}_bins", pearson_corr
                )
            )
        return metrics


def create_distfield_datasets(
    recording: mea.CompressedSpikeRecording,
    input_len: int,
    output_len: int,
    downsample_factor: int,
):
    rec = mea.decompress_recording(recording, downsample=downsample_factor)
    train_val_test_splits = mea.mirror_split(rec, split_ratio=(6, 2, 2))
    snippet_len = input_len + output_len
    train_val_test_datasets = [
        retinapy.dataset.SpikeDistanceFieldDataset(
            r,
            snippet_len=snippet_len,
            mask_begin=input_len,
            mask_end=snippet_len,
            pad=PAD_FOR_LOSS_CALC,
            dist_clamp=DIST_CLAMP,
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
    downsample_factor: int,
):
    """
    Creates the spike count datasets for the given recording data.

    Three datasets are returned: train, validation and test.

    These datasets take the form:
        X,y = (stimulus-spike history, num spikes)

    The length of the input history, the output binning duration and the
    downsample rate can be configured.
    """
    rec = mea.decompress_recording(recording, downsample=downsample_factor)
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

    def create_trainable(self, rec, config):
        raise NotImplementedError


class DistFieldCnnTrainableGroup(TrainableGroup):
    @staticmethod
    def trainable_label(config):
        return (
            f"DistFieldCnn-{config.downsample_factor}"
            f"ds_{config.input_len}in"
        )

    @staticmethod
    def create_trainable(rec, config):
        #num_halves = {
        #    1984: 4,
        #    992: 3,
        #    3174: 5,
        #    1586: 4,
        #}
        num_halves = {
            1984: 4,
            992: 4,
            3174: 4,
            1586: 4,
        }
        output_lens = {
            1984: 200,
            992: 100,
            3174: 400,
            1586: 200
            }
        model_out_len = output_lens[config.input_len]
        if config.input_len not in num_halves:
            return None
        train_ds, val_ds, test_ds = create_distfield_datasets(
            rec, config.input_len, model_out_len, 
            config.downsample_factor
        )
        model = retinapy.models.DistanceFieldCnnModel(
            DIST_CLAMP,
            config.input_len + model_out_len,
            model_out_len,
            num_halves[config.input_len],
        )
        res = DistFieldTrainable(
            train_ds,
            val_ds,
            test_ds,
            model,
            DistFieldCnnTrainableGroup.trainable_label(config),
            eval_lengths=[1, 2, 5, 10, 20, 50],
        )
        return res


class LinearNonLinearTrainableGroup(TrainableGroup):
    @staticmethod
    def trainable_label(config):
        return (
            f"LinearNonLinear-{config.downsample_factor}"
            f"ds_{config.input_len}in_{config.output_len}out"
        )

    @staticmethod
    def create_trainable(rec, config):
        num_inputs = IN_CHANNELS * config.input_len
        m = retinapy.models.LinearNonlinear(in_n=num_inputs, out_n=1)
        train_ds, val_ds, test_ds = create_count_datasets(
            rec, config.input_len, config.output_len, config.downsample_factor
        )
        label = LinearNonLinearTrainableGroup.trainable_label(config)
        return LinearNonLinearTrainable(train_ds, val_ds, test_ds, m, label)


def checkpoint_path(model_name, config):
    filename = f"{model_name}_{config.input_len}i_{config.output_len}o_{config.downsample_factor}d.pt"
    # TODO
    # return pathlib.Path(model_name} / pathlib.Path(MODEL_CHECKPOINT_DIR) / filename
    return None


def test_single_model(model, test_dl):
    model.eval()
    model.cuda()
    spike_seq = []
    pred_seq = []
    with torch.no_grad():
        for (X, y) in test_dl:
            X = X.cuda()
            y = y.cuda()
            prediction = model(X)
            spike_seq.append(torch.flatten(y))
            pred_seq.append(torch.flatten(prediction))
    spike_seq = torch.cat(spike_seq).cpu()
    pred_seq = torch.cat(pred_seq).cpu()
    return spike_seq, pred_seq


def calc_metrics(spike_seq, pred_seq):
    num_correct = torch.sum(spike_seq == pred_seq)
    assert spike_seq.shape == pred_seq.shape
    assert len(spike_seq.shape) == 1
    num_bins = spike_seq.shape[0]
    accuracy = num_correct / num_bins
    pearson_corr = scipy.stats.pearsonr(pred_seq, spike_seq).statistic
    spearman_corr = scipy.stats.spearmanr(pred_seq, spike_seq).statistic
    return (spike_seq, pred_seq, accuracy, pearson_corr, spearman_corr)


def test_single_config(config):
    dataloader = None

    def _create_dataloader():
        assert dataloader == None, "This should only be called once."
        _, _, test_dataset = create_count_datasets(
            config.input_len,
            config.output_len,
            config.downsample_factor,
            config.cluster_idx,
        )
        return test_dataloader(test_dataset, opt.batch_size)

    res = {}
    for model_fn in model_fns:
        m = model_fn(config)
        if not m:
            _logger.warning(
                f"Skipping. Model ({m.name}) doesn't support "
                f"configuration({config}). "
            )
            continue
        _checkpoint_path = checkpoint_path(m.name, config)
        if not _checkpoint_path.exists():
            _logger.warning(
                f"Skipping. Model ({m.name}) doesn't have a checkpoint for "
                f"configuration({config})."
            )
            continue
        # Lazy load dataloader.
        if not dataloader:
            dataloader = _create_dataloader()
        retinapy.models.load_model(m, _checkpoint_path)
        spike_seq, pred_seq = test_single_model(m, dataloader)
        metrics = calc_metrics(spike_seq, pred_seq)
        res[m.name] = metrics
    return res


def test_all():
    logging.info("Testing all configurations.")
    logging.info(f"{len(all_configs)} configurations to test.")
    res = {}
    for c in all_configs:
        c_res = test_single_config(c)
        res[c] = c_res
    return res


def test_dataloader(test_ds, batch_size):
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=20,
    )
    return test_dl


def create_dataloaders(train_ds, val_ds, test_ds):
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=20,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=opt.batch_size,
        # For debugging, it's nice to see a variety:
        shuffle=True,
        drop_last=True,
        num_workers=20,
    )

    test_dl = test_dataloader(test_ds, opt.batch_size)
    return train_dl, val_dl, test_dl


def _train(out_dir):
    rec = mea.single_3brain_recording(
        opt.recording_name,
        mea.load_stimulus_pattern(opt.stimulus_pattern),
        mea.load_recorded_stimulus(opt.stimulus),
        mea.load_response(opt.response),
        include_clusters={opt.cluster_id},
    )
    print("Models & Configurations")
    print("=======================")
    # Product of models and configs
    trainable_groups = {
        "LinearNonLinear": LinearNonLinearTrainableGroup,
        "DistFieldCnn": DistFieldCnnTrainableGroup,
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
    # filtered_configs = all_configs[14:]
    done_trainables = set()
    for c in all_configs:
        for tg in trainable_groups.values():
            t_label = tg.trainable_label(c)
            if t_label in done_trainables:
                continue
            if not do_trainable[t_label]:
                continue
            t = tg.create_trainable(rec, c)
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
            train(t, sub_dir)
            logging.info(f"Finished training model")
            done_trainables.add(t_label)
    logging.info("Finished training all linear non-linear models.")


@contextmanager
def evaluating(model):
    """
    Context manager to set the model to eval mode and then back to train mode.

    This is used just in case there is an caught exception that leads to
    unexpected training state.
    """
    original_mode = model.training
    model.eval()
    try:
        model.eval()
        yield
    finally:
        # Switch back to the original training mode.
        model.train(original_mode)


def train(trainable, out_dir):
    logging.info(f"Training {trainable.model_label}")

    # Setup output (logging & checkpoints).
    tensorboard_dir = out_dir / "tensorboard"
    tb_logger = retinapy._logging.TbLogger(tensorboard_dir)

    # Load the model & loss fn.
    model = trainable.model
    if opt.initial_checkpoint is not None:
        retinapy.models.load_model(model, opt.initial_checkpoint)

    # Load the data.
    train_dl, val_dl, test_dl = create_dataloaders(
        trainable.train_ds, trainable.val_ds, trainable.test_ds
    )
    _logger.info(
        f"Dataset sizes: train ({len(train_dl.dataset)}), "
        f"val ({len(val_dl.dataset)}), test ({len(test_dl.dataset)})."
    )
    model.train()
    model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
    )

    model_saver = retinapy._logging.ModelSaver(out_dir, model, optimizer)
    num_epochs = opt.epochs
    step = 0

    for epoch in range(num_epochs):
        loss_meter = retinapy._logging.Meter("loss")
        for sample in train_dl:
            optimizer.zero_grad()
            model_out, total_loss = trainable.forward(sample)
            total_loss.backward()
            optimizer.step()
            batch_size = len(sample[0])
            loss_meter.update(total_loss.item(), batch_size)
            metrics = [
                retinapy._logging.Metric("loss", total_loss.item()/batch_size),
            ]
            tb_logger.log(step, metrics, log_group="train")

            if step % opt.log_interval == 0:
                model_mean = torch.mean(model_out)
                model_sd = torch.std(model_out)
                _logger.info(
                    f"epoch: {epoch}/{num_epochs} | "
                    f"step: {step}/{len(train_dl)*num_epochs} | "
                    f"loss: {loss_meter.avg:.5f} | "
                    f"out mean (sd) : {model_mean:.5f} ({model_sd:.5f})"
                )
                loss_meter.reset()

            if opt.steps_til_val and step % opt.steps_til_val == 0:
                _logger.info("Running evaluation (val ds)")
                with evaluating(model), torch.no_grad():
                    metrics = trainable.evaluate(val_dl)
                    tb_logger.log(step, metrics, "val-ds")
                    retinapy._logging.print_metrics(metrics)

            test_ds_eval_enabled = (
                opt.steps_til_val and opt.val_with_train_ds_period
            )
            if (
                test_ds_eval_enabled
                and (step + 1)
                % (opt.steps_til_val * opt.val_with_train_ds_period)
                == 0
            ):
                _logger.info("Running evaluation (train ds)")
                with evaluating(model), torch.no_grad():
                    metrics = trainable.evaluate(val_dl)
                    tb_logger.log(step, metrics, "train-ds")
                    retinapy._logging.print_metrics(metrics)
            step += 1
        # Evaluate and save at end of epoch.
        _logger.info("Running epoch evaluation (val ds)")
        with evaluating(model), torch.no_grad():
            metrics = trainable.evaluate(val_dl)
            tb_logger.log(step, metrics, "val-ds")
            retinapy._logging.print_metrics(metrics)
        model_saver.save_checkpoint(epoch, metrics)


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
