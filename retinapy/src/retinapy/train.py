from contextlib import contextmanager
import logging
import pathlib
import time
import datetime
from typing import Optional, Union

import retinapy._logging
import retinapy.models
import torch


_logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        train_ds: torch.utils.data.Dataset,
        val_ds: torch.utils.data.Dataset,
        test_ds: torch.utils.data.Dataset,
        model: torch.nn.Module,
        label: str,
    ):
        """
        Args:
            train_ds: the dataset to train the model on.
            val_ds: the dataset to evaluate the model on and guide model
                training. This dataset is used to decide what model states to
                keep, and when to stop training, if early termination is
                enabled.
            test_ds: the test dataset. Similar to the validation dataset, this
                dataset is available for evaluating the model; however,
                its purpose is to be a datasat which has no influence
                on guiding the training. This includes any hyperparameter
                turing and the design of inference procedures. If more stages
                of data holdout are desired, then the validation dataset
                should be split again, rather than using the test dataset.
            model: the PyTorch model to train.
            label: a string label for this trainable.
        """
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.model = model
        self.label = label

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

    def __str__(self) -> str:
        return f"Trainable ({self.label})"


def _create_dataloaders(train_ds, val_ds, test_ds, batch_size):
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=20,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        # For debugging, it's nice to see a variety:
        shuffle=True,
        drop_last=True,
        num_workers=20,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=20,
    )
    return train_dl, val_dl, test_dl


@contextmanager
def evaluating(model):
    """
    Context manager to set the model to eval mode and then back to train mode.

    Used this to prevent an exception leading to unexpected training state.
    """
    original_mode = model.training
    model.eval()
    try:
        model.eval()
        yield
    finally:
        # Switch back to the original training mode.
        model.train(original_mode)


class TrainingTimer:
    """A little timer to track training.

    All times are in seconds (fractional).
    """
    def __init__(self):
        self.epoch_w = 0.1
        self.batch_w = 0.1
        self.training_start = None
        self.epoch_start = None
        self.batch_start = None
        self._rolling_epoch_duration = None
        self._rolling_batch_duration = None

    @property
    def rolling_epoch_duration(self) -> float:
        if self._rolling_epoch_duration is None:
            raise ValueError("Not yet started")
        if self._rolling_epoch_duration is None:
            return self.epoch_elapsed()
        else:
            return self._rolling_epoch_duration

    @property
    def rolling_batch_duration(self) -> float:
        if self.batch_start is None:
            raise ValueError("Not yet started")
        if self._rolling_batch_duration is None:
            return self.batch_elapsed()
        else:
            return self._rolling_batch_duration

    def start_training(self):
        self.training_start = time.monotonic()

    def start_epoch(self):
        next_epoch_start = time.monotonic()
        # First call?
        if self.epoch_start is None:
            self.epoch_start = next_epoch_start
            return
        assert self.epoch_start is not None
        epoch_elapsed = next_epoch_start - self.epoch_start
        self.epoch_start = next_epoch_start
        # Second call? No existing duration to average.
        if self._rolling_epoch_duration is None:
            self._rolling_epoch_duration = epoch_elapsed
            return
        assert self._rolling_epoch_duration is not None
        self._rolling_epoch_duration = (
                self.epoch_w * epoch_elapsed
                + (1 - self.epoch_w) * self._rolling_epoch_duration
        )

    def start_batch(self):
        next_batch_start = time.monotonic()
        # First call?
        if self.batch_start is None:
            self.batch_start = next_batch_start
            return
        assert self.batch_start is not None
        batch_elapsed = next_batch_start - self.batch_start
        self.batch_start = next_batch_start
        # Second call? No existing duration to average.
        if self._rolling_batch_duration is None:
            self._rolling_batch_duration = batch_elapsed
            return 
        # Third and subsequent call.
        assert self._rolling_batch_duration is not None
        self._rolling_batch_duration = (
                self.batch_w * batch_elapsed
                + (1 - self.batch_w) * self._rolling_batch_duration
        )

    def training_elapsed(self) -> float:
        if self.training_start is None:
            raise ValueError("Training has not started")
        return time.monotonic() - self.training_start

    def epoch_elapsed(self) -> float:
        if self.epoch_start is None:
            raise ValueError("Epoch has not started")
        return time.monotonic() - self.epoch_start

    def batch_elapsed(self) -> float:
        if self.batch_start is None:
            raise ValueError("Batch has not started")
        return time.monotonic() - self.batch_start



def train(
    trainable: Trainable,
    num_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    out_dir: Union[str, pathlib.Path],
    steps_til_log: int = 1000,
    steps_til_eval: Optional[int] = None,
    evals_til_eval_test_ds: Optional[int] = None,
    initial_checkpoint: Optional[Union[str, pathlib.Path]] = None,
):
    """
    Train a model.

    This is a training loop that works with any Trainable object.

    It encapsulates basic functionality like logging, checkpointing and
    choosing when to run an evalutaion. Users might be just as well off 
    by copying the code to use as a baseline and modifying it to their needs.
    """
    logging.info(f"Training {trainable.label}")

    # Setup output (logging & checkpoints).
    tensorboard_dir = pathlib.Path(out_dir) / "tensorboard"
    tb_logger = retinapy._logging.TbLogger(tensorboard_dir)

    # Load the model & loss fn.
    model = trainable.model
    if initial_checkpoint is not None:
        retinapy.models.load_model(model, initial_checkpoint)

    # Load the data.
    train_dl, val_dl, test_dl = _create_dataloaders(
        trainable.train_ds,
        trainable.val_ds,
        trainable.test_ds,
        batch_size=batch_size,
    )
    _logger.info(
        f"Dataset sizes: train ({len(train_dl.dataset)}), "
        f"val ({len(val_dl.dataset)}), test ({len(test_dl.dataset)})."
    )
    model.train()
    model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    def _eval():
        nonlocal num_evals
        if num_evals == evals_til_eval_test_ds:
            dl = test_dl
            label = "test-ds"
        else:
            dl = val_dl
            label = "val-ds"
        _logger.info(f"Running evaluation {label}")
        with evaluating(trainable.model), torch.no_grad():
            metrics = trainable.evaluate(dl)
            tb_logger.log(step, metrics, label)
            retinapy._logging.print_metrics(metrics)
        num_evals += 1
        return metrics

    model_saver = retinapy._logging.ModelSaver(out_dir, model, optimizer)
    timer = TrainingTimer()
    timer.start_training()
    step = 0
    num_evals = 0
    for epoch in range(num_epochs):
        timer.start_epoch()
        loss_meter = retinapy._logging.Meter("loss")
        for batch_step, sample in enumerate(train_dl):
            timer.start_batch()
            optimizer.zero_grad()
            model_out, total_loss = trainable.forward(sample)
            total_loss.backward()
            optimizer.step()
            batch_size = len(sample[0])
            loss_meter.update(total_loss.item(), batch_size)
            metrics = [
                retinapy._logging.Metric(
                    "loss", total_loss.item() / batch_size
                ),
            ]
            tb_logger.log(step, metrics, log_group="train")

            # Log to console.
            if step % steps_til_log == 0:
                model_mean = torch.mean(model_out)
                model_sd = torch.std(model_out)
                elapsed_min, elapsed_sec = divmod(
                        round(timer.training_elapsed()), 60)
                _logger.info(
                    f"epoch: {epoch}/{num_epochs} | "
                    f"step: {batch_step:>4}/{len(train_dl)} "
                    f"({batch_step/len(train_dl):>3.0%}) | "
                    f"elapsed: {elapsed_min:>2}m:{elapsed_sec:02d}s | "
                    f"({round(1/timer.rolling_batch_duration):>2} batches/s) | "
                    f"loss: {loss_meter.avg:.5f} | "
                    f"out mean (sd) : {model_mean:>4.3f} ({model_sd:>4.3f})"
                )
                loss_meter.reset()

            # Evaluate.
            # (step + 1), as we don't want to evaluate on the first step.
            if steps_til_eval and (step + 1) % steps_til_eval == 0:
                _eval()
            step += 1

        # Evaluate and save at end of epoch.
        # Print seconds using format strings

        _logger.info(f"Finished epoch in {round(timer.epoch_elapsed())} secs "
                     f"(rolling duration: "
                     f"{round(timer.rolling_batch_duration)} s/batch)")
        metrics = _eval()
        model_saver.save_checkpoint(epoch, metrics)
