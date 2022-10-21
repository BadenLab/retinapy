from contextlib import contextmanager
import logging
import pathlib
from typing import Optional, Union

import retinapy._logging
import retinapy.models
import torch


_logger = logging.getLogger(__name__)

"""
Max training time before a new recovery is made.

Standard checkpointing only happens every epoch. But a checkpoint for recovery
purposes will be made every 30 minutes. Only a single recovery is kept.
"""
RECOVERY_CKPT_PERIOD_SEC = 30 * 60

"""
Here we take part in the rite of of passage for a deep learning project by
yet again reinventing the training loop architecture. No one wants their 
project stitched together with the callbacks of some soon to be abandonded or 
rewritten DL framework.
"""


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
        # TODO: manage GPU usage.
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

    def in_device(self):
        """Returns the device on which the model expects input to be located.

        Most likely, the whole model is on a single device, and it is
        sufficient to use `next(self.model.parameters()).device`.
        """
        raise NotImplementedError("Override")

    def __str__(self) -> str:
        return f"Trainable ({self.label})"

    def model_summary(self, sample) -> str:
        """Returns a detailed description of the model.

        Args:
            sample: a training sample given to allow the model structure to
            be inferred.

        At the moment, this is called by train() with the intent of saving
        out a file containing info like torchinfo.summary. The torch module
        input shape isn't known by train(), so the actual summary creation
        must be done somewhere like Trainable.

        Override this to add more features.
        """
        return f"Trainable ({self.label})"


def _create_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers):
    # Setting pin_memory=True. This is generally recommended when training on
    # Nvidia GPUs. See:
    #   - https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    #   - https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        # For debugging, it's nice to see a variety:
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
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


class TrainingTimers:
    """Collect timers here for convenience."""

    def __init__(self):
        self.batch = retinapy._logging.Timer()
        self.epoch = retinapy._logging.Timer()
        self.validation = retinapy._logging.Timer()
        self.recovery = retinapy._logging.Timer()

    @staticmethod
    def create_and_start():
        timer = TrainingTimers()
        timer.batch.restart()
        timer.epoch.restart()
        timer.recovery.restart()
        return timer


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
    num_workers: int = 20,
):
    """
    Train a model.

    This is a training loop that works with any Trainable object.

    It encapsulates basic functionality like logging, checkpointing and
    choosing when to run an evalutaion. Users might be just as well off
    by copying the code to use as a baseline and modifying it to their needs.
    """
    logging.info(f"Training {trainable.label}")

    # Enable Cuda convolution auto-tuning. For more details, see:
    #   https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    torch.backends.cudnn.benchmark = True

    # Create a GradScalar. For more details, see:
    #   https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
    grad_scaler = torch.cuda.amp.GradScaler()

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
        num_workers=num_workers
    )
    _logger.info(
        f"Dataset sizes: train ({len(train_dl.dataset)}), "
        f"val ({len(val_dl.dataset)}), test ({len(test_dl.dataset)})."
    )
    model.train()
    model.cuda()
    # Before going any further, log model structure.
    out_file = pathlib.Path(out_dir) / "model_summary.txt"
    with open(out_file, "w") as f:
        # This is allowed to fail, as if the model has issues, we want to
        # get the errors from the actual training forward pass.
        summary = None
        try:
            summary = trainable.model_summary(next(iter(train_dl)))
        except Exception as e:
            msg = (
                "Failed to generating model summary. Exception raised:\n"
                f"{str(e)}"
            )
            _logger.error(msg)
            summary = msg
        f.write(str(summary))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    model_saver = retinapy._logging.ModelSaver(out_dir, model, optimizer)
    metric_tracker = retinapy._logging.MetricTracker(out_dir)

    def _eval():
        nonlocal num_evals
        if num_evals == evals_til_eval_test_ds:
            dl = test_dl
            label = "test-ds"
        else:
            dl = val_dl
            label = "val-ds"
        _logger.info(f"Running evaluation {label}")
        with evaluating(trainable.model), torch.no_grad(), timers.validation:
            eval_results = trainable.evaluate(dl)
            tb_logger.log(step, eval_results, label)
            if "metrics" not in eval_results:
                raise ValueError("Trainable.evaluate() must return metrics.")
            metrics = eval_results["metrics"]
            retinapy._logging.print_metrics(metrics)
        tb_logger.log_scalar(
            step, "eval-time", timers.validation.elapsed(), label
        )
        num_evals += 1
        _logger.info(
            f"Finished evaluation in {round(timers.validation.elapsed())} sec "
            f"(rolling ave: {round(timers.validation.rolling_duration())} sec)"
        )
        num_evals += 1
        return metrics

    _logger.info("Starting training loop.")
    step = 0
    num_evals = 0
    timers = TrainingTimers.create_and_start()
    for epoch in range(num_epochs):
        timers.epoch.restart()
        loss_meter = retinapy._logging.Meter("loss")
        for batch_step, sample in enumerate(train_dl):
            timers.batch.restart()
            # set_to_none=True is suggested to improve performance, according to:
            #   https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            # A recipe for autocast, grad scaling and set_to_none:
            #   https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                model_out, total_loss = trainable.forward(sample)
            grad_scaler.scale(total_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            # Loss is expected to be returned already batch averaged.
            loss_meter.update(total_loss.item())
            metrics = [
                retinapy._logging.Metric("loss", total_loss.item()),
            ]
            tb_logger.log_metrics(step, metrics, log_group="train")

            # Log to console.
            if step % steps_til_log == 0:
                model_mean = torch.mean(model_out)
                model_sd = torch.std(model_out)
                elapsed_min, elapsed_sec = divmod(
                    round(timers.epoch.elapsed()), 60
                )
                # Floor total minutes to align with batch minutes.
                total_hrs, total_min = divmod(
                    int(timers.epoch.total_elapsed() / 60), 60
                )
                _logger.info(
                    f"epoch: {epoch}/{num_epochs} | "
                    f"step: {batch_step:>4}/{len(train_dl)} "
                    f"({batch_step/len(train_dl):>3.0%}) "
                    f"{round(1/timers.batch.rolling_duration()):>2}/s | "
                    f"elapsed: {elapsed_min:>1}m:{elapsed_sec:02d}s "
                    f"({total_hrs:>1}h:{total_min:02d}m) | "
                    f"loss: {loss_meter.avg:.3f} | "
                    f"out μ (σ): {model_mean:>3.2f} ({model_sd:>3.2f})"
                )
                loss_meter.reset()

            # Evaluate.
            # (step + 1), as we don't want to evaluate on the first step.
            if steps_til_eval and (batch_step + 1) % steps_til_eval == 0:
                is_near_epoch_end = batch_step + steps_til_eval >= len(train_dl)
                if not is_near_epoch_end:
                    _eval()
            step += 1

            # Recovery.
            # Don't allow training to proceed too long without checkpointing.
            if timers.recovery.elapsed() > RECOVERY_CKPT_PERIOD_SEC:
                model_saver.save_recovery()
                timers.recovery.restart()

        _logger.info(
            f"Finished epoch in {round(timers.epoch.elapsed())} secs "
            f"(rolling duration: "
            f"{round(timers.epoch.rolling_duration())} s/epoch)"
        )
        # Evaluate and save at end of epoch.
        metrics = _eval()
        # If this on_metric_end type of behaviour grows, consider switching
        # to callbacks.
        improved_metrics = metric_tracker.on_epoch_end(metrics, epoch)
        model_saver.save_checkpoint(epoch, improved_metrics)
        timers.recovery.restart()
    _logger.info(
        f"Finished training. {round(timers.epoch.total_elapsed())} "
        "secs elsapsed."
    )
