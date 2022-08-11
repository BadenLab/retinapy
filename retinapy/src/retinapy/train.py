import torch
import retinapy
import retinapy.models
import retinapy.nn
import retinapy.dataset
import retinapy.mea as mea
import configargparse
import pathlib
import logging
import logging.handlers
import yaml
import spikedistancefield as sdf
from typing import Union
import numpy as np


DEFAULT_OUT_BASE_DIR = "./out/train"
LOG_FILENAME = "train.log"
ARGS_FILENAME = "args.yaml"
CKPT_FILENAME_FORMAT = "checkpoint-{epoch}_step_{step}.pth"

REC_NAME = "Chicken_17_08_21_Phase_00"
DOWNSAMPLE_FACTOR = 18
STIMULUS_PATTERN_PATH = "./data/ff_noise.h5"
RECORDED_STIMULUS_PATH = "./data/ff_recorded_noise.pickle"
RESPONSE_PATH = "./data/ff_spike_response.pickle"
IN_CHANNELS = 4 + 1
BATCH_SIZE = 128
X_LEN = 1200
MODEL_OUT_LEN = 400
VAL_LEN = 400
PAD_FOR_LOSS_CALC = 500  # Quite often there are lengths in the range 300.
# The pad acts as the maximum, so it's a good candidate for a norm factor.
SNIPPET_LEN = X_LEN + MODEL_OUT_LEN
# Example: setting normalization to 400 would cause 400 time steps to be fit
# into the [0,1] region.
DIST_CLAMP = 100
DIST_NORM = MODEL_OUT_LEN + DIST_CLAMP * 2
# We need to clamp the 
DIST_CLAMP_NORM = DIST_CLAMP / DIST_NORM

_logger = logging.getLogger(__name__)

p = configargparse.ArgumentParser(
    config_file_parser_class=configargparse.YAMLConfigFileParser
)
# fmt: off
p.add('-c', '--config', is_config_file=True, help='config file path')
p.add('--lr', type=float, default=1e-5, help='Learning rate.')
p.add('--steps_til_val', type=int, default=2000, help='Steps until validation.')
p.add('--steps_til_ckpt', type=int, default=3000, help='Steps until checkpoint.')
p.add('--log-interval', type=int, default=100, help='How many batches to wait before logging a status update.')
p.add('--initial-checkpoint', type=str, default=None, help='Initialize model from the checkpoint at this path.')
#p.add('--resume', type=str, default=None, help='Resume full model and optimizer state from checkpoint path.')
p.add('--output', type=str, default=None, help='Path to output folder (default: current dir).')
p.add('--labels', type=str, default=None, help='List of experiment labels. Used for naming files and/or subfolders.')
# fmt: on
opt = p.parse_known_args()[0]

# TODO: checkpointing like: https://github.com/rwightman/pytorch-image-models/blob/7c4682dc08e3964bc6eb2479152c5cdde465a961/timm/utils/checkpoint_saver.py#L21


def setup_logging(level):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    console_handler = logging.StreamHandler()
    root_logger.addHandler(console_handler)


def enable_file_logging(log_path):
    root_logger = logging.getLogger()
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=(2 ** (10 * 2) * 5), backupCount=3
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s"
        )
        root_logger.addHandler(file_handler)


def get_outdir(base_dir, labels=None):
    base_dir = pathlib.Path(base_dir)
    if labels is None:
        labels = []
    base_dir = base_dir.joinpath(*labels)
    if not base_dir.exists():
        base_dir.mkdir(parents=True)
    count = 0
    folder_path_f = lambda cout: base_dir / str(count)
    while folder_path_f(count).exists():
        count += 1
    is_probably_too_many = count > 1000
    if is_probably_too_many:
        _logger.warn(f"Reached {count} output sub-folders.")
    out_dir = folder_path_f(count)
    out_dir.mkdir()
    return out_dir


def create_datasets():
    stimulus_pattern = mea.load_stimulus_pattern(STIMULUS_PATTERN_PATH)
    recorded_stimulus = mea.load_recorded_stimulus(RECORDED_STIMULUS_PATH)
    response = mea.load_response(RESPONSE_PATH)
    rec = mea.single_3brain_recording(
        REC_NAME, stimulus_pattern, recorded_stimulus, response
    )
    rec = mea.decompress_recording(rec, downsample=DOWNSAMPLE_FACTOR)
    train_val_test_splits = mea.split(rec, split_ratio=(6, 2, 2))
    train_val_test_datasets = [
        retinapy.dataset.SpikeDistanceFieldDataset(
            r,
            cluster_idx=5, 
            snippet_len=SNIPPET_LEN,
            mask_begin=X_LEN,
            mask_end=X_LEN + MODEL_OUT_LEN,
            pad=PAD_FOR_LOSS_CALC,
            dist_clamp=DIST_CLAMP,
            dist_norm=DIST_NORM,
            enable_augmentation=True,
            allow_cheating=False
        )
        for (r, use_augmentation) in zip(train_val_test_splits, 
                                         [True, True, False])
    ]
    return train_val_test_datasets


def create_dataloaders(train_ds, val_ds, test_ds):
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=20,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=True, # For debugging, it's nice to see a range.
        drop_last=True,
        num_workers=20,
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
        num_workers=20,
    )
    return train_dl, val_dl, test_dl


def load_model(model, checkpoint_path: Union[str, pathlib.Path]):
    checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file/folder ({checkpoint_path}) not found."
        )
    if checkpoint_path.is_dir():
        checkpoint_path = list(checkpoint_path.glob("*.pth"))[-1]

    _logger.info(f"Loading model from {checkpoint_path}.")
    checkpoint_state = torch.load(checkpoint_path)
    model_state = checkpoint_state["model"]
    model.load_state_dict(model_state)


def save_model(model, path: pathlib.Path, optimizer=None):
    _logger.info(f"Saving model to {path}.")
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    state = {
        "model": model.state_dict(),
    }
    if optimizer:
        state.update({"optimizer": optimizer.state_dict()})
    torch.save(state, path)


def model_fn():
    """
    Create the model.
    """
    def model1_fn():
        return retinapy.models.DistanceFieldCnnModel(clamp_max=DIST_CLAMP_NORM)

    def model2_fn():
        return retinapy.nn.FcBlock(
            hidden_ch=1024 * 3,
            num_hidden_layers=4,
            in_features=SNIPPET_LEN * IN_CHANNELS,
            out_features=MODEL_OUT_LEN,
            outermost_linear=True,
        )

    model = model1_fn()
    return model


def train():
    # Setup output (logging & checkpoints).
    labels = opt.labels.split(",") if opt.labels else None
    base_dir = opt.output if opt.output else DEFAULT_OUT_BASE_DIR
    out_dir = get_outdir(base_dir, labels)
    enable_file_logging(out_dir / LOG_FILENAME)
    p.write_config_file(opt, [str(out_dir / ARGS_FILENAME)])

    model = model_fn()

    if opt.initial_checkpoint is not None:
        load_model(model, opt.initial_checkpoint)

    train_ds, val_ds, test_ds = create_datasets()
    train_dl, val_dl, test_dl = create_dataloaders(train_ds, val_ds, test_ds)
    _logger.info(
        f"Loaded datasets. Counts: train ({len(train_dl)}), "
        f"val ({len(val_dl)}), test ({len(test_dl)})."
    )

    model.train()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=opt.lr, 
                                 weight_decay=1e-5)
    dist_loss = torch.nn.L1Loss()

    num_epochs = 50
    step = 0
    for epoch in range(num_epochs):
        for (masked_snippet, _, dist) in train_dl:
            target = dist.float().cuda()
            masked_snippet = masked_snippet.float().cuda()

            optimizer.zero_grad()
            model_output = model(masked_snippet)
            total_loss = dist_loss(model_output, target)
            total_loss.backward()
            optimizer.step()

            if step % opt.log_interval == 0:
                _logger.info(
                    f"Epoch: {epoch}/{num_epochs} | "
                    f"Step: {step}/{len(train_dl)*num_epochs} |"
                    f"Loss: {total_loss.item():.5f}"
                )
            if step % opt.steps_til_val == 0:
                print("Running validation (val ds)")
                validate(model, val_dl, dist_loss)
                model.train()

            if (step + 1) % (opt.steps_til_val*6) == 0:
                print("Running validation (train ds)")
                validate(model, train_dl, dist_loss)
                model.train()

            if (step + 1) % opt.steps_til_ckpt == 0:
                path = out_dir / CKPT_FILENAME_FORMAT.format(
                    epoch=epoch, step=step
                )
                save_model(model, path, optimizer)
            step += 1
    print("Saving checkpoint")
    path = out_dir / CKPT_FILENAME_FORMAT.format(epoch=num_epochs, step=0)
    save_model(model, path, optimizer)


def validate(model, val_dl, loss_fn):
    model.eval()
    correct = 0
    thresholds = [6, 2, 20]
    correct_t2 = 0
    correct_t3 = 0
    naive_correct = 0
    best_case_correct = 0
    count = 0
    with torch.no_grad():
        loss_cumulative = 0
        for (masked_snippet, target_spikes, dist) in val_dl:
            batch_len = masked_snippet.shape[0]
            masked_snippet = masked_snippet.float().cuda()
            dist = dist.float().cuda()
            target_spikes = target_spikes.cuda()
            device = masked_snippet.device
            model_output = model(masked_snippet)
            # Loss
            loss = loss_fn(model_output, target=dist)
            loss_cumulative += loss

            # Accuracy
            # Unnormalize for accuracy.
            model_output *= DIST_NORM
            dist *= DIST_NORM
            target_interval = [0, VAL_LEN]
            target_spikes = target_spikes[:, 0: VAL_LEN]
            actual_counts = torch.count_nonzero(target_spikes, dim=1)
            infer_count = sdf.quick_inference_from_df2(
                model_output,
                target_interval=target_interval,
                threshold=thresholds[0],
            )
            infer_count2 = sdf.quick_inference_from_df2(
                model_output,
                target_interval=target_interval,
                threshold=thresholds[1],
            )
            infer_count3 = sdf.quick_inference_from_df2(
                model_output,
                target_interval=target_interval,
                threshold=thresholds[2],
            )

            # It's always 1, so commenting out.
            best_possible_infer = sdf.quick_inference_from_df(
                dist, target_interval=target_interval, threshold=thresholds[1]
            )
            best_case_correct += torch.sum(best_possible_infer == actual_counts)
            naive_correct += torch.sum(actual_counts == 0)
            correct += torch.sum(infer_count == actual_counts)
            correct_t2 += torch.sum(infer_count2 == actual_counts)
            correct_t3 += torch.sum(infer_count3 == actual_counts)
            count += batch_len
        ave_loss = loss_cumulative / len(val_dl)
        acc = correct / count
        acc_2 = correct_t2 / count
        acc_3 = correct_t3 / count
        naive_acc = naive_correct / count
        best_case_acc = best_case_correct / count
        with np.printoptions(precision=1, floatmode='fixed', suppress=True, linewidth=100):
            print_idx = 0 # 0 not good for cluster idx=8
            print(f"Actual dist: {dist[print_idx].cpu().numpy()}")
            print(f"Output dist: {model_output[print_idx].cpu().numpy()}")
            print(f"Diff: {(dist[print_idx] - model_output[print_idx]).cpu().numpy()}")
        print(
            "guessed:",
            infer_count,
            "actual:",
            actual_counts,
            sep="\n",
        )
        print(f"Accuracy (t: {thresholds[0]:.1f}): {acc:.4f}")
        print(f"Accuracy (t: {thresholds[1]:.1f}): {acc_2:.4f}")
        print(f"Accuracy (t: {thresholds[2]:.1f}): {acc_3:.4f}")
        print(f"Naive accuracy: {naive_acc:.4f}")
        print(f"Best case correct: {best_case_acc:.4f}")
        print(f"Loss (average --- total) : {ave_loss} --- {loss_cumulative}")


def main():
    setup_logging(logging.INFO)
    train()


if __name__ == "__main__":
    main()
