import torch
from typing import Union
import pathlib
import configargparse
import logging
import logging.handlers
import numpy as np
import retinapy
import retinapy.models
import retinapy.train
import retinapy.dataset
import spikedistancefield as sdf

LOG_FILENAME = "test.log"
DEFAULT_OUT_BASE_DIR = "./out/test"
BATCH_SIZE = 1
TEST_LEN = retinapy.train.VAL_LEN

p = configargparse.ArgumentParser(
    config_file_parser_class=configargparse.YAMLConfigFileParser
)
p.add("-c", "--config", is_config_file=True, help="config file path")
p.add(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to the checkpoint to test.",
)
opt = p.parse_known_args()[0]


def setup_logging(level):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    console_handler = logging.StreamHandler()
    root_logger.addHandler(console_handler)


def test():
    model = retinapy.train.model_fn()
    retinapy.train.load_model(model, opt.checkpoint)
    _, _, test_ds = retinapy.train.create_datasets()
    test_ds.enable_augmentation = False
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
    )
    model.eval()
    model.cuda()
    correct = 0
    naive_correct = 0
    count = 0
    with torch.no_grad():
        for (masked_snippet, target_spikes, dist) in test_dl:
            assert BATCH_SIZE == 1
            masked_snippet = masked_snippet.float()
            dist = dist.float().cuda()
            target_spikes = target_spikes[:, 0:TEST_LEN]
            model_output = model(masked_snippet.cuda())
            # Upscale
            model_output *= retinapy.train.DIST_NORM
            actual_count = torch.count_nonzero(target_spikes, dim=1)[0]
            # Set oldest timestamp to have a spike, in case there are none.
            # We need at least one to set lhs_spike.
            SPIKE_DIM = 4
            masked_snippet[:, SPIKE_DIM, 0] = 1
            _len = masked_snippet.shape[-1]
            lhs_spike = (
                torch.argmax(
                    torch.arange(_len) * (masked_snippet[0, SPIKE_DIM] == 1)
                )
                - _len
            )
            assert lhs_spike < 0
            # This is hacky. But what is better?
            rhs_spike = dist[0][-1] + _len
            _, infer_seq = sdf.mle_inference_from_df(
                model_output[0],
                lhs_spike,
                rhs_spike,
                spike_pad=5,
                max_clamp=retinapy.train.DIST_CLAMP,
                resolution=20
            )

            # We only care about half.
            infer_seq = [s for s in infer_seq if s < TEST_LEN]
            infer_count = len(infer_seq)
            correct += infer_count == actual_count
            naive_correct = torch.sum(actual_count == 0)
            count += 1
            if actual_count or infer_count:
                print(f"Actual count: {actual_count}")
                print(f"Infer count: {infer_count}")
                print("Actual:", torch.nonzero(target_spikes)[:,1].cpu().numpy(), sep='\n')
                print("Infer:", np.array(infer_seq), sep='\n')
            print(f"Accuracy (online): {correct/count:.4f}")
        naive_acc = naive_correct / count
        accuracy = correct / count
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Naive accuracy: {naive_acc:.4f}")


def main():
    setup_logging(logging.INFO)
    test()


if __name__ == "__main__":
    main()
